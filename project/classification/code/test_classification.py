"""Tests use training/validation and synthetic examples, never test inference."""
import copy
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from train import (ROOT, CONFIGS, TEST_HASH, read_json, load_partition, eda,
                   build_model, train_one, seed_all, choose_threshold, metrics)

class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol=read_json(ROOT/'.cache/classification/protocol.json')

    def test_partition_integrity(self):
        p=self.protocol
        self.assertEqual(p['index_hashes']['test'],TEST_HASH)
        self.assertEqual([len(p['indices'][k]) for k in ['train','validation','test']],[7891,1973,2466])
        ids=p['indices']
        self.assertEqual(len(set(sum(ids.values(),[]))),12330)
        with self.assertRaises(ValueError):load_partition(p,'test')
        a,_=load_partition(p,'train');b,_=load_partition(p,'validation')
        ga=eda.pd.util.hash_pandas_object(a[eda.RETAINED],index=False)
        gb=eda.pd.util.hash_pandas_object(b[eda.RETAINED],index=False)
        self.assertTrue(set(ga).isdisjoint(gb))

    def test_preprocessor_train_only(self):
        x,_=load_partition(self.protocol,'train');v,_=load_partition(self.protocol,'validation')
        pre=eda.build_preprocessor();t=pre.fit_transform(x)
        vt,names=eda.validate_preprocessor(pre,x,v,t)
        self.assertEqual(t.shape,(7891,74));self.assertEqual(vt.shape,(1973,74))
        self.assertFalse(any('Revenue' in n or 'PageValues' in n for n in names))

    def test_model_gradients_and_parameters(self):
        seed_all(42);model=build_model(74,CONFIGS[0])
        self.assertEqual(sum(p.numel() for p in model.parameters()),2433)
        x=torch.randn(8,74);y=torch.tensor([0.,1.]*4)
        before=copy.deepcopy(model.state_dict());opt=torch.optim.Adam(model.parameters())
        loss=torch.nn.BCEWithLogitsLoss()(model(x).squeeze(1),y)
        loss.backward();opt.step()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertTrue(any(not torch.equal(v,before[k]) for k,v in model.state_dict().items()))

    def test_restoration_and_determinism(self):
        rng=np.random.default_rng(42);x=rng.normal(size=(48,4));y=(x[:,0]>0).astype(int)
        p=dict(self.protocol,max_epochs=10,patience=3,batch_size=16)
        first=train_one(x[:32],y[:32],x[32:],y[32:],CONFIGS[2],101,p)
        second=train_one(x[:32],y[:32],x[32:],y[32:],CONFIGS[2],101,p)
        self.assertEqual(first[1],second[1]);self.assertEqual(first[2],second[2])
        self.assertEqual(first[1]['best_loss'],min(h['validation_loss'] for h in first[2]))
        self.assertFalse(first[0].training)

    def test_frozen_protocol_tampering_blocks_test_access(self):
        from evaluate import evaluate_frozen
        source=ROOT/'.cache/classification'
        if not (source/'frozen.json').exists():
            self.skipTest('Freeze guard is tested again after freeze.')
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)
            shutil.copy2(source/'frozen.json',run/'frozen.json')
            (run/'protocol.json').write_text('{}')
            with patch('evaluate.pd.read_csv') as read_csv:
                with self.assertRaisesRegex(ValueError,'Protocol or experiment source changed'):
                    evaluate_frozen(run)
                read_csv.assert_not_called()

    def test_completed_evaluation_does_not_read_test_again(self):
        from evaluate import evaluate_frozen
        run=ROOT/'.cache/classification'
        if not (run/'test_predictions.csv').exists():
            self.skipTest('Idempotence is tested again after final evaluation.')
        with patch('evaluate.pd.read_csv') as read_csv:
            evaluate_frozen(run)
            read_csv.assert_not_called()

    def test_final_predictions_independent_count_audit(self):
        import pandas as pd
        run=ROOT/'.cache/classification'
        if not (run/'test_predictions.csv').exists():
            self.skipTest('Final counts are audited only after evaluation.')
        data=pd.read_csv(run/'test_predictions.csv',float_precision='round_trip')
        recorded=read_json(run/'results.json')['test']
        for name, frame in data.groupby('model'):
            self.assertEqual(frame.row_id.tolist(),self.protocol['indices']['test'])
            np.testing.assert_array_equal(frame.prediction,(frame.score>=frame.threshold).astype(int))
            y=frame.target.to_numpy();pred=frame.prediction.to_numpy()
            tn=int(((y==0)&(pred==0)).sum());fp=int(((y==0)&(pred==1)).sum())
            fn=int(((y==1)&(pred==0)).sum());tp=int(((y==1)&(pred==1)).sum())
            expected=recorded['models'][name]
            self.assertEqual([[tn,fp],[fn,tp]],expected['confusion'])
            manual=(2*tn/(2*tn+fp+fn)+2*tp/(2*tp+fp+fn))/2
            self.assertAlmostEqual(manual,expected['macro_f1'],places=14)
        values=[recorded['models'][f'MLP_{seed}']['macro_f1'] for seed in [101,202,303]]
        self.assertAlmostEqual(float(np.std(values,ddof=1)),recorded['aggregate']['macro_f1']['std'],places=14)
        self.assertEqual(recorded['success'],recorded['delta']>=.02)

    def test_threshold_and_metrics(self):
        y=np.array([0,0,1,1]);s=np.array([.1,.2,.8,.9])
        t,_=choose_threshold(y,s)
        self.assertEqual(t,.5);self.assertEqual(metrics(y,s,t)['macro_f1'],1.)
        dummy=metrics(y,np.zeros(4),.5)
        self.assertAlmostEqual(dummy['macro_f1'],1/3)
        self.assertEqual(dummy['confusion'],[[2,0],[2,0]])

if __name__=='__main__':unittest.main()
