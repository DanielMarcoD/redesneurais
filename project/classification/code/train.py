"""CPU experiment. Only prepare reconstructs the legacy split; no test selection."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import random
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             precision_recall_curve, precision_recall_fscore_support,
                             roc_auc_score, average_precision_score)
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[2]
sys.path.insert(0, str(PROJECT.parent / 'eda/code'))
import eda

CONFIGS = [dict(id=f'C{i+1}', hidden=h, lr=lr, weight_decay=w, dropout=p)
           for i, (h, lr, w, p) in enumerate([
               ([32], .001, 0., 0.), ([32], .001, .001, 0.),
               ([32], .001, 0., .2), ([32], .001, .001, .2),
               ([32], .0003, 0., 0.), ([32], .003, 0., 0.),
               ([64], .001, 0., 0.), ([32, 16], .001, 0., 0.)])]
TEST_HASH = 'c89e8f23df3e55bc5af85208983c2ffca78e0a3b7264fbcc72a5347f481c9f87'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def read_json(path):
    return json.loads(Path(path).read_text())


def publish(run):
    # Alternative run directories never overwrite the published experiment.
    if run.resolve() == (ROOT / '.cache/classification').resolve():
        for name in ['protocol.json', 'frozen.json', 'results.json', 'history.csv', 'test_predictions.csv']:
            if (run / name).exists():
                (PROJECT / 'artifacts' / name).write_bytes((run / name).read_bytes())


def source_hashes():
    return {str(p.relative_to(ROOT)): digest(p) for p in
            [Path(__file__), PROJECT / 'code/evaluate.py', Path(eda.__file__)]}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def load_partition(protocol, name):
    if name not in ('train', 'validation'):
        raise ValueError('Training code cannot load test rows.')
    ids = protocol['indices'][name]
    wanted = set(ids)
    if digest(eda.DATA_PATH) != protocol['csv_sha256']:
        raise ValueError('Dataset identity changed')
    df = pd.read_csv(eda.DATA_PATH, skiprows=lambda row: row > 0 and row - 1 not in wanted)
    df.index = ids
    return df.drop(columns='Revenue'), df.Revenue.astype(int).to_numpy()


def prepare(run):
    if (run / 'protocol.json').exists():
        raise FileExistsError('Use a new --run-dir; never overwrite a registered protocol.')
    data = eda.load_data()
    x, xtest, y, _ = eda.split_data(data)
    groups = pd.util.hash_pandas_object(x[eda.RETAINED], index=False)
    a, b = next(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(x, y, groups))
    indices = {'train': x.iloc[a].index.tolist(), 'validation': x.iloc[b].index.tolist(),
               'test': xtest.index.tolist()}
    hashes = {k: hashlib.sha256(np.asarray(v, dtype='<i8').tobytes()).hexdigest()
              for k, v in indices.items()}
    assert hashes['test'] == TEST_HASH
    all_groups = pd.util.hash_pandas_object(data[eda.RETAINED], index=False)
    for left, right in [('train','validation'), ('train','test'), ('validation','test')]:
        assert set(indices[left]).isdisjoint(indices[right])
        assert set(all_groups.iloc[indices[left]]).isdisjoint(all_groups.iloc[indices[right]])
    assert sorted(sum(indices.values(), [])) == list(range(len(data)))
    protocol = dict(indices=indices, index_hashes=hashes, csv_sha256=eda.CSV_SHA256,
                    split_seed=42, search_seed=42, final_seeds=[101,202,303],
                    configs=CONFIGS, batch_size=128, max_epochs=200, patience=20,
                    pos_weight=float((~y.iloc[a]).sum()/y.iloc[a].sum()),
                    threshold_grid=[i/100 for i in range(1,100)],
                    threshold_tie='closest to 0.5, then largest',
                    ranking='macro_f1 descending, parameters, best_loss, id ascending',
                    success='mean MLP test macro-F1 - max(Dummy, Logistic test macro-F1) >= 0.02',
                    success_margin=.02, subgroup='VisitorType', example_seed=101,
                    permutation_seed=42, permutation_repeats=10,
                    rubric_url='https://insper.github.io/ann-dl/2026.2/projects/supervised/',
                    rubric_checked='2026-11-03',
                    versions={p:importlib.metadata.version(p) for p in
                              ['numpy','pandas','scikit-learn','torch','matplotlib','mkdocs']})
    run.mkdir(parents=True, exist_ok=True)
    save_json(run/'protocol.json', protocol)
    xtr, _ = load_partition(protocol, 'train')
    xv, _ = load_partition(protocol, 'validation')
    prep = eda.build_preprocessor()
    transformed = prep.fit_transform(xtr)
    eda.validate_preprocessor(prep, xtr, xv, transformed)
    joblib.dump(prep, run/'preprocessor.joblib')
    publish(run)
    print('Prepared', {k:len(v) for k,v in indices.items()}, 'features', transformed.shape[1], flush=True)


def metrics(y, score, threshold):
    pred = (np.asarray(score) >= threshold).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=[0,1], zero_division=0)
    p, r, _ = precision_recall_curve(y, score)
    return dict(macro_f1=float(f1.mean()), accuracy=float(accuracy_score(y,pred)),
                roc_auc=float(roc_auc_score(y,score)), pr_auc=float(auc(r,p)),
                average_precision=float(average_precision_score(y,score)),
                precision=precision.tolist(), recall=recall.tolist(), f1=f1.tolist(),
                support=support.tolist(), confusion=confusion_matrix(y,pred,labels=[0,1]).tolist())


def choose_threshold(y, scores, grid=None):
    grid = grid if grid is not None else [i/100 for i in range(1,100)]
    values = [(float(t), float(f1_score(y, scores >= t, average='macro', labels=[0,1], zero_division=0))) for t in grid]
    t, _ = min(values, key=lambda pair: (-pair[1], abs(pair[0]-.5), -pair[0]))
    return t, values


def build_model(d, config):
    layers = []
    for width in config['hidden']:
        linear = nn.Linear(d,width)
        nn.init.kaiming_normal_(linear.weight, nonlinearity='relu')
        nn.init.zeros_(linear.bias)
        layers.extend([linear, nn.ReLU(), nn.Dropout(config['dropout'])])
        d = width
    output = nn.Linear(d,1)
    nn.init.xavier_uniform_(output.weight)
    nn.init.zeros_(output.bias)
    return nn.Sequential(*layers, output)


def scores(model, x):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.as_tensor(np.asarray(x),dtype=torch.float32)).squeeze(1)).numpy()


def train_one(x, y, xv, yv, config, seed, protocol):
    seed_all(seed)
    model = build_model(x.shape[1],config)
    opt = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(protocol['pos_weight'],dtype=torch.float32))
    tx, ty = torch.tensor(x,dtype=torch.float32), torch.tensor(y,dtype=torch.float32)
    vx, vy = torch.tensor(xv,dtype=torch.float32), torch.tensor(yv,dtype=torch.float32)
    loader = DataLoader(TensorDataset(tx,ty), batch_size=protocol['batch_size'], shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    best, wait, best_epoch, state = float('inf'), 0, 0, None
    history = []
    for epoch in range(1,protocol['max_epochs']+1):
        model.train()
        for xb,yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb).squeeze(1),yb)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            tr_logits, va_logits = model(tx).squeeze(1), model(vx).squeeze(1)
            tr_loss, va_loss = loss_fn(tr_logits,ty).item(), loss_fn(va_logits,vy).item()
        history.append(dict(epoch=epoch, train_loss=tr_loss, validation_loss=va_loss,
                            train_f1=f1_score(y,tr_logits.numpy()>=0,average='macro',zero_division=0),
                            validation_f1=f1_score(yv,va_logits.numpy()>=0,average='macro',zero_division=0)))
        if va_loss < best:
            best, wait, best_epoch, state = va_loss, 0, epoch, copy.deepcopy(model.state_dict())
        else:
            wait += 1
        if wait >= protocol['patience']:
            break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        restored = loss_fn(model(vx).squeeze(1),vy).item()
    assert restored == best
    val_scores = scores(model,xv)
    threshold, curve = choose_threshold(yv,val_scores,protocol['threshold_grid'])
    result = dict(config=config, seed=seed, parameters=sum(p.numel() for p in model.parameters()),
                  best_epoch=best_epoch, stopped_epoch=epoch, best_loss=best, restored_loss=restored,
                  threshold=threshold, threshold_curve=curve,
                  validation=metrics(yv,val_scores,threshold), at_half=metrics(yv,val_scores,.5))
    return model, result, history


def internal_data(run):
    protocol = read_json(run/'protocol.json')
    prep = joblib.load(run/'preprocessor.joblib')
    x,y = load_partition(protocol,'train')
    xv,yv = load_partition(protocol,'validation')
    return protocol, prep, x, y, xv, yv, prep.transform(x), prep.transform(xv)


def fit_baselines(x,y,xv,yv,protocol):
    models = {'Dummy': DummyClassifier(strategy='most_frequent'),
              'Logistic': LogisticRegression(solver='lbfgs', C=1., max_iter=2000, tol=1e-4,
                                             class_weight={0:1,1:protocol['pos_weight']})}
    results = {}
    for name, model in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model.fit(x,y)
        s = model.predict_proba(xv)[:,1]
        threshold, curve = (.5, []) if name=='Dummy' else choose_threshold(yv,s,protocol['threshold_grid'])
        results[name] = dict(threshold=threshold, threshold_curve=curve,
                             validation=metrics(yv,s,threshold), at_half=metrics(yv,s,.5))
    return models, results


def search(run):
    if (run/'results.json').exists():
        raise FileExistsError('Search already exists; use a new run directory for reproduction.')
    protocol, _, _, y, _, yv, x, xv = internal_data(run)
    seed_all(42)
    _, baselines = fit_baselines(x,y,xv,yv,protocol)
    results, histories = [], []
    for config in protocol['configs']:
        model, result, history = train_one(x,y,xv,yv,config,42,protocol)
        torch.save(model.state_dict(),run/f"search_{config['id']}.pt")
        results.append(result)
        histories.extend(dict(stage='search',config=config['id'],seed=42,**h) for h in history)
        print(config['id'],result['validation']['macro_f1'],result['best_epoch'],flush=True)
    results.sort(key=lambda r:(-r['validation']['macro_f1'],r['parameters'],r['best_loss'],r['config']['id']))
    save_json(run/'results.json',dict(baselines_validation=baselines,search=results))
    pd.DataFrame(histories).to_csv(run/'history.csv',index=False)
    publish(run)


def freeze(run):
    if (run/'frozen.json').exists():
        raise FileExistsError('Frozen experiment already exists')
    protocol, prep, _, y, raw_val, yv, x, xv = internal_data(run)
    results = read_json(run/'results.json')
    config = results['search'][0]['config']
    finals, histories = [], []
    for seed in protocol['final_seeds']:
        model, result, history = train_one(x,y,xv,yv,config,seed,protocol)
        torch.save(model.state_dict(),run/f'final_{seed}.pt')
        finals.append(result)
        histories.extend(dict(stage='final',config=config['id'],seed=seed,**h) for h in history)
        if seed == protocol['example_seed']:
            reference_model = model
        print('Retrained',seed,result['validation']['macro_f1'],flush=True)
    models, baseline_results = fit_baselines(x,y,xv,yv,protocol)
    joblib.dump(models,run/'baselines.joblib')
    reference = next(r for r in finals if r['seed']==protocol['example_seed'])
    rng = np.random.default_rng(protocol['permutation_seed'])
    importance = []
    for feature in eda.RETAINED:
        drops = []
        for _ in range(protocol['permutation_repeats']):
            shuffled = raw_val.copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            pred = scores(reference_model,prep.transform(shuffled)) >= reference['threshold']
            drops.append(reference['validation']['macro_f1']-f1_score(yv,pred,average='macro',zero_division=0))
        importance.append(dict(feature=feature,mean=float(np.mean(drops)),std=float(np.std(drops,ddof=1)),drops=drops))
    results.update(final_validation=finals,baselines_final_validation=baseline_results,
                   permutation=sorted(importance,key=lambda r:-r['mean']))
    save_json(run/'results.json',results)
    pd.concat([pd.read_csv(run/'history.csv'),pd.DataFrame(histories)],ignore_index=True).to_csv(run/'history.csv',index=False)
    filenames = ['preprocessor.joblib','baselines.joblib']+[f'final_{s}.pt' for s in protocol['final_seeds']]
    frozen = dict(protocol_sha256=digest(run/'protocol.json'),sources=source_hashes(),
                  files={f:digest(run/f) for f in filenames},config=config,input_dim=x.shape[1],
                  seeds=protocol['final_seeds'],networks=finals,baselines=baseline_results,
                  criterion=protocol['success'],versions=protocol['versions'])
    save_json(run/'frozen.json',frozen)
    publish(run)
    print('FROZEN; test has not been evaluated.',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','search','freeze'])
    parser.add_argument('--run-dir',type=Path,default=ROOT/'.cache/classification')
    args=parser.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        globals()[args.stage](args.run_dir.resolve())

if __name__=='__main__':
    main()
