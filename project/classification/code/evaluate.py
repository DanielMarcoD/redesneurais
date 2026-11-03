"""Evaluate a fully frozen experiment once; render using saved predictions only."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from train import (ROOT, PROJECT, eda, digest, read_json, save_json, publish, source_hashes,
                   seed_all, build_model, scores, metrics)


def verify_frozen(run):
    frozen=read_json(run/'frozen.json')
    if frozen['protocol_sha256'] != digest(run/'protocol.json') or frozen['sources'] != source_hashes():
        raise ValueError('Protocol or experiment source changed after freezing')
    for name, expected in frozen['files'].items():
        if digest(run/name) != expected:
            raise ValueError(f'Frozen artifact changed: {name}')
    return frozen


def evaluate_frozen(run):
    frozen=verify_frozen(run)
    if (run/'test_predictions.csv').exists():
        print('Existing final predictions retained; use render. No new inference.')
        return
    protocol=read_json(run/'protocol.json')
    if digest(eda.DATA_PATH) != protocol['csv_sha256']:
        raise ValueError('CSV changed')
    ids=protocol['indices']['test']; wanted=set(ids)
    raw=pd.read_csv(eda.DATA_PATH,skiprows=lambda row:row>0 and row-1 not in wanted)
    raw.index=ids
    y=raw.Revenue.astype(int).to_numpy()
    x=joblib.load(run/'preprocessor.joblib').transform(raw.drop(columns='Revenue'))
    models=joblib.load(run/'baselines.joblib')
    predictions=[]; results={}
    seed_all(42)
    for name, model in models.items():
        s=model.predict_proba(x)[:,1]; t=frozen['baselines'][name]['threshold']
        results[name]=metrics(y,s,t)
        predictions.append(pd.DataFrame(dict(row_id=ids,model=name,seed=0,target=y,score=s,
                                              threshold=t,prediction=(s>=t).astype(int),VisitorType=raw.VisitorType.to_numpy())))
    for info in frozen['networks']:
        seed=info['seed']; name=f'MLP_{seed}'
        model=build_model(frozen['input_dim'],frozen['config'])
        model.load_state_dict(torch.load(run/f'final_{seed}.pt',weights_only=True,map_location='cpu'))
        s=scores(model,x); t=info['threshold']
        results[name]=metrics(y,s,t)
        predictions.append(pd.DataFrame(dict(row_id=ids,model=name,seed=seed,target=y,score=s,
                                              threshold=t,prediction=(s>=t).astype(int),VisitorType=raw.VisitorType.to_numpy())))
    df=pd.concat(predictions,ignore_index=True)
    df.to_csv(run/'test_predictions.csv',index=False,float_format='%.17g')
    aggregate={}
    for key in ['macro_f1','accuracy','roc_auc','pr_auc','average_precision','precision','recall','f1']:
        values=np.asarray([results[f'MLP_{s}'][key] for s in frozen['seeds']])
        aggregate[key]=dict(mean=np.mean(values,axis=0).tolist(),std=np.std(values,axis=0,ddof=1).tolist())
    best_baseline=max(models,key=lambda n:results[n]['macro_f1'])
    delta=aggregate['macro_f1']['mean']-results[best_baseline]['macro_f1']
    groups=[]
    for (name,group), sub in df.groupby(['model','VisitorType']):
        yy=sub.target.to_numpy(); pp=sub.prediction.to_numpy()
        fp=int(((yy==0)&(pp==1)).sum()); fn=int(((yy==1)&(pp==0)).sum())
        tp=int(((yy==1)&(pp==1)).sum())
        groups.append(dict(model=name,group=group,n=len(sub),positives=int(yy.sum()),fp=fp,fn=fn,
                           error=float((fp+fn)/len(sub)),precision=tp/(tp+fp) if tp+fp else 0.,
                           recall=tp/(tp+fn) if tp+fn else 0.))
    all_results=read_json(run/'results.json')
    all_results['test']=dict(models=results,aggregate=aggregate,best_baseline=best_baseline,delta=delta,
                             success=bool(delta>=protocol['success_margin']),subgroups=groups,
                             frozen_sha256=digest(run/'frozen.json'),predictions_sha256=digest(run/'test_predictions.csv'))
    save_json(run/'results.json',all_results)
    publish(run)
    print('Final evaluation:',aggregate['macro_f1'],'delta',delta,'success',delta>=.02,flush=True)


def render(run):
    results=read_json(run/'results.json'); frozen=read_json(run/'frozen.json')
    df=pd.read_csv(run/'test_predictions.csv',float_precision='round_trip')
    assert digest(run/'frozen.json')==results['test']['frozen_sha256']
    assert digest(run/'test_predictions.csv')==results['test']['predictions_sha256']
    for name, sub in df.groupby('model'):
        measured=metrics(sub.target,sub.score.to_numpy(),float(sub.threshold.iloc[0]))
        for key in measured:
            np.testing.assert_allclose(measured[key],results['test']['models'][name][key],rtol=1e-12,atol=1e-12)
    out=PROJECT/'figures' if run.resolve()==(ROOT/'.cache/classification').resolve() else run/'figures'
    out.mkdir(parents=True,exist_ok=True)
    def save(fig,n,title):
        fig.suptitle(f'Figura {n} — {title}')
        fig.tight_layout()
        fig.savefig(out/f'figure_{n:02d}.png',dpi=150,bbox_inches='tight')
        plt.close(fig)
    history=pd.read_csv(run/'history.csv')
    chosen=frozen['config']['id']
    cases=list(dict.fromkeys(['C1','C2','C3','C4',chosen]))
    fig,axes=plt.subplots(1,len(cases),figsize=(4*len(cases),3.6),squeeze=False)
    for ax,c in zip(axes.flat,cases):
        h=history[(history.stage=='search')&(history.config==c)]
        for col,label in [('train_loss','Treino'),('validation_loss','Validação')]: ax.plot(h.epoch,h[col],label=label)
        ax.set(title=c,xlabel='Época',ylabel='BCE ponderada'); ax.legend()
    save(fig,1,'Comparação controlada de regularização')
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    for ax,info in zip(axes,frozen['networks']):
        h=history[(history.stage=='final')&(history.seed==info['seed'])]
        ax.plot(h.epoch,h.train_loss,label='Treino'); ax.plot(h.epoch,h.validation_loss,label='Validação')
        ax.axvline(info['best_epoch'],linestyle='--',color='gray',label='Época restaurada')
        ax.set(title=f"Seed {info['seed']}",xlabel='Época',ylabel='BCE ponderada'); ax.legend()
    save(fig,2,'Históricos dos retreinamentos finais')
    fig,ax=plt.subplots(figsize=(8,4))
    for name,info in [('Logistic',frozen['baselines']['Logistic'])]+[(f"MLP {v['seed']}",v) for v in frozen['networks']]:
        curve=np.asarray(info['threshold_curve']); ax.plot(curve[:,0],curve[:,1],label=f"{name}; t={info['threshold']:.2f}")
    ax.set(xlabel='Limiar',ylabel='Macro-F1 de validação'); ax.legend()
    save(fig,3,'Limiares selecionados exclusivamente na validação')
    fig,axes=plt.subplots(1,5,figsize=(17,3.6))
    for ax,(name,info) in zip(axes,results['test']['models'].items()):
        cm=np.asarray(info['confusion']); ax.imshow(cm,cmap='Blues')
        for i in range(2):
            for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='red')
        ax.set(title=name,xlabel='Classe prevista',ylabel='Classe real',xticks=[0,1],yticks=[0,1])
    save(fig,4,'Matrizes de confusão no teste; 0 = sem compra, 1 = com compra')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name,sub in df.groupby('model'):
        fpr,tpr,_=roc_curve(sub.target,sub.score); p,r,_=precision_recall_curve(sub.target,sub.score)
        axes[0].plot(fpr,tpr,label=name); axes[1].plot(r,p,label=name)
    axes[0].plot([0,1],[0,1],'k--',label='Acaso')
    axes[1].axhline(df.target.mean(),color='gray',linestyle='--',label='Proporção de compras')
    axes[0].set(xlabel='Taxa de falsos positivos',ylabel='Taxa de verdadeiros positivos',title='ROC')
    axes[1].set(xlabel='Revocação',ylabel='Precisão',title='Precisão–revocação')
    for ax in axes:ax.legend()
    save(fig,5,'Desempenho de ordenação no teste')
    subgroup=pd.DataFrame(results['test']['subgroups']); net=subgroup[subgroup.model.str.startswith('MLP')]
    grouped=net.groupby('group').agg(error=('error','mean'),std=('error','std'),n=('n','first'))
    fig,ax=plt.subplots(figsize=(8,4))
    ax.bar([f'{g}\n(n={int(r.n)})' for g,r in grouped.iterrows()],grouped.error,yerr=grouped['std'],capsize=5,label='MLP: média ± desvio amostral')
    ax.set(xlabel='Tipo de visitante',ylabel='Taxa de erro');ax.legend()
    save(fig,6,'Erros no teste por subgrupo predefinido')
    imp=pd.DataFrame(results['permutation']).iloc[::-1]
    fig,ax=plt.subplots(figsize=(9,6))
    ax.barh(imp.feature,imp['mean'],xerr=imp['std'],label='10 permutações: média ± desvio amostral')
    ax.axvline(0,color='gray');ax.set(xlabel='Queda no Macro-F1 de validação',ylabel='Atributo original');ax.legend()
    save(fig,7,'Importância por permutação; seed 101, apenas validação')
    print('Verified saved metrics and rendered 7 figures',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['run','render'])
    parser.add_argument('--run-dir',type=Path,default=ROOT/'.cache/classification')
    args=parser.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        (evaluate_frozen if args.stage=='run' else render)(args.run_dir.resolve())

if __name__=='__main__':main()
