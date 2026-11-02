"""Original measured MMR figure; reads saved aggregates without inventing points."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('experiments/2026-11-02');data=json.loads((root/'results.json').read_text())
    assert data['status']=='completed'
    rows=data['across_seed_descriptive'];labels=[str(r['lambda']) for r in rows]
    navy='#15324b';teal='#087f8c';amber='#d89427';muted='#4c6170'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':15,'text.color':navy,'axes.labelcolor':navy,
        'xtick.color':muted,'ytick.color':navy,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(12,14),dpi=120,facecolor='white')
    fig.text(.065,.956,'Relevance and genre diversity trade off',fontsize=24,weight='bold')
    fig.text(.065,.918,'95 warm users  |  3 frozen BPR seeds  |  same unseen candidates',fontsize=16)
    fig.text(.065,.89,'MovieLens100K · known chronological split · exploratory reuse',fontsize=14,color=muted)
    for position,metric,title,color,xmax in [([.13,.57,.33,.25],'ndcg_at_10','NDCG@10',teal,.22),([.6,.57,.33,.25],'intra_list_diversity','Genre diversity (ILD)',amber,1.18)]:
        ax=fig.add_axes(position);values=np.array([r[metric]['mean'] for r in rows]);low=np.array([r[metric]['min'] for r in rows]);high=np.array([r[metric]['max'] for r in rows]);y=np.arange(5)
        ax.barh(y,values,color=color,height=.6)
        ax.errorbar(values,y,xerr=[values-low,high-values],fmt='none',ecolor=navy,capsize=3,lw=1.5)
        ax.set_yticks(y,labels if metric=='ndcg_at_10' else []);ax.invert_yaxis();ax.set_xlim(0,xmax)
        ax.set_title(title,fontsize=19,weight='bold',pad=18);ax.set_xlabel('Higher is larger; separate scales',fontsize=12)
        if metric=='ndcg_at_10':ax.set_ylabel('Relevance weight λ',fontsize=15)
        for j,value in enumerate(values):ax.text(high[j]+xmax*.025,j,f'{value:.3f}',va='center',fontsize=17,weight='bold')
        ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    ax=fig.add_axes([.06,.24,.88,.24]);ax.axis('off')
    cells=[]
    for r in rows:
        v=r['lambda'];per_seed=[next(c for c in run['conditions'] if c['lambda']==v)['warm_primary'] for run in data['runs']]
        cells.append([str(v),f"{r['ndcg_at_10']['mean']:.4f}",f"{r['recall_at_10']['mean']:.4f}",f"{r['intra_list_diversity']['mean']:.4f}",f"{100*r['catalog_coverage']['mean']:.1f}%"])
    table=ax.table(cellText=cells,colLabels=['λ','NDCG@10','Recall@10','Genre ILD','Coverage'],loc='center',cellLoc='center',colLoc='center',bbox=[0,0,1,1])
    table.auto_set_font_size(False);table.set_fontsize(18)
    for (i,j),cell in table.get_celld().items():
        cell.set_edgecolor('white')
        if i==0:cell.set_facecolor(navy);cell.set_text_props(color='white',weight='bold')
        else:cell.set_facecolor('#edf5f6' if i%2 else '#f7f9fb')
    fig.text(.065,.192,'Bars/table: seed means. Whiskers: min–max across 3 seeds, not CIs.',fontsize=13)
    fig.text(.065,.163,'ILD = mean pairwise cosine distance of binary genre vectors.',fontsize=14)
    fig.text(.065,.135,'Coverage = distinct top-10 items / 1,616-item training catalog, per seed.',fontsize=13)
    fig.text(.065,.103,'A diversity proxy is not user satisfaction. No “best λ” selected.',fontsize=15,weight='bold')
    fig.text(.065,.066,'Data: GroupLens; Harper & Konstan (2015), DOI 10.1145/2827872.',fontsize=12,color=muted)
    fig.text(.065,.043,'MMR adapted from Carbonell & Goldstein (SIGIR 1998).',fontsize=12,color=muted)
    for ext in ['png','svg']:
        target=root/f'diversity-tradeoff.{ext}'
        fig.savefig(target,facecolor='white')
        if ext=='svg':
            target.write_bytes(('\n'.join(line.rstrip() for line in target.read_text(encoding='utf-8').splitlines())+'\n').encode())


if __name__=='__main__':main()
