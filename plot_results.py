"""Plots saved aggregates only; no hand-entered benchmark curves."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__=='__main__':
    r=json.loads(Path('results/2026-10-19.json').read_text())
    plt.rcParams.update({'font.size':14,'axes.labelsize':15,'xtick.labelsize':13,'ytick.labelsize':12})
    fig,grid=plt.subplots(2,2,figsize=(8,9),layout='constrained');axs=grid.ravel()
    methods=['popularity','item_knn'];labels=['Popularity','Item-kNN']
    for ax,metric,label in zip(axs,['ndcg_at_10','recall_at_10','catalog_coverage'],['NDCG@10','Recall@10','Catalog coverage']):
        values=[r['results'][m]['warm_primary'][metric] for m in methods]
        bars=ax.bar(labels,values,color=['#527b9c','#cf8624'],width=.58)
        ax.set_ylim(0,max(values)*1.28 if max(values)>0 else 1);ax.set_ylabel(label);ax.spines[['top','right']].set_visible(False)
        for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height(),f'{value:.4f}',ha='center',va='bottom',fontsize=15)
    n=r['eligible_warm_users'];cold=r['eligible_cold_users']
    axs[3].axis('off')
    axs[3].text(.03,.85,f'Primary cohort: {n} warm users\n\n{cold} eligible cold users\nexcluded from primary.\n\n1,616 training-catalog items\nFull candidates; seen excluded.\n\n20 neighbors; shrunk cosine.\nSingle offline experiment;\nno online-impact claim.',va='top',fontsize=14,linespacing=1.5)
    fig.suptitle('Same chronological task · top10',fontsize=20,weight='bold')
    fig.supxlabel('MovieLens 100K · GroupLens Research\nCutoff: 1998-03-07 · rating ≥ 4 relevance\n192 cold users included separately in saved secondary results.',fontsize=12)
    fig.savefig('results/ranking-comparison.png',dpi=180);fig.savefig('results/ranking-comparison.svg');plt.close(fig)
    svg=Path('results/ranking-comparison.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
