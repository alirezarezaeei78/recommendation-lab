"""Plot only the executed aggregate results; seed ranges are not CIs."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MultipleLocator
import numpy as np


def main():
    source=Path('results/2026-10-23.json')
    data=json.loads(source.read_text())
    assert data['status']=='completed' and len(data['bpr_runs'])==3
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':19,
        'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False})
    fig,axes=plt.subplots(3,1,figsize=(8.4,11.2))
    colors=['#758497','#c16b13','#51449a']
    names=['Popularity','Item-kNN','BPR-MF']
    for ax,(metric,title,limit) in zip(axes,[('ndcg_at_10','NDCG@10',.23),
        ('recall_at_10','Recall@10',.18),('catalog_coverage','Catalog coverage',.32)]):
        values=[data['baselines'][m]['warm_primary'][metric] for m in ['popularity','item_knn']]
        summary=data['bpr_seed_summary'][metric]
        values.append(summary['mean'])
        ax.barh([2,1,0],values,color=colors,height=.58,zorder=2)
        for i,v in enumerate(values):
            label=f'{v:.4f}' if metric!='catalog_coverage' else f'{100*v:.2f}%'
            label_x=(summary['max']+.007) if i==2 else v+.003
            ax.text(label_x,2-i,label,va='center',fontsize=20,fontweight='bold')
        for offset,marker,run in zip([-.14,0,.14],['o','D','^'],data['bpr_runs']):
            ax.scatter(run['warm_primary'][metric],offset,s=65,marker=marker,
                color='white',edgecolor='#182f48',linewidth=1.2,zorder=5)
        ax.set_yticks([2,1,0],names,fontsize=20)
        ax.tick_params(axis='y',length=0,pad=8)
        ax.set_xlim(0,limit);ax.set_ylim(-.55,2.55)
        ax.set_title(title+'  ↑',loc='left',fontsize=23,fontweight='bold',pad=12)
        ax.grid(axis='x',color='#dce3eb',zorder=0)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(MultipleLocator(.1 if metric=='catalog_coverage' else .05))
        if metric=='catalog_coverage':ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    fig.text(.035,.969,'MovieLens 100K · 95 eligible warm users',fontsize=21,fontweight='bold')
    fig.text(.035,.934,'Same full training-catalog candidates; seen items excluded',fontsize=14)
    fig.text(.035,.072,'BPR bars: three-seed mean. Markers: 13 ○ / 29 ◇ / 47 △.',fontsize=14)
    fig.text(.035,.045,'Seed variation is not a confidence interval.',fontsize=14)
    fig.text(.035,.019,'Exploratory reuse of the 19 Oct split · GroupLens research data',fontsize=13)
    fig.subplots_adjust(left=.23,right=.91,top=.88,bottom=.13,hspace=.62)
    for extension in ['png','svg']:
        fig.savefig(f'results/bpr-comparison.{extension}',dpi=180,facecolor='white')
    svg=Path('results/bpr-comparison.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)
    alt=("Three bar charts compare the same 95 warm users in the exposed chronological MovieLens100K task. "
         "NDCG at10: popularity0.158664, item-kNN0.134159, BPR mean0.170954 across seeds13,29,47 "
         "with range0.168067 to0.173222. Recall at10: popularity0.106064, item-kNN0.073715, "
         "BPR mean0.114578. Catalog coverage:4.58%,14.60%,21.60% respectively. "
         "BPR markers show all three seeds; they are not confidence intervals. All axes begin at zero. "
         "This is exploratory offline benchmark reuse, not an adoption or causal result.")
    Path('results/bpr-alt-text.txt').write_text(alt+'\n',encoding='utf-8')


if __name__=='__main__':main()
