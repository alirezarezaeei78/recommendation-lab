"""Measured figures from saved aggregates; seed ranges are not confidence intervals."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

result = json.loads(Path('results/2026-10-26.json').read_text())
assert result['status'] == 'completed'
plt.rcParams.update({'font.size':15,'axes.spines.top':False,'axes.spines.right':False,
    'axes.labelcolor':'#162c45','text.color':'#162c45','font.family':'DejaVu Sans'})
colors={'popularity':'#697684','item_knn':'#aa6a15','factor_centroid':'#3568ab','history_auto':'#087d7b'}
names={'popularity':'Popularity','item_knn':'Item-kNN','factor_centroid':'Factor centroid','history_auto':'Sparse-history policy'}
caps=result['query_caps'];x=np.arange(len(caps))
fig,axes=plt.subplots(2,1,figsize=(11.5,12),sharex=True)
for metric,ax in zip(['ndcg_at_10','recall_at_10'],axes):
    for method in colors:
        values=np.array([[next(c for c in run['conditions'] if c['cap']==cap)['methods'][method]['warm_primary'][metric]
            for cap in caps] for run in result['runs']])
        ax.plot(x,values.mean(axis=0),marker='o',lw=2.6,label=names[method],color=colors[method])
        if method in ['factor_centroid','history_auto']:
            ax.fill_between(x,values.min(axis=0),values.max(axis=0),color=colors[method],alpha=.12)
    reference=[next(c for c in run['conditions'] if c['cap']=='all')['methods']['bpr_full_profile_reference']['warm_primary'][metric] for run in result['runs']]
    ax.scatter([5],[np.mean(reference)],color='#7a4c94',marker='D',s=75,label='Fitted-user BPR (full profile only)',zorder=5)
    ax.set_ylim(0,.21 if metric.startswith('ndcg') else .17);ax.grid(axis='y',alpha=.18)
    ax.set_ylabel('Macro NDCG@10' if metric.startswith('ndcg') else 'Macro Recall@10')
axes[0].legend(loc='lower right',fontsize=11,frameon=True,ncol=2)
axes[0].set_title('95 fixed eligible warm users; same candidates and relevance',fontsize=16,pad=12)
axes[1].set_xticks(x,[str(c) for c in caps]);axes[1].set_xlabel('Retained positive-query cap (categorical); latest items, or all')
fig.subplots_adjust(left=.12,right=.97,top=.94,bottom=.26,hspace=.12)
fig.text(.12,.135,'All 95 users retained in every condition; short histories are not padded.\nFull-history supports across 287 eligible users: 0 likes=193; 1–2=1; 3–4=0; 5+=93.',fontsize=12)
fig.text(.12,.065,'MovieLens100K · frozen19/23 October task · seeds13/29/47, mean + min/max shading\nQuery ablation only: removed likes still informed frozen item models; all seen items excluded.\nExploratory, not independent validation or genuine cold-start evaluation. Source: GroupLens.',fontsize=11)
for suffix in ['png','svg']:fig.savefig('results/history-sensitivity.'+suffix,dpi=160,facecolor='white')
plt.close(fig)

fig,ax=plt.subplots(figsize=(11.5,7));ax.set_axis_off()
nodes=[(.5,.87,'Unique positive catalog items in supplied history'),
       (.5,.61,'Fewer than 3 unique likes?'),(.22,.32,'YES → Past-only popularity'),
       (.78,.32,'NO → Mean learned item vectors\n→ Candidate dot products')]
for px,py,text in nodes:
    ax.text(px,py,text,ha='center',va='center',transform=ax.transAxes,fontsize=15,
        bbox=dict(boxstyle='round,pad=.8',fc='#eef7f6' if py<.4 else '#f0f4fa',ec='#087d7b',lw=1.5))
for src,dst in [((.5,.80),(.5,.69)),((.44,.54),(.22,.41)),((.56,.54),(.78,.43))]:
    ax.annotate('',xy=dst,xytext=src,xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color='#162c45',lw=2))
ax.text(.5,.09,'Exclude every supplied seen item → deterministic ranking → CSV/JSON with method + policy metadata',
    transform=ax.transAxes,ha='center',fontsize=13)
fig.text(.07,.02,'Declared heuristic; threshold3 was not optimized. Centroid scores use fitted item vectors, not a fitted user vector.',fontsize=11)
fig.tight_layout(rect=[0,.04,1,1])
for suffix in ['png','svg']:fig.savefig('results/history-policy.'+suffix,dpi=160,facecolor='white')
