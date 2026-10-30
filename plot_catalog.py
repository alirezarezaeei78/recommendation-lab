"""Original static measured table and schematic catalog workflow."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def main():
    p=Path('experiments/2026-10-30');d=json.loads((p/'results.json').read_text(encoding='utf-8'))
    navy='#142B43';teal='#087F83';amber='#B56110';muted='#456073'
    plt.rcParams.update({'font.family':'DejaVu Sans','text.color':navy,'font.size':14,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(9,12.5),dpi=160,facecolor='white')
    fig.text(.065,.95,'Metadata before interactions',fontsize=24,weight='bold')
    fig.text(.065,.922,'Local workflow · synthetic demo · no API key',fontsize=17,color=muted)
    ax=fig.add_axes([.05,.70,.90,.19]);ax.set_axis_off()
    boxes=[(.02,'NEW item\ntags only\n0 interactions'),(.36,'Genre IDF\n+ liked history\n→ cosine'),(.70,'Inspect rank\n+ components\nJSON / CSV')]
    for x,t in boxes:
        ax.add_patch(FancyBboxPatch((x,.18),.28,.69,boxstyle='round,pad=.015',facecolor='#EDF5F5',edgecolor=teal,lw=1.5))
        ax.text(x+.14,.525,t,ha='center',va='center',fontsize=18,linespacing=1.5)
    for x in [.305,.645]:ax.annotate('',xy=(x+.045,.52),xytext=(x,.52),arrowprops={'arrowstyle':'->','color':navy,'lw':2})
    ax.text(.5,.02,'Schematic workflow; scores are ranking signals, not probabilities.',ha='center',fontsize=15,color=muted)
    fig.text(.065,.672,'Measured ranking · simulated cold items',fontsize=22,weight='bold')
    fig.text(.065,.642,'Same full candidates · mean over three BPR seeds',fontsize=16,color=muted)
    ax=fig.add_axes([.065,.348,.87,.268]);ax.axis('off')
    rows=[]
    for method,label in [('popularity','Popularity'),('collaborative','BPR'),('content','Genre content'),('hybrid','Hybrid (RRF)')]:
        m=d['mean_results'][method]
        rows.append([label,f"{m['all']['ndcg_at_10']:.4f}",f"{m['cold']['ndcg_at_10']:.4f}"])
    table=ax.table(cellText=rows,colLabels=['Method','All NDCG@10↑','Cold NDCG@10↑'],loc='center',cellLoc='center',colWidths=[.34,.33,.33])
    table.auto_set_font_size(False);table.set_fontsize(20);table.scale(1,3.1)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('#D5E0E4');cell.set_linewidth(.7)
        if r==0:cell.set_facecolor(navy);cell.get_text().set_color('white');cell.get_text().set_weight('bold')
        elif r==3:cell.set_facecolor('#E2F1EF')
        elif r%2==0:cell.set_facecolor('#F3F6F8')
    fig.text(.065,.328,'All: 94 users / 1,554 positives. Cold: 56 users / 271 positives.',fontsize=15)
    fig.text(.065,.301,'Content reached 18 / 318 cold items; other methods reached zero.',fontsize=16,color=muted)
    fig.text(.065,.264,'Hybrid failure: no cold items in its top 10 in any seed.',fontsize=18,weight='bold',color=amber)
    fig.text(.065,.234,'Cold items receive one rank contribution; warm items can receive two.',fontsize=15,color=muted)
    fig.text(.065,.194,'Protocol',fontsize=16,weight='bold')
    fig.text(.065,.166,'1,616 known items → mask 318 → fit 65,255 past interactions.',fontsize=15)
    fig.text(.065,.141,'Label-independent item mask. Future-only items excluded.',fontsize=15)
    fig.text(.065,.104,'Known MovieLens 100K split reused; exploratory, not independent validation.',fontsize=13,color=muted)
    fig.text(.065,.080,'Simulated cold start. Full past seen-item exclusion is oracle information.',fontsize=13,color=muted)
    fig.text(.065,.056,'Genre snapshot assumed available at cutoff; historical availability unverified.',fontsize=13,color=muted)
    fig.text(.065,.027,'Data: GroupLens; Harper & Konstan (2015), DOI 10.1145/2827872. Research terms.',fontsize=11,color=muted)
    fig.savefig(p/'catalog-cold.png',dpi=160,facecolor='white')
    fig.savefig(p/'catalog-cold.svg',facecolor='white')
    svg=p/'catalog-cold.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')

if __name__=='__main__':main()
