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
    fig=plt.figure(figsize=(12,12.8),dpi=160,facecolor='white')
    fig.text(.065,.95,'A catalog item can have metadata before interactions',fontsize=22,weight='bold')
    fig.text(.065,.922,'Local workflow • authored synthetic demo • no API key',fontsize=15,color=muted)
    ax=fig.add_axes([.05,.70,.90,.19]);ax.set_axis_off()
    boxes=[(.02,'NEW item\nmetadata tags\n0 interactions'),(.36,'Frozen genre IDF\n+ liked-item profile\n→ content cosine'),(.70,'Choose method\ninspect components\nexport JSON / CSV')]
    for x,t in boxes:
        ax.add_patch(FancyBboxPatch((x,.18),.28,.69,boxstyle='round,pad=.015',facecolor='#EDF5F5',edgecolor=teal,lw=1.5))
        ax.text(x+.14,.525,t,ha='center',va='center',fontsize=16,linespacing=1.5)
    for x in [.305,.645]:ax.annotate('',xy=(x+.045,.52),xytext=(x,.52),arrowprops={'arrowstyle':'->','color':navy,'lw':2})
    ax.text(.5,.02,'Hybrid sums content and collaborative reciprocal ranks; cold items lack the collaborative term.',ha='center',fontsize=12.5,color=muted)
    fig.text(.065,.672,'Measured ranking results · simulated cold items',fontsize=21,weight='bold')
    fig.text(.065,.642,'Same full candidate lists; top 10; mean over three BPR seeds',fontsize=15,color=muted)
    ax=fig.add_axes([.065,.348,.87,.268]);ax.axis('off')
    rows=[]
    for method,label in [('popularity','Popularity'),('collaborative','BPR'),('content','Genre content'),('hybrid','Hybrid (RRF)')]:
        m=d['mean_results'][method]
        rows.append([label,f"{m['all']['ndcg_at_10']:.4f}",f"{m['all']['recall_at_10']:.4f}",f"{m['cold']['ndcg_at_10']:.4f}",f"{m['cold']['coverage']*100:.2f}%"])
    table=ax.table(cellText=rows,colLabels=['Method','All NDCG↑','All Recall↑','Cold NDCG↑','Cold coverage'],loc='center',cellLoc='center',colWidths=[.26,.185,.185,.185,.185])
    table.auto_set_font_size(False);table.set_fontsize(15);table.scale(1,2.75)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('#D5E0E4');cell.set_linewidth(.7)
        if r==0:cell.set_facecolor(navy);cell.get_text().set_color('white');cell.get_text().set_weight('bold')
        elif r==3:cell.set_facecolor('#E2F1EF')
        elif r%2==0:cell.set_facecolor('#F3F6F8')
    fig.text(.065,.328,'All: 94 users / 1,554 positives. Cold relevance: 56 users / 271 positives.',fontsize=14)
    fig.text(.065,.301,'Cold coverage = unique cold items recommended to all 94 users / 318 cold items.',fontsize=13,color=muted)
    fig.text(.065,.264,'Failure retained: the hybrid retrieved no cold items in its top 10 in any seed.',fontsize=15,weight='bold',color=amber)
    fig.text(.065,.234,'Genre content reached 18 / 318 cold items, but ranked worse overall than popularity.',fontsize=14,color=muted)
    fig.text(.065,.194,'Protocol',fontsize=16,weight='bold')
    fig.text(.065,.166,'1,616 known items → mask 318 items → fit on 65,255 past interactions.',fontsize=14)
    fig.text(.065,.141,'Catalog-ID hash mask is label independent. Future-only items are excluded.',fontsize=14)
    fig.text(.065,.104,'Exploratory reuse of the known MovieLens 100K time split; not independent validation.',fontsize=13,color=muted)
    fig.text(.065,.080,'Cold start is simulated. Full past seen-item exclusion is oracle side information.',fontsize=13,color=muted)
    fig.text(.065,.056,'Fixed genre snapshot assumed available at cutoff; historical availability unverified.',fontsize=13,color=muted)
    fig.text(.065,.027,'Data: GroupLens Research; Harper & Konstan (2015), DOI 10.1145/2827872. Research terms.',fontsize=11,color=muted)
    fig.savefig(p/'catalog-cold.png',dpi=160,facecolor='white')
    fig.savefig(p/'catalog-cold.svg',facecolor='white')
    svg=p/'catalog-cold.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')

if __name__=='__main__':main()
