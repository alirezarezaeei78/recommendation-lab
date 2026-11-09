"""Original measured candidate-policy table, generated from actual aggregate JSON."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    dest = Path('experiments/2026-11-09')
    data = json.loads((dest / 'results.json').read_text())
    assert data['status'] == 'completed' and data['total_conditions'] == 96
    rows = {(x['nonrelevant_budget'], x['method']): x for x in data['summary']}
    navy, teal, gray, light = '#152f46', '#087f82', '#526475', '#f5f8fa'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none'})
    fig = plt.figure(figsize=(10.8, 14), dpi=100, facecolor=light)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')
    def text(x, y, value, size=22, color=navy, weight='normal', **kw):
        ax.text(x, y, value, fontsize=size, color=color, fontweight=weight,
                va='top', transform=ax.transAxes, **kw)
    text(.06, .955, 'Same model. Different candidates.', 30, weight='bold')
    text(.06, .902, '95 warm queries | frozen BPR vs training popularity', 19, gray)
    text(.06, .868, 'All observed positives retained; Recall / ideal-DCG\ndenominators unchanged. Only distractors removed.', 19, gray)
    text(.06, .785, 'Candidate policy', 20, weight='bold')
    text(.54, .785, 'NDCG@10', 20, weight='bold')
    text(.80, .785, 'Recall@10', 20, weight='bold')
    for j, budget in enumerate((None, 500, 100, 20)):
        y = .735 - j * .145
        ax.plot([.06, .94], [y+.015, y+.015], color='#ccd7de', transform=ax.transAxes, lw=1)
        label = 'Full eligible catalog' if budget is None else f'All positives + {budget} others'
        bpr, pop = rows[budget, 'bpr'], rows[budget, 'popularity']
        text(.06, y, label, 19, weight='bold')
        text(.06, y-.034, f"Pool median: {bpr['candidate_count_median']:,.0f} items", 17, gray)
        for offset, item, name, color in ((.0, bpr, 'BPR', teal), (.047, pop, 'Pop.', gray)):
            text(.44, y-offset, name, 17, color)
            text(.56, y-offset, f"{item['ndcg_at_10']['mean']:.4f}", 22, color)
            text(.81, y-offset, f"{item['recall_at_10']['mean']:.4f}", 22, color)
    text(.06, .145, 'Higher sampled scores are not model improvement.', 21, weight='bold')
    text(.06, .108, 'Full: mean over 3 frozen model seeds. Sampled: 3 model × 5 sampling seeds.\nShared candidate pools across methods. Descriptive means, no intervals.\nLabel-assisted offline test; known exposed split; not independent validation.', 15, gray)
    text(.06, .042, 'MovieLens 100K | GroupLens research terms | Harper & Konstan (2015)\nDOI 10.1145/2827872 | Original measured comparison', 14, gray)
    fig.savefig(dest / 'candidate-policy.png', dpi=100, facecolor=light)
    fig.savefig(dest / 'candidate-policy.svg', facecolor=light)
    plt.close(fig)


if __name__ == '__main__':
    main()
