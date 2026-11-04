"""Original workflow and measured tables from the saved batch-serving audit."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path('experiments/2026-11-04')
NAVY, TEAL, AMBER, GRAY = '#152A44', '#087F82', '#AC670A', '#536479'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': NAVY,
    'axes.labelcolor': NAVY, 'font.size': 17, 'svg.fonttype': 'none'})


def canvas(title, subtitle):
    fig = plt.figure(figsize=(12, 10), dpi=150, facecolor='#F7F9FB')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    ax.text(.06, .94, title, fontsize=27, weight='bold', va='top')
    ax.text(.06, .875, subtitle, fontsize=15, color=GRAY, va='top', linespacing=1.45)
    return fig, ax


def save(fig, name):
    for extension in ('png', 'svg'):
        path = ROOT / f'{name}.{extension}'
        fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
        if extension == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines()) + '\n',
                encoding='utf-8', newline='\n')
    plt.close(fig)


def box(ax, y, heading, body, color=TEAL):
    ax.add_patch(FancyBboxPatch((.07, y), .86, .105, boxstyle='round,pad=0.012',
        facecolor='white', edgecolor=color, linewidth=1.7))
    ax.text(.095, y+.073, heading, fontsize=19, weight='bold', color=color)
    ax.text(.095, y+.029, body, fontsize=15)


def main():
    result = json.loads((ROOT/'results.json').read_text(encoding='utf-8'))
    assert result['status'] == 'completed'
    fig, ax = canvas('One batch, traceable ranking evidence',
        'Implemented local CLI + loopback API  |  workflow schematic, not a timing trace')
    steps = [(.70, '1  Validate the complete query batch', 'Unique query IDs, known exclusions, bounded batch size'),
        (.54, '2  Load the fixed scoring snapshot', 'Known fitted user: BPR factors  |  cold user: popularity'),
        (.38, '3  Score, filter, then rank', 'Fixed factor order; remove all seen items; deterministic ties'),
        (.22, '4  Export the same result object', 'Full-precision JSON + CSV, fallback reason + candidate hash')]
    for y, heading, body in steps:
        box(ax, y, heading, body)
    for y in (.70, .54, .38):
        ax.add_patch(FancyArrowPatch((.5, y-.015), (.5, y-.041),
            arrowstyle='-|>', mutation_scale=20, color=TEAL, linewidth=2))
    ax.text(.075, .12, 'Empty candidates remain visible as a query-status record.', fontsize=17, color=AMBER)
    ax.text(.075, .073, 'Default demo: authored synthetic data. Research records and factors stay private.',
        fontsize=13, color=GRAY)
    save(fig, 'batch-workflow')

    fig, ax = canvas('Serving changed; ranking metrics did not',
        'Measured rerun  |  MovieLens100K  |  95 eligible warm users  |  top10')
    ax.text(.07, .77, 'Legacy offline scorer → new batch scorer', fontsize=20, weight='bold')
    methods = [('Popularity', result['runs'][0]['conditions'][4]['warm_primary'])]
    bpr = [r['conditions'][0]['warm_primary'] for r in result['runs']]
    methods.append(('BPR: mean of 3 seeds', {k: float(np.mean([r[k] for r in bpr]))
        for k in ('ndcg_at_10','recall_at_10','catalog_coverage')}))
    columns = [('Method', .07), ('NDCG@10', .50), ('Recall@10', .69), ('Coverage', .85)]
    for label, x in columns:
        ax.text(x, .68, label, fontsize=16, weight='bold', ha='left' if x==.07 else 'center')
    for y, (method, metrics) in zip((.56, .43), methods):
        ax.text(.07, y, method, fontsize=17, weight='bold')
        for metric, x in [('ndcg_at_10',.50),('recall_at_10',.69),('catalog_coverage',.85)]:
            value = metrics[metric]
            label = f'{100*value:.2f}%' if metric=='catalog_coverage' else f'{value:.6f}'
            ax.text(x, y, label, fontsize=20, color=TEAL, ha='center')
    ax.axhline(.64, xmin=.07, xmax=.94, color='#CDD8E3', linewidth=1)
    ax.axhline(.36, xmin=.07, xmax=.94, color='#CDD8E3', linewidth=1)
    ax.text(.07, .285, 'Every displayed metric has zero change from its legacy counterpart.', fontsize=16, color=TEAL)
    ax.text(.07, .23, 'BPR seeds 13 / 29 / 47; no new benchmark fitting or selection.', fontsize=16)
    ax.text(.07, .175, 'Coverage = unique recommended items / 1,616 training-catalog items.', fontsize=15)
    ax.text(.07, .12, 'Repeatedly exposed split: engineering verification, not independent validation.', fontsize=14, color=GRAY)
    ax.text(.07, .072, 'GroupLens Research; Harper & Konstan (2015), DOI 10.1145/2827872.', fontsize=13, color=GRAY)
    save(fig, 'batch-metrics')

    fig, ax = canvas('Batch-boundary audit',
        'Measured rerun  |  3 frozen seeds × 287 eligible users = 861 queries per cell')
    ax.text(.07, .765, 'Top10 order mismatches against the legacy scorer', fontsize=20, weight='bold')
    ax.text(.07, .68, 'Batch size', fontsize=17, weight='bold')
    xs = [.42, .57, .72, .87]
    for x, size in zip(xs, result['batch_sizes']):
        ax.text(x, .68, str(size), ha='center', fontsize=20, weight='bold')
    for y, method in ((.57, 'bpr'), (.46, 'popularity')):
        ax.text(.07, y, 'BPR' if method=='bpr' else 'Popularity', fontsize=19, weight='bold')
        for x, size in zip(xs, result['batch_sizes']):
            errors = sum(c['legacy_top10_order_mismatches'] for r in result['runs']
                for c in r['conditions'] if c['method']==method and c['batch_size']==size)
            ax.text(x, y, f'{errors} / 861', ha='center', fontsize=20, color=TEAL)
    ax.text(.07, .35, 'New scores + exports: exact equality across all four batch sizes.', fontsize=17, color=TEAL)
    ax.text(.07, .28, f"Independent scalar oracle: max error {result['max_scalar_score_error']:.2e}", fontsize=17)
    ax.text(.07, .225, f"Legacy float32 scorer: max error {result['max_legacy_score_error']:.2e}", fontsize=17)
    ax.text(.07, .17, f"{result['scalar_candidate_scores_checked']:,} unseen BPR scores checked against both references.", fontsize=16)
    ax.text(.07, .115, 'Fixed-order float64 scoring; numerical compatibility ≠ bitwise legacy equality.', fontsize=14, color=GRAY)
    ax.text(.07, .068, 'This is a correctness audit. No speedup, quality gain or production claim.', fontsize=14, color=GRAY)
    save(fig, 'batch-parity')


if __name__ == '__main__':
    main()
