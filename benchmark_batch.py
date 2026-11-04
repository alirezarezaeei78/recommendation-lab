"""Frozen-factor batch-serving audit. Provider rows and query-level audits stay private."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import platform
import time
import numpy as np
from batch_serving import BatchServing, batch_csv, fixed_order_scores
from history_policy import load_fixed_model
from recommender import Row, validate_rows, ranking_metrics

SEEDS = (13, 29, 47)
BATCH_SIZES = (1, 7, 32, 128)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def independent_metric(ids, relevant):
    hits = set(ids) & relevant
    dcg = math.fsum(1 / math.log2(rank + 2) for rank, item in enumerate(ids) if item in relevant)
    ideal = math.fsum(1 / math.log2(rank + 2) for rank in range(min(10, len(relevant))))
    return {'ndcg': dcg / ideal, 'recall': len(hits) / len(relevant)}


def aggregate(records, catalog_size):
    return {'eligible_users': len(records),
        'ndcg_at_10': float(np.mean([r['metric']['ndcg'] for r in records])),
        'recall_at_10': float(np.mean([r['metric']['recall'] for r in records])),
        'catalog_coverage': len({i for r in records for i in r['ids']}) / catalog_size,
        'unique_recommended_items': len({i for r in records for i in r['ids']}),
        'fallback_users': sum(r['fallback'] for r in records)}


def verify_csv(result):
    rows = list(csv.DictReader(io.StringIO(batch_csv(result))))
    statuses = [r for r in rows if r['row_type'] == 'query_status']
    ranking = defaultdict(list)
    for row in rows:
        if row['row_type'] == 'ranking':
            ranking[row['query_id']].append(row)
    assert [r['query_id'] for r in statuses] == [q['query_id'] for q in result['results']]
    for query in result['results']:
        exported = ranking[query['query_id']]
        assert len(exported) == len(query['ranking'])
        for a, b in zip(exported, query['ranking']):
            assert a['item'] == b['item'] and int(a['rank']) == b['rank']
            assert float(a['score']) == b['score']
            assert a['candidate_sha256'] == query['candidate_sha256']


def main(data, factors, audit, output):
    started = time.perf_counter()
    protocol = Path('experiments/2026-11-04/PROTOCOL.md')
    frozen = json.loads(protocol.with_name('protocol-frozen.json').read_text(encoding='utf-8'))
    assert sha(protocol.read_bytes()) == frozen['protocol_sha256']
    prior = json.loads(Path('results/2026-10-23.json').read_text(encoding='utf-8'))
    raw = Path(data).read_bytes()
    assert sha(raw) == prior['input_sha256']
    rows = validate_rows([Row(u, i, float(r), int(t)) for u, i, r, t in
        (line.split() for line in raw.decode('ascii').splitlines())])
    train = [r for r in rows if r.timestamp <= prior['cutoff_unix_seconds']]
    future_rows = [r for r in rows if r.timestamp > prior['cutoff_unix_seconds']]
    assert max(r.timestamp for r in train) < min(r.timestamp for r in future_rows)
    for key, part in [('train_rows_sha256', train), ('test_rows_sha256', future_rows)]:
        assert sha('\n'.join(f'{r.user}\t{r.item}\t{r.rating}\t{r.timestamp}' for r in part).encode()) == prior[key]
    future = defaultdict(set)
    for row in future_rows:
        if row.rating >= 4:
            future[row.user].add(row.item)
    audit = Path(audit)
    audit.mkdir(parents=True, exist_ok=True)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {'date': '2026-11-04', 'status': 'running', 'protocol_sha256': frozen['protocol_sha256'],
        'protocol_frozen_at': frozen['frozen_at'], 'input_sha256': sha(raw),
        'train_rows_sha256': prior['train_rows_sha256'], 'test_rows_sha256': prior['test_rows_sha256'],
        'cutoff_unix_seconds': prior['cutoff_unix_seconds'], 'train_rows': len(train),
        'future_rows': len(future_rows), 'seeds': list(SEEDS), 'batch_sizes': list(BATCH_SIZES),
        'new_benchmark_fits': 0, 'exposure': 'Repeatedly exposed19/23/26October task; engineering parity audit, not independent predictive validation.',
        'data_attribution': prior['data_attribution'], 'data_license': 'GroupLens research terms; no provider records/factors redistributed.',
        'score_policy': 'float64 products accumulated in ascending factor order; vectorized over queries/items',
        'tolerances': {'scalar_absolute': 1e-12, 'legacy_absolute': 2e-5, 'legacy_relative': 2e-6},
        'environment': {'python': platform.python_version(), 'platform': platform.platform(),
            'processor': platform.processor(), 'dependencies': {n: importlib.metadata.version(n)
                for n in ['numpy', 'scipy', 'torch', 'matplotlib']}},
        'runtime_comparison_claim': False, 'confidence_interval': None, 'runs': []}
    for seed in SEEDS:
        factor = Path(factors) / f'factors-{seed}.npz'
        assert sha(factor.read_bytes()) == prior['restricted_audit_sha256'][factor.name]
        model = load_fixed_model(train, factor)
        engine = BatchServing(model)
        catalog = set(model.items)
        cohort = [(user, (future[user] & catalog) - model.seen.get(user, set()), user in model.user_index)
                  for user in sorted({r.user for r in future_rows})]
        cohort = [(u, rel, warm) for u, rel, warm in cohort if rel]
        assert len(cohort) == 287 and sum(w for _, _, w in cohort) == 95 and len(catalog) == 1616
        result['cohort'] = {'warm_primary': 95, 'cold_secondary': 192, 'all_eligible': 287,
            'catalog_denominator': len(catalog), 'excluded_future_users_without_eligible_positive': 14,
            'future_positive_pairs_outside_train_catalog': 95}
        queries = [{'query_id': f'query-{i}', 'user_id': user} for i, (user, _, _) in enumerate(cohort)]
        run = {'seed': seed, 'factor_sha256': sha(factor.read_bytes()),
            'model_fingerprint': engine.fingerprint, 'max_scalar_score_error': 0.0,
            'max_legacy_score_error': 0.0, 'scalar_scores_checked': 0, 'conditions': []}
        fitted = [u for u, _, _ in cohort if u in model.trained_users]
        for start in range(0, len(fitted), 32):
            block = fitted[start:start + 32]
            new_scores = fixed_order_scores(model.user_factors[[model.user_index[u] for u in block]], model.item_factors)
            for index, user in enumerate(block):
                p = model.user_factors[model.user_index[user]]
                legacy = model.item_factors @ p
                oracle = np.array([math.fsum(float(a) * float(b) for a, b in zip(p, q)) for q in model.item_factors])
                allowed = [i for i, item in enumerate(model.items) if item not in model.seen[user]]
                scalar_error = float(np.max(np.abs(new_scores[index, allowed] - oracle[allowed])))
                legacy_error = float(np.max(np.abs(new_scores[index, allowed] - legacy[allowed])))
                run['max_scalar_score_error'] = max(run['max_scalar_score_error'], scalar_error)
                run['max_legacy_score_error'] = max(run['max_legacy_score_error'], legacy_error)
                run['scalar_scores_checked'] += len(allowed)
                assert scalar_error <= 1e-12
                assert np.allclose(new_scores[index, allowed], legacy[allowed], atol=2e-5, rtol=2e-6)
        for method in ('bpr', 'popularity'):
            first = None
            legacy = [model.recommend(method=method, user_id=u if warm else None)
                      for u, _, warm in cohort]
            for size in BATCH_SIZES:
                got = engine.recommend(queries, method=method, n=10, batch_size=size)
                if first is None:
                    first = got
                    (audit / f'{method}-{seed}.json').write_text(json.dumps(got, separators=(',', ':')) + '\n', encoding='utf-8')
                    (audit / f'{method}-{seed}.csv').write_text(batch_csv(got), encoding='utf-8', newline='\n')
                assert got == first
                assert json.loads(json.dumps(got, allow_nan=False)) == got
                verify_csv(got)
                records = []
                mismatches = 0
                for query, old, (user, relevant, warm) in zip(got['results'], legacy, cohort):
                    ids = [r['item'] for r in query['ranking']]
                    allowed = catalog - model.seen.get(user, set())
                    assert len(ids) == 10 and len(set(ids)) == 10 and set(ids) <= allowed
                    assert query['candidate_count'] == len(allowed) == old['candidate_count']
                    assert query['candidate_sha256'] == sha('\n'.join(sorted(allowed)).encode())
                    assert query['fallback'] == old['fallback']
                    mismatches += ids != [r['item'] for r in old['ranking']]
                    metric = ranking_metrics(ids, relevant)
                    oracle = independent_metric(ids, relevant)
                    assert all(math.isclose(metric[k], oracle[k], rel_tol=0, abs_tol=1e-12) for k in metric)
                    records.append({'warm': warm, 'ids': ids, 'metric': metric, 'fallback': query['fallback']})
                condition = {'method': method, 'batch_size': size, 'queries': len(queries),
                    'legacy_top10_order_mismatches': mismatches, 'partition_exact': True,
                    'csv_roundtrip_exact': True, 'candidate_checks_passed': True,
                    'warm_primary': aggregate([r for r in records if r['warm']], len(catalog)),
                    'cold_secondary': aggregate([r for r in records if not r['warm']], len(catalog)),
                    'all_eligible_secondary': aggregate(records, len(catalog))}
                target = (next(r for r in prior['bpr_runs'] if r['seed'] == seed) if method == 'bpr'
                          else prior['baselines']['popularity'])
                condition['warm_metric_delta_from_legacy'] = {m: condition['warm_primary'][m] - target['warm_primary'][m]
                    for m in ['ndcg_at_10', 'recall_at_10', 'catalog_coverage']}
                run['conditions'].append(condition)
                print(json.dumps({'seed': seed, 'method': method, 'batch_size': size,
                    'rank_mismatches': mismatches, 'warm_metric_delta': condition['warm_metric_delta_from_legacy']}), flush=True)
                if time.perf_counter() - started > 1200:
                    raise TimeoutError('20minute evaluation budget exceeded')
        result['runs'].append(run)
        out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    result.update(status='completed', finished_at=datetime.now(timezone.utc).isoformat(),
        wall_seconds_observed=time.perf_counter() - started,
        total_conditions=sum(len(r['conditions']) for r in result['runs']),
        total_query_results=sum(c['queries'] for r in result['runs'] for c in r['conditions']),
        total_legacy_top10_order_mismatches=sum(c['legacy_top10_order_mismatches'] for r in result['runs'] for c in r['conditions']),
        max_scalar_score_error=max(r['max_scalar_score_error'] for r in result['runs']),
        max_legacy_score_error=max(r['max_legacy_score_error'] for r in result['runs']),
        scalar_candidate_scores_checked=sum(r['scalar_scores_checked'] for r in result['runs']),
        private_audit_sha256={p.name: sha(p.read_bytes()) for p in sorted(audit.iterdir()) if p.is_file()})
    out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['status', 'total_conditions', 'total_query_results',
        'total_legacy_top10_order_mismatches', 'max_scalar_score_error', 'max_legacy_score_error', 'scalar_candidate_scores_checked']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('data', 'factors', 'audit', 'output'):
        parser.add_argument('--' + option, required=True)
    args = parser.parse_args()
    main(args.data, args.factors, args.audit, args.output)
