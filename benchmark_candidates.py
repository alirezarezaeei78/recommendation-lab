"""Frozen-model candidate-policy experiment; provider data and audit stay private."""
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
from batch_serving import fixed_order_scores
from history_policy import load_fixed_model
from recommender import Row, validate_rows

MODEL_SEEDS = (13, 29, 47)
SAMPLING_SEEDS = (101, 202, 303, 404, 505)
BUDGETS = (20, 100, 500)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def candidate_pool(eligible, relevant, budget, seed, user):
    """Offline label-assisted pool: keep all positives and sample other items."""
    eligible, relevant = set(eligible), set(relevant)
    if not relevant or not relevant <= eligible:
        raise ValueError('Require nonempty eligible relevance')
    if budget is None:
        return sorted(eligible)
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError('budget must be nonnegative integer or None')
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in (seed, user)):
        raise ValueError('Nonnegative integer seed/user required')
    other = sorted(eligible - relevant)
    rng = np.random.default_rng(np.random.SeedSequence([seed, user]))
    selected = rng.permutation(len(other))[:budget]
    return sorted(relevant | {other[int(i)] for i in selected})


def rank_pool(pool, scores, popularity, index, n=10):
    if len(set(pool)) != len(pool) or any(x not in index for x in pool):
        raise ValueError('Unique known candidates required')
    if not np.isfinite(scores).all() or not np.isfinite(popularity).all():
        raise ValueError('Finite scores required')
    return sorted(pool, key=lambda x: (-float(scores[index[x]]),
                                       -float(popularity[index[x]]), x))[:n]


def metric(ranking, relevant):
    if not relevant or len(set(ranking)) != len(ranking):
        raise ValueError('Nonempty relevance and unique ranking required')
    hits = np.array([x in relevant for x in ranking[:10]], dtype=np.float64)
    discounts = 1 / np.log2(np.arange(len(hits)) + 2)
    ideal = np.sum(1 / np.log2(np.arange(min(10, len(relevant))) + 2))
    got = {'ndcg': float(np.sum(hits * discounts) / ideal),
           'recall': float(np.sum(hits) / len(relevant))}
    # Independently written scalar oracle, including partial pools and ties.
    oracle_dcg = math.fsum(1 / math.log2(i + 2) for i, x in enumerate(ranking[:10]) if x in relevant)
    oracle_ideal = math.fsum(1 / math.log2(i + 2) for i in range(min(10, len(relevant))))
    assert math.isclose(got['ndcg'], oracle_dcg / oracle_ideal, abs_tol=1e-12, rel_tol=0)
    assert math.isclose(got['recall'], len(set(ranking[:10]) & relevant) / len(relevant), abs_tol=1e-12, rel_tol=0)
    return got


def summarize(records):
    assert records
    counts = [r['candidate_count'] for r in records]
    return {'queries': len(records),
            'ndcg_at_10': math.fsum(r['ndcg'] for r in records) / len(records),
            'recall_at_10': math.fsum(r['recall'] for r in records) / len(records),
            'candidate_count_min': min(counts),
            'candidate_count_median': float(np.median(counts)),
            'candidate_count_max': max(counts),
            'relevant_pairs': sum(r['relevant_count'] for r in records),
            'fallback_queries': sum(r['fallback'] for r in records)}


def csv_audit(records):
    stream = io.StringIO(newline='')
    fields = ['model_seed', 'sampling_seed', 'budget', 'method', 'user', 'warm',
              'fallback', 'candidate_count', 'candidate_sha256', 'relevant_count',
              'ranking', 'scores', 'ndcg', 'recall']
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for record in records:
        row = dict(record)
        row['ranking'] = json.dumps(row['ranking'], separators=(',', ':'))
        row['scores'] = json.dumps(row['scores'], separators=(',', ':'))
        for key in ('ndcg', 'recall'):
            row[key] = format(row[key], '.17g')
        writer.writerow(row)
    raw = stream.getvalue()
    decoded = list(csv.DictReader(io.StringIO(raw)))
    assert len(decoded) == len(records)
    for row, record in zip(decoded, records):
        assert json.loads(row['ranking']) == record['ranking']
        assert json.loads(row['scores']) == record['scores']
        assert all(float(row[k]) == record[k] for k in ('ndcg', 'recall'))
        assert row['candidate_sha256'] == record['candidate_sha256']
        assert int(row['candidate_count']) == record['candidate_count']
        assert int(row['relevant_count']) == record['relevant_count']
        assert row['warm'] == str(record['warm']) and row['fallback'] == str(record['fallback'])
        assert int(row['model_seed']) == record['model_seed']
        assert row['method'] == record['method'] and row['user'] == record['user']
        for key in ('sampling_seed', 'budget'):
            assert row[key] == ('' if record[key] is None else str(record[key]))
    return raw


def main(data, factors, audit, output):
    started = time.perf_counter()
    audit, output = Path(audit), Path(output)
    if audit.exists() or output.exists() or audit.resolve() == output.resolve():
        raise FileExistsError('Preserve existing paths; use new audit/output names')
    if any(output.resolve() == Path(x).resolve() for x in (data, factors)):
        raise ValueError('Output must not replace an input')
    protocol = Path('experiments/2026-11-09/PROTOCOL.md')
    frozen = json.loads(protocol.with_name('protocol-frozen.json').read_text(encoding='utf-8'))
    assert sha(protocol.read_bytes()) == frozen['protocol_sha256']
    prior = json.loads(Path('results/2026-10-23.json').read_text(encoding='utf-8'))
    batch_prior = json.loads(Path('experiments/2026-11-04/results.json').read_text(encoding='utf-8'))
    raw = Path(data).read_bytes()
    assert sha(raw) == prior['input_sha256']
    rows = validate_rows([Row(u, i, float(r), int(t)) for u, i, r, t in
                          (line.split() for line in raw.decode('ascii').splitlines())])
    train = [r for r in rows if r.timestamp <= prior['cutoff_unix_seconds']]
    future_rows = [r for r in rows if r.timestamp > prior['cutoff_unix_seconds']]
    assert len(train) == 80003 and len(future_rows) == 19997
    assert max(r.timestamp for r in train) < min(r.timestamp for r in future_rows)
    for key, part in [('train_rows_sha256', train), ('test_rows_sha256', future_rows)]:
        assert sha('\n'.join(f'{r.user}\t{r.item}\t{r.rating}\t{r.timestamp}' for r in part).encode()) == prior[key]
    future = defaultdict(set)
    for row in future_rows:
        if row.rating >= 4:
            future[row.user].add(row.item)
    conditions = [(None, None)] + [(seed, budget) for seed in SAMPLING_SEEDS for budget in BUDGETS]
    audit.mkdir(parents=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {'date': '2026-11-09', 'status': 'running', 'new_fits': 0,
              'protocol_sha256': frozen['protocol_sha256'], 'frozen_at': frozen['frozen_at'],
              'input_sha256': sha(raw), 'train_rows_sha256': prior['train_rows_sha256'],
              'test_rows_sha256': prior['test_rows_sha256'], 'cutoff': prior['cutoff_unix_seconds'],
              'train_rows': len(train), 'future_rows': len(future_rows), 'catalog_size': 1616,
              'model_seeds': list(MODEL_SEEDS), 'sampling_seeds': list(SAMPLING_SEEDS),
              'nonrelevant_budgets': list(BUDGETS), 'conditions': [], 'factor_sha256': {},
              'full_metrics_reproduced': True, 'candidate_hashes_identical_across_models': True,
              'scope': 'Label-assisted sampled offline candidate policies; known repeatedly exposed chronological task. No model improvement, confidence interval, speed, production or causal claim.',
              'data_attribution': prior['data_attribution'],
              'data_licence': 'GroupLens research-only terms; no provider records or factors redistributed.',
              'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                              'processor': platform.processor(), 'dependencies': {n: importlib.metadata.version(n) for n in ('numpy', 'scipy', 'torch', 'matplotlib')}}}
    all_records, all_pools, candidate_hash_reference = [], [], {}
    for seed in MODEL_SEEDS:
        factor = Path(factors) / f'factors-{seed}.npz'
        factor_hash = sha(factor.read_bytes())
        assert factor_hash == prior['restricted_audit_sha256'][factor.name]
        result['factor_sha256'][str(seed)] = factor_hash
        model = load_fixed_model(train, factor)
        before = sha(model.user_factors.tobytes() + model.item_factors.tobytes())
        catalog = set(model.items)
        cohort = [(u, (future[u] & catalog) - model.seen.get(u, set()), u in model.user_index)
                  for u in sorted({r.user for r in future_rows})]
        cohort = [(u, rel, warm) for u, rel, warm in cohort if rel]
        assert len(cohort) == 287 and sum(w for _, _, w in cohort) == 95 and len(catalog) == 1616
        result['cohort'] = {'warm_primary': 95, 'cold_secondary': 192, 'all_eligible': 287,
                            'excluded_future_users': 14, 'future_positive_pairs_outside_train_catalog': 95}
        users = [u for u, _, _ in cohort]
        scores_by_user = {}
        fitted = [u for u in users if u in model.trained_users]
        for start in range(0, len(fitted), 32):
            block = fitted[start:start + 32]
            values = fixed_order_scores(model.user_factors[[model.user_index[u] for u in block]], model.item_factors)
            scores_by_user.update(zip(block, values))
        for sampling_seed, budget in conditions:
            records_by_method = {method: [] for method in ('bpr', 'popularity')}
            joined_hashes = []
            for user, relevant, warm in cohort:
                eligible = catalog - model.seen.get(user, set())
                pool = candidate_pool(eligible, relevant, budget, sampling_seed, int(user))
                assert relevant <= set(pool) <= eligible and len(pool) == len(set(pool))
                if budget is not None:
                    assert len(pool) == len(relevant) + min(budget, len(eligible - relevant))
                    if budget != BUDGETS[0]:
                        smaller = candidate_pool(eligible, relevant, BUDGETS[BUDGETS.index(budget)-1], sampling_seed, int(user))
                        assert set(smaller) <= set(pool)
                digest = sha('\n'.join(pool).encode())
                key = (sampling_seed, budget, user)
                if seed == MODEL_SEEDS[0]:
                    candidate_hash_reference[key] = digest
                    all_pools.append({'sampling_seed': sampling_seed, 'budget': budget, 'user': user,
                                      'candidates': pool, 'relevant': sorted(relevant), 'sha256': digest})
                else:
                    assert digest == candidate_hash_reference[key]
                joined_hashes.append(user + ':' + digest)
                for method in ('bpr', 'popularity'):
                    fallback = method == 'bpr' and user not in scores_by_user
                    scores = scores_by_user[user] if method == 'bpr' and not fallback else model.popularity
                    ranking = rank_pool(pool, scores, model.popularity, model.index)
                    assert len(ranking) == min(10, len(pool)) and len(set(ranking)) == len(ranking)
                    row = {'model_seed': seed, 'sampling_seed': sampling_seed, 'budget': budget,
                           'method': method, 'user': user, 'warm': warm, 'fallback': fallback,
                           'candidate_count': len(pool), 'candidate_sha256': digest,
                           'relevant_count': len(relevant), 'ranking': ranking,
                           'scores': [float(scores[model.index[x]]) for x in ranking], **metric(ranking, relevant)}
                    records_by_method[method].append(row)
                    all_records.append(row)
            for method, records in records_by_method.items():
                c = {'model_seed': seed, 'sampling_seed': sampling_seed, 'nonrelevant_budget': budget,
                     'candidate_policy': 'full' if budget is None else 'all_observed_relevant_plus_uniform_unobserved',
                     'method': method, 'query_candidate_pairs_sha256': sha('\n'.join(joined_hashes).encode()),
                     'warm_primary': summarize([x for x in records if x['warm']]),
                     'cold_secondary': summarize([x for x in records if not x['warm']]),
                     'all_eligible': summarize(records)}
                if budget is None:
                    prev = next(x for x in batch_prior['runs'] if x['seed'] == seed)
                    target = next(x for x in prev['conditions'] if x['method'] == method and x['batch_size'] == 1)
                    c['full_metric_deltas'] = {k: c['warm_primary'][k] - target['warm_primary'][k] for k in ('ndcg_at_10', 'recall_at_10')}
                    assert all(abs(x) <= 1e-12 for x in c['full_metric_deltas'].values())
                result['conditions'].append(c)
            if time.perf_counter() - started > 1200:
                raise TimeoutError('20minute budget exceeded')
        assert before == sha(model.user_factors.tobytes() + model.item_factors.tobytes())
        print(json.dumps({'model_seed': seed, 'conditions_completed': len(result['conditions'])}), flush=True)
        output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    (audit / 'candidate-pools.json').write_text(json.dumps(all_pools, separators=(',', ':')) + '\n', encoding='utf-8')
    (audit / 'query-results.json').write_text(json.dumps(all_records, separators=(',', ':'), allow_nan=False) + '\n', encoding='utf-8')
    assert json.loads((audit / 'query-results.json').read_text()) == all_records
    (audit / 'query-results.csv').write_text(csv_audit(all_records), encoding='utf-8', newline='')
    result['summary'] = []
    for budget in (None, *BUDGETS):
        for method in ('bpr', 'popularity'):
            values = [c for c in result['conditions'] if c['method'] == method and c['nonrelevant_budget'] == budget]
            summary = {'method': method, 'nonrelevant_budget': budget, 'conditions': len(values),
                       'warm_queries_each': 95, 'candidate_count_min': min(c['warm_primary']['candidate_count_min'] for c in values),
                       'candidate_count_median': float(np.median([c['warm_primary']['candidate_count_median'] for c in values])),
                       'candidate_count_max': max(c['warm_primary']['candidate_count_max'] for c in values)}
            for key in ('ndcg_at_10', 'recall_at_10'):
                v = [c['warm_primary'][key] for c in values]
                summary[key] = {'mean': math.fsum(v) / len(v), 'min': min(v), 'max': max(v)}
            result['summary'].append(summary)
    result.update(status='completed', finished_at=datetime.now(timezone.utc).isoformat(),
                  total_conditions=len(result['conditions']), query_results=len(all_records),
                  scalar_metric_checks=len(all_records), exact_JSON_CSV_audit=True,
                  unchanged_factors=True, nested_pools_checked=True, positives_and_denominators_retained=True,
                  private_audit_sha256={f.name: sha(f.read_bytes()) for f in sorted(audit.iterdir())},
                  wall_seconds_audit_only=time.perf_counter() - started)
    assert result['total_conditions'] == 96 and result['query_results'] == 27552
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'completed', 'summary': result['summary'], 'query_results': len(all_records)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'factors', 'audit', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    main(args.data, args.factors, args.audit, args.output)
