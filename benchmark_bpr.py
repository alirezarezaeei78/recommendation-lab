"""Exploratory BPR extension of the exposed 19 October chronological task."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import numpy as np
from benchmark import evaluate
from bpr import BPRRecommender
from recommender import Row, validate_rows, ranking_metrics

SEEDS = (13, 29, 47)


def main(data, audit, output):
    audit = Path(audit)
    audit.mkdir(parents=True, exist_ok=True)
    protocol = Path('experiments/2026-10-23/PROTOCOL.md')
    if not protocol.exists():
        raise ValueError('Run from the repository root with the frozen protocol present')
    raw = Path(data).read_bytes()
    baseline = evaluate(data, audit/'baseline', audit/'baseline-aggregate.json')
    for method in ['popularity', 'item_knn']:
        # Published table is rounded; exact immutable JSON is authoritative.
        saved = json.loads(Path('results/2026-10-19.json').read_text())
        for metric in ['ndcg_at_10','recall_at_10','catalog_coverage']:
            if not np.isclose(baseline['results'][method]['warm_primary'][metric],
                              saved['results'][method]['warm_primary'][metric], atol=1e-12, rtol=0):
                raise AssertionError('Baseline reproduction differs from the saved task')
    rows = validate_rows([Row(u,i,float(r),int(t)) for u,i,r,t in
        (line.split() for line in raw.decode('ascii').splitlines())])
    cutoff = baseline['cutoff_unix_seconds']
    train = [r for r in rows if r.timestamp <= cutoff]
    future_rows = [r for r in rows if r.timestamp > cutoff]
    assert max(r.timestamp for r in train) < min(r.timestamp for r in future_rows)
    future = defaultdict(set)
    for r in future_rows:
        if r.rating >= 4:
            future[r.user].add(r.item)
    result = {k: baseline[k] for k in ['dataset','input_sha256','train_rows_sha256',
        'test_rows_sha256','cutoff_unix_seconds','cutoff_utc','train_rows','future_rows',
        'train_users','train_catalog_items','future_users','cold_future_users',
        'future_positive_pairs_outside_train_catalog','future_users_without_eligible_positive',
        'eligible_users_total','eligible_warm_users','eligible_cold_users','data_attribution']}
    result.update(date='2026-10-23',status='running',baseline_reproduction_passed=True,
        protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
        exposure='Same 19 October development-visible data and split. Exploratory extension, not fresh independent validation.',
        baselines=baseline['results'], bpr_runs=[],
        environment={'python':platform.python_version(),'processor':platform.processor(),
            'dependencies':{n:importlib.metadata.version(n) for n in ['numpy','scipy','torch']}},
        scope='Established-method minibatch Adam BPR variant; no tuning, scientific novelty, causal benefit or production claim.',
        default_app_seed=13, uncertainty_interval=None)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    def save(): destination.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    save()
    for seed in SEEDS:
        print(f'Fit BPR seed={seed}',flush=True)
        model = BPRRecommender(train,seed=seed)
        catalog = set(model.items)
        cohort = []
        for user in sorted({r.user for r in future_rows}):
            relevant = (future[user] & catalog) - model.seen.get(user,set())
            if relevant: cohort.append((user,relevant,user in model.user_index))
        assert len(cohort)==baseline['eligible_users_total']
        assert sum(w for u,r,w in cohort)==baseline['eligible_warm_users']
        all_scores,warm_scores,all_catalog,warm_catalog = [],[],set(),set()
        audit_rows=[]; all_fallback=warm_fallback=0
        for user,relevant,warm in cohort:
            ranking=model.recommend(method='bpr',user_id=user if warm else None,n=10)
            ids=[r['item'] for r in ranking['ranking']]
            candidates=catalog-model.seen.get(user,set())
            assert ranking['candidate_count']==len(candidates)
            assert len(ids)==len(set(ids)) and set(ids)<=candidates
            metric=ranking_metrics(ids,relevant,10)
            all_scores.append(metric);all_catalog.update(ids);all_fallback+=ranking['fallback']
            if warm:
                warm_scores.append(metric);warm_catalog.update(ids);warm_fallback+=ranking['fallback']
            audit_rows.append({'user':user,'warm':warm,'relevant':sorted(relevant),
                'candidate_sha256':hashlib.sha256('\n'.join(sorted(candidates)).encode()).hexdigest(),
                'result':ranking,'metrics':metric})
        def summarize(scores,items,fallback):
            return {'eligible_users':len(scores),'ndcg_at_10':float(np.mean([r['ndcg'] for r in scores])),
                'recall_at_10':float(np.mean([r['recall'] for r in scores])),
                'catalog_coverage':len(items)/len(catalog),'unique_recommended_items':len(items),'fallback_users':fallback}
        factors=audit/f'factors-{seed}.npz'
        np.savez_compressed(factors,p=model.user_factors,q=model.item_factors)
        with np.load(factors) as replay:
            np.testing.assert_array_equal(replay['p'],model.user_factors)
            np.testing.assert_array_equal(replay['q'],model.item_factors)
        (audit/f'predictions-{seed}.json').write_text(json.dumps(audit_rows,indent=2)+'\n',encoding='utf-8')
        record={'seed':seed,'training':model.training,
                'warm_primary':summarize(warm_scores,warm_catalog,warm_fallback),
                'all_eligible_secondary':summarize(all_scores,all_catalog,all_fallback),
                'factor_roundtrip_exact':True}
        result['bpr_runs'].append(record)
        save()
        print(json.dumps({k:v for k,v in record.items() if k!='training'}),flush=True)
    result['bpr_seed_summary']={}
    for metric in ['ndcg_at_10','recall_at_10','catalog_coverage']:
        values=[r['warm_primary'][metric] for r in result['bpr_runs']]
        result['bpr_seed_summary'][metric]={'mean':float(np.mean(values)),'min':min(values),'max':max(values),'all_seeds':values}
    result['restricted_audit_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in audit.iterdir() if p.is_file()}
    result['status']='completed';save()
    print(json.dumps(result['bpr_seed_summary'],indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True)
    parser.add_argument('--audit-dir',required=True)
    parser.add_argument('--output',default='results/2026-10-23.json')
    args=parser.parse_args()
    main(args.data,args.audit_dir,args.output)
