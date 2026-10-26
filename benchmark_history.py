"""Frozen-model query-history ablation; restricted row-level audit stays local."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import time
import numpy as np
from history_policy import load_fixed_model, MIN_POSITIVES
from recommender import Row, validate_rows, ranking_metrics

CAPS = (0, 1, 3, 5, 10, 'all')
SEEDS = (13, 29, 47)
METHODS = ('popularity', 'item_knn', 'factor_centroid', 'history_auto')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def metric_oracle(ids, relevant):
    # Independent scalar computation, no call to ranking_metrics.
    gains = [1/math.log(i+2, 2) for i,item in enumerate(ids[:10]) if item in relevant]
    ideal = math.fsum(1/math.log(i+2, 2) for i in range(min(10,len(relevant))))
    return math.fsum(gains)/ideal, len(set(ids[:10]) & relevant)/len(relevant)


def history_bin(count):
    return '0' if count == 0 else '1-2' if count < 3 else '3-4' if count < 5 else '5+'


def summarize(records, catalog_size):
    ids = {row['item'] for r in records for row in r['result']['ranking']}
    return {'eligible_users':len(records),
        'ndcg_at_10':float(np.mean([r['metrics']['ndcg'] for r in records])) if records else None,
        'recall_at_10':float(np.mean([r['metrics']['recall'] for r in records])) if records else None,
        'catalog_coverage':len(ids)/catalog_size, 'unique_recommended_items':len(ids),
        'fallback_users':sum(r['result']['fallback'] for r in records),
        'retained_positive_min':min((r['retained'] for r in records),default=None),
        'retained_positive_max':max((r['retained'] for r in records),default=None)}


def public_bin_summary(records, catalog_size):
    summary = summarize(records, catalog_size)
    if 0 < len(records) < 5:
        return {'eligible_users':len(records), 'metrics_suppressed':True,
                'reason':'Fewer than5 users; individual-level audit remains private'}
    return summary


def main(data, prior_audit, audit, output):
    start = time.perf_counter()
    raw = Path(data).read_bytes()
    old = json.loads(Path('results/2026-10-23.json').read_text())
    protocol = Path('experiments/2026-10-26/PROTOCOL.md')
    assert protocol.exists() and sha(raw) == old['input_sha256']
    rows = validate_rows([Row(u,i,float(r),int(t)) for u,i,r,t in
        (line.split() for line in raw.decode('ascii').splitlines())])
    cutoff = old['cutoff_unix_seconds']
    train = [r for r in rows if r.timestamp <= cutoff]
    future_rows = [r for r in rows if r.timestamp > cutoff]
    assert max(r.timestamp for r in train) < min(r.timestamp for r in future_rows)
    for key,part in [('train_rows_sha256',train),('test_rows_sha256',future_rows)]:
        assert sha('\n'.join(f'{r.user}\t{r.item}\t{r.rating}\t{r.timestamp}' for r in part).encode()) == old[key]
    positive_history, future = defaultdict(list), defaultdict(set)
    for r in train:
        if r.rating >= 4: positive_history[r.user].append(r)
    for u in positive_history:
        positive_history[u].sort(key=lambda r:(-r.timestamp,r.item))
    for r in future_rows:
        if r.rating >= 4: future[r.user].add(r.item)
    audit = Path(audit);audit.mkdir(parents=True,exist_ok=True)
    result = {'date':'2026-10-26','status':'running','input_sha256':sha(raw),
        'protocol_sha256':sha(protocol.read_bytes()),'threshold':MIN_POSITIVES,
        'query_caps':list(CAPS),'seeds':list(SEEDS), 'new_training_fits':0,
        'prior_commit':'92a1c41b23453db4dd53a33724b3a36dfafcb6f5',
        'dataset':old['dataset'],'data_attribution':old['data_attribution'],
        'cutoff_unix_seconds':cutoff,'train_rows':len(train),'future_rows':len(future_rows),
        'train_rows_sha256':old['train_rows_sha256'],'test_rows_sha256':old['test_rows_sha256'],
        'exposure':'Known19/23 October chronological task; exploratory frozen-model query ablation. Removed query likes still informed the frozen training objects. Full seen exclusion preserved. Not a cold-start or retraining experiment.',
        'parameters':{'neighbors':20,'shrinkage':10,'n':10,'positive_rating_min':4,'threshold_tuned':False},
        'uncertainty_interval':None,'runtime_comparison_claim':False,'runs':[],
        'hardware':{'system':platform.platform(),'processor':platform.processor()},
        'python':platform.python_version(),
        'dependencies':{n:importlib.metadata.version(n) for n in ['numpy','scipy','torch','matplotlib']}}
    output = Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    def save(): output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    save()
    for seed in SEEDS:
        factor = Path(prior_audit)/f'factors-{seed}.npz'
        assert sha(factor.read_bytes()) == old['restricted_audit_sha256'][factor.name]
        model = load_fixed_model(train,factor)
        catalog = set(model.items)
        cohort = []
        for u in sorted({r.user for r in future_rows}):
            relevant = (future[u] & catalog)-model.seen.get(u,set())
            if relevant:cohort.append((u,relevant,u in model.user_index))
        assert len(cohort)==287 and sum(w for _,_,w in cohort)==95
        result['cohort'] = {'eligible_warm':95,'eligible_cold':192,
            'excluded_future_users_without_eligible_positive':old['future_users_without_eligible_positive'],
            'future_positive_pairs_outside_train_catalog':old['future_positive_pairs_outside_train_catalog'],
            'training_catalog':len(catalog), 'full_history_bin_supports':{
                key:sum(history_bin(len(positive_history[u]))==key for u,_,_ in cohort)
                for key in ['0','1-2','3-4','5+']}}
        record = {'seed':seed,'factor_sha256':sha(factor.read_bytes()),'conditions':[]}
        with (audit/f'predictions-{seed}.jsonl').open('w',encoding='utf-8') as stream:
            for cap in CAPS:
                per_method = {method:[] for method in METHODS}
                if cap=='all':per_method['bpr_full_profile_reference']=[]
                for user,relevant,warm in cohort:
                    if not warm and cap!='all':continue
                    selected = positive_history[user] if cap=='all' else positive_history[user][:cap]
                    liked = [r.item for r in selected]
                    seen = model.seen.get(user,set())
                    candidates = catalog-seen
                    candidate_hash = sha('\n'.join(sorted(candidates)).encode())
                    for method in per_method:
                        if method=='bpr_full_profile_reference':
                            ranking=model.recommend(method='bpr',user_id=user if warm else None,n=10)
                        else:
                            ranking=model.recommend(liked,seen,method,10)
                        ids=[r['item'] for r in ranking['ranking']]
                        assert len(set(ids))==len(ids) and set(ids)<=candidates
                        assert ranking['candidate_count']==len(candidates)
                        metrics=ranking_metrics(ids,relevant)
                        oracle=metric_oracle(ids,relevant)
                        assert math.isclose(metrics['ndcg'],oracle[0],abs_tol=1e-12)
                        assert math.isclose(metrics['recall'],oracle[1],abs_tol=1e-12)
                        row={'user':user,'warm':warm,'cap':cap,'method':method,
                            'retained':len(liked),'history_bin':history_bin(len(positive_history[user])),
                            'relevant':sorted(relevant),'candidate_sha256':candidate_hash,
                            'result':ranking,'metrics':metrics}
                        per_method[method].append(row)
                        stream.write(json.dumps(row,separators=(',',':'))+'\n')
                condition={'cap':cap,'methods':{}}
                for method,records in per_method.items():
                    warm_records=[r for r in records if r['warm']]
                    summary={'warm_primary':summarize(warm_records,len(catalog))}
                    if cap=='all':
                        summary['all_eligible_secondary']=summarize(records,len(catalog))
                        summary['full_history_bins']={key:public_bin_summary([r for r in records if r['history_bin']==key],len(catalog))
                            for key in ['0','1-2','3-4','5+']}
                        if method in ['popularity','item_knn','bpr_full_profile_reference']:
                            prior = next(r for r in old['bpr_runs'] if r['seed']==seed)['warm_primary'] if method.startswith('bpr') else old['baselines'][method]['warm_primary']
                            for metric in ['ndcg_at_10','recall_at_10','catalog_coverage']:
                                assert math.isclose(summary['warm_primary'][metric],prior[metric],abs_tol=1e-12)
                    condition['methods'][method]=summary
                record['conditions'].append(condition)
                print(json.dumps({'seed':seed,'cap':cap,'warm':{m:v['warm_primary']['ndcg_at_10'] for m,v in condition['methods'].items()}}),flush=True)
                if time.perf_counter()-start > 1200:raise TimeoutError('Declared20minute evaluation budget exhausted')
        result['runs'].append(record);save()
    result.update(status='completed',baseline_reproduction_passed=True,
        independent_metric_oracles_passed=True,all_candidates_and_split_hashes_passed=True,
        runtime_seconds_observed=time.perf_counter()-start,finished_at=datetime.now(timezone.utc).isoformat(),
        restricted_audit_sha256={p.name:sha(p.read_bytes()) for p in audit.iterdir() if p.is_file()})
    save()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True);parser.add_argument('--prior-audit',required=True)
    parser.add_argument('--audit-dir',required=True);parser.add_argument('--output',default='results/2026-10-26.json')
    args=parser.parse_args();main(args.data,args.prior_audit,args.audit_dir,args.output)
