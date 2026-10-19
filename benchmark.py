"""Run the frozen chronological MovieLens100K task; restricted audit stays local."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import importlib.metadata
import time
import numpy as np
from recommender import Row, Recommender, ranking_metrics, validate_rows

def digest(data):
    return hashlib.sha256(data).hexdigest()

def evaluate(data_path, audit_dir, output):
    start = time.perf_counter()
    raw = Path(data_path).read_bytes()
    rows = validate_rows([Row(u, i, float(r), int(t)) for u,i,r,t in (line.split() for line in raw.decode('ascii').splitlines())])
    if len(rows) != 100000:
        raise ValueError('Frozen benchmark expects exactly100000 MovieLens rows')
    cutoff = sorted(r.timestamp for r in rows)[79999]
    train = [r for r in rows if r.timestamp <= cutoff]
    test = [r for r in rows if r.timestamp > cutoff]
    assert max(r.timestamp for r in train) < min(r.timestamp for r in test)
    model = Recommender(train, neighbors=20, shrinkage=10)
    catalog = set(model.items)
    future = defaultdict(set)
    for r in test:
        if r.rating >= 4:
            future[r.user].add(r.item)
    cohort = []
    for user in sorted({r.user for r in test}):
        eligible = (future[user] & catalog) - model.seen.get(user, set())
        if eligible:
            cohort.append((user, eligible, user in model.seen))
    audit_dir = Path(audit_dir); audit_dir.mkdir(parents=True, exist_ok=True)
    predictions, results = [], {}
    for method in ('popularity','item_knn'):
        scores, warm_scores, warm_catalog, all_catalog = [], [], set(), set()
        fallbacks = warm_fallbacks = 0
        for user, relevant, warm in cohort:
            ranking = model.recommend(model.likes.get(user,set()), model.seen.get(user,set()), method, 10)
            ids = [r['item'] for r in ranking['ranking']]
            assert len(set(ids)) == len(ids) and not set(ids) & model.seen.get(user,set())
            assert all(x in catalog for x in ids)
            metric = ranking_metrics(ids, relevant)
            scores.append(metric); all_catalog.update(ids)
            if warm:
                warm_scores.append(metric);warm_catalog.update(ids)
            fallbacks += ranking['fallback'];warm_fallbacks += bool(warm and ranking['fallback'])
            predictions.append({'method':method,'user':user,'warm':warm,'relevant':sorted(relevant),'result':ranking,'metrics':metric})
        def summarize(values, recommended, fallback_count):
            return {'eligible_users':len(values),'ndcg_at_10':float(np.mean([x['ndcg'] for x in values])) if values else None,'recall_at_10':float(np.mean([x['recall'] for x in values])) if values else None,'catalog_coverage':len(recommended)/len(catalog),'unique_recommended_items':len(recommended),'fallback_users':fallback_count}
        results[method]={'warm_primary':summarize(warm_scores,warm_catalog,warm_fallbacks),'all_eligible_secondary':summarize(scores,all_catalog,fallbacks)}
    # Restricted predictions include identifiers; NEVER place audit_dir in public tree.
    (audit_dir/'predictions.json').write_text(json.dumps(predictions,indent=2)+'\n',encoding='utf-8')
    def row_hash(part):
        return digest('\n'.join(f'{r.user}\t{r.item}\t{r.rating}\t{r.timestamp}' for r in part).encode())
    all_test_users = {r.user for r in test};positive_pairs = {(r.user,r.item) for r in test if r.rating>=4}
    metadata = {'dataset':'MovieLens100K; GroupLens research terms; not redistributed','input_sha256':digest(raw),'train_rows_sha256':row_hash(train),'test_rows_sha256':row_hash(test),'cutoff_unix_seconds':cutoff,'cutoff_utc':datetime.fromtimestamp(cutoff,timezone.utc).isoformat(),'train_rows':len(train),'future_rows':len(test),'train_users':len(model.users),'train_catalog_items':len(catalog),'future_users':len(all_test_users),'cold_future_users':len(all_test_users-set(model.users)),'future_positive_pairs':len(positive_pairs),'future_positive_pairs_outside_train_catalog':sum(i not in catalog for u,i in positive_pairs),'future_users_without_eligible_positive':len(all_test_users)-len(cohort),'eligible_users_total':len(cohort),'eligible_warm_users':sum(w for u,r,w in cohort),'eligible_cold_users':sum(not w for u,r,w in cohort),'parameters':{'neighbors':20,'shrinkage':10,'cutoff_position_zero_based':79999,'positive_rating_min':4,'k':10},'results':results,'fit_status':'Closed-form counts/cosine, no iterative optimizer','hardware':{'system':platform.system(),'machine':platform.machine(),'processor':platform.processor()},'dependencies':{n:importlib.metadata.version(n) for n in ['numpy','scipy','matplotlib']},'python':platform.python_version(),'run_seconds_observed':time.perf_counter()-start,'runtime_comparison_claim':False,'uncertainty_interval':None,'scope':'Single fixed chronological local experiment. Binary observed-rating relevance; warm-user primary cohort, many cold users. No tuning. After this run the benchmark is development-visible, not future independent validation.','data_attribution':'Harper and Konstan2015, The MovieLens Datasets: History and Context, DOI10.1145/2827872; GroupLens Research, University of Minnesota.'}
    Path(output).parent.mkdir(exist_ok=True,parents=True)
    Path(output).write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    with (audit_dir/'exposure-history.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps({'run_at':datetime.now(timezone.utc).isoformat(),'input_sha256':metadata['input_sha256'],'cutoff':cutoff,'protocol':'Frozen2026-10-19','selection':'No tuning; first output inspection and any subsequent checks are exploratory'})+'\n')
    return metadata

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);p.add_argument('--audit-dir',required=True);p.add_argument('--output',default='results/2026-10-19.json');a=p.parse_args()
    print(json.dumps(evaluate(a.data,a.audit_dir,a.output),indent=2))
