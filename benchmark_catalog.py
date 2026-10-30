"""Frozen simulated catalog-cold experiment; records/factors stay private."""
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
import zipfile
import joblib
import numpy as np
from bpr import BPRRecommender
from recommender import Row,validate_rows,ranking_metrics
from catalog_model import CatalogModel

METHODS=('popularity','collaborative','content','hybrid')
SEEDS=(13,29,47)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def independent_metrics(ranking,relevant):
    positions=[j+1 for j,item in enumerate(ranking[:10]) if item in relevant]
    numerator=sum(math.log(2)/math.log(p+1) for p in positions)
    denominator=sum(math.log(2)/math.log(j+2) for j in range(min(10,len(relevant))))
    return {'ndcg':numerator/denominator,'recall':len(positions)/len(relevant)}


def main(archive,audit_dir,output):
    started=time.perf_counter();audit=Path(audit_dir);audit.mkdir(parents=True,exist_ok=True)
    protocol=Path('experiments/2026-10-30/PROTOCOL.md')
    frozen=json.loads(protocol.with_name('protocol-frozen.json').read_text(encoding='utf-8'))
    assert sha(protocol.read_bytes())==frozen['protocol_sha256']
    for name,digest in frozen['code_sha256'].items():assert sha(Path(name).read_bytes())==digest
    with zipfile.ZipFile(archive) as z:
        raw=z.read('ml-100k/u.data');metadata=z.read('ml-100k/u.item')
    rows=validate_rows([Row(u,i,float(r),int(t)) for u,i,r,t in (l.split() for l in raw.decode('ascii').splitlines())])
    assert len(rows)==100000
    cutoff=sorted(r.timestamp for r in rows)[79999]
    past=[r for r in rows if r.timestamp<=cutoff];future=[r for r in rows if r.timestamp>cutoff]
    catalog_ids={r.item for r in past}
    cold={i for i in catalog_ids if int(sha(('catalog-cold-v1:'+i).encode()),16)%5==0}
    train=[r for r in past if r.item not in cold]
    assert not cold & {r.item for r in train}
    assert max(r.timestamp for r in train)<min(r.timestamp for r in future)
    catalog={}
    for line in metadata.decode('latin-1').splitlines():
        fields=line.split('|')
        if fields[0] in catalog_ids:
            catalog[fields[0]]=[f'g{j:02}' for j,x in enumerate(fields[5:]) if x=='1']
    assert set(catalog)==catalog_ids
    seen=defaultdict(set);likes=defaultdict(set);positives=defaultdict(set)
    for r in past:seen[r.user].add(r.item)
    for r in train:
        if r.rating>=4:likes[r.user].add(r.item)
    for r in future:
        if r.rating>=4:positives[r.user].add(r.item)
    cohort=[]
    for u in sorted({r.user for r in future}):
        relevant=(positives[u]&catalog_ids)-seen[u]
        if likes[u] and relevant:cohort.append((u,relevant))
    out={'date':'2026-10-30','status':'running','protocol_sha256':sha(protocol.read_bytes()),
         'input_hashes':{'u.data':sha(raw),'u.item':sha(metadata)},'cutoff':cutoff,
         'counts':{'original_past_rows':len(past),'fit_rows':len(train),'removed_past_rows':len(past)-len(train),
             'future_rows':len(future),'catalog_items':len(catalog_ids),'cold_items':len(cold),'warm_items':len(catalog_ids-cold),
             'future_users':len({r.user for r in future}),'eligible_primary_users':len(cohort),
             'excluded_future_users':len({r.user for r in future})-len(cohort),
             'future_positive_pairs_outside_catalog':sum(i not in catalog_ids for u,items in positives.items() for i in items)},
         'exposure':'Known19/23/26October time split; new label-independent simulated item mask; exploratory, not independent validation',
         'caveats':['Catalog-cold interaction deletion is simulated; original past seen exclusion is oracle side information',
                    'Fixed metadata snapshot assumed available at cutoff, not historically verified',
                    'Warm-positive users only; historical observed ratings, not clicks, revenue or first-launch performance',
                    'Cold/warm relevance cohorts differ; same full rankings, no subgroup reranking',
                    'Seed spread is optimizer sensitivity, not confidence interval; no parameter tuning'],
         'environment':{'python':platform.python_version(),'processor':platform.processor(),
           'dependencies':{n:importlib.metadata.version(n) for n in ['numpy','scipy','torch','scikit-learn','joblib','matplotlib']}},
         'runs':[],'metric_rows_independently_checked':0,'reload_parity_checks':0}
    def save():Path(output).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    save()
    for seed in SEEDS:
        print('Fitting catalog BPR seed',seed,flush=True)
        cf=BPRRecommender(train,seed=seed)
        # Serving-only oracle exclusion; fitted likes/factors/counts remain retained-train only.
        cf.seen={u:set(seen[u]) for u in cf.users}
        model=CatalogModel(cf,catalog)
        private=audit/f'model-{seed}.joblib';joblib.dump(model,private)
        replay=joblib.load(private)
        scores={method:{group:[] for group in ('all','cold','warm')} for method in METHODS}
        support={g:0 for g in ('all','cold','warm')};pairs={g:0 for g in support}
        coverage={method:set() for method in METHODS};records=[]
        for user,relevant in cohort:
            expected=catalog_ids-seen[user]
            reference_hash=None
            for method in METHODS:
                result=model.recommend(user_id=user,method=method,n=10)
                assert result==replay.recommend(user_id=user,method=method,n=10)
                out['reload_parity_checks']+=1
                ranking=[r['item'] for r in result['ranking']]
                assert len(set(ranking))==len(ranking) and set(ranking)<=expected
                assert result['candidate_count']==len(expected)
                candidate_hash=sha('\n'.join(sorted(set(model.items)-set(result['seen']))).encode())
                if reference_hash is None:reference_hash=candidate_hash
                assert candidate_hash==reference_hash
                coverage[method].update(ranking)
                metrics={}
                for group,rel in [('all',relevant),('cold',relevant&cold),('warm',relevant-cold)]:
                    if not rel:continue
                    metric=ranking_metrics(ranking,rel)
                    oracle=independent_metrics(ranking,rel)
                    np.testing.assert_allclose([metric['ndcg'],metric['recall']],list(oracle.values()),atol=1e-14,rtol=0)
                    scores[method][group].append(metric);metrics[group]=metric
                    out['metric_rows_independently_checked']+=1
                    if method==METHODS[0]:support[group]+=1;pairs[group]+=len(rel)
                records.append({'user':user,'relevant':sorted(relevant),'candidate_sha256':candidate_hash,'result':result,'metrics':metrics})
        summary={}
        for method in METHODS:
            summary[method]={}
            for group,items in [('all',catalog_ids),('cold',cold),('warm',catalog_ids-cold)]:
                values=scores[method][group]
                summary[method][group]={'eligible_users':len(values),'relevant_pairs':pairs[group],
                    'ndcg_at_10':float(np.mean([r['ndcg'] for r in values])) if values else None,
                    'recall_at_10':float(np.mean([r['recall'] for r in values])) if values else None,
                    'coverage_denominator_items':len(items),'unique_recommended_items':len(coverage[method]&items),
                    'coverage':len(coverage[method]&items)/len(items)}
        (audit/f'predictions-{seed}.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
        out['runs'].append({'seed':seed,'training':cf.training,'results':summary,'genre_features':len(model.vocabulary)})
        save();print(json.dumps({'seed':seed,'results':summary}),flush=True)
    out['mean_results']={method:{group:{metric:float(np.mean([r['results'][method][group][metric] for r in out['runs']]))
                         for metric in ['ndcg_at_10','recall_at_10','coverage']} for group in ['all','cold','warm']} for method in METHODS}
    (audit/'partition.json').write_text(json.dumps({'cold':sorted(cold),'warm':sorted(catalog_ids-cold),'users':[u for u,r in cohort]})+'\n',encoding='utf-8')
    out['restricted_audit_hashes']={p.name:sha(p.read_bytes()) for p in audit.iterdir() if p.is_file()}
    out['observed_run_seconds']=time.perf_counter()-started;out['status']='completed';out['completed_at']=datetime.now(timezone.utc).isoformat();save()
    print(json.dumps(out['mean_results'],indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',required=True);p.add_argument('--audit-dir',required=True);p.add_argument('--output',default='experiments/2026-10-30/results.json');a=p.parse_args();main(a.archive,a.audit_dir,a.output)
