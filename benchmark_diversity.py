"""Predeclared fixed-factor MMR sweep. Restricted provider rows/ranks never enter public outputs."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import time
import numpy as np
from recommender import Row,validate_rows,ranking_metrics
from history_policy import load_fixed_model
from diversity import DiversityRecommender

LAMBDAS=(1,.75,.5,.25,0)
SEEDS=(13,29,47)


def sha(data):return hashlib.sha256(data).hexdigest()


def scalar_oracles(ids,relevant,tags):
    dcg=math.fsum(1/math.log2(i+2) for i,item in enumerate(ids) if item in relevant)
    ideal=math.fsum(1/math.log2(i+2) for i in range(min(10,len(relevant))))
    distances=[]
    for i,a in enumerate(ids):
        for b in ids[i+1:]:
            left,right=set(tags[a]),set(tags[b])
            distances.append(1-len(left&right)/math.sqrt(len(left)*len(right)))
    return dcg/ideal,len(set(ids)&relevant)/len(relevant),math.fsum(distances)/len(distances)


def summarize(records,catalog):
    selected={item for r in records for item in r['ids']}
    return {'eligible_users':len(records),'ndcg_at_10':float(np.mean([r['ndcg'] for r in records])),
        'recall_at_10':float(np.mean([r['recall'] for r in records])),
        'intra_list_diversity':float(np.mean([r['ild'] for r in records])),
        'catalog_coverage':len(selected)/len(catalog),'unique_recommended_items':len(selected),
        'catalog_denominator':len(catalog),'fallback_users':sum(r['fallback'] for r in records)}


def main(data,items,factors,audit,output):
    started=time.perf_counter();data,items=Path(data),Path(items);old=json.loads(Path('results/2026-10-23.json').read_text())
    raw=data.read_bytes();assert sha(raw)==old['input_sha256']
    rows=validate_rows([Row(u,i,float(r),int(t)) for u,i,r,t in (line.split() for line in raw.decode('ascii').splitlines())])
    train=[r for r in rows if r.timestamp<=old['cutoff_unix_seconds']];future_rows=[r for r in rows if r.timestamp>old['cutoff_unix_seconds']]
    for key,part in [('train_rows_sha256',train),('test_rows_sha256',future_rows)]:
        assert sha('\n'.join(f'{r.user}\t{r.item}\t{r.rating}\t{r.timestamp}' for r in part).encode())==old[key]
    tags={}
    for line in items.read_text(encoding='latin-1').splitlines():
        fields=line.split('|');assert len(fields)==24
        tags[fields[0]]=[f'genre-{i}' for i,x in enumerate(fields[5:]) if int(x)==1]
    future=defaultdict(set)
    for r in future_rows:
        if r.rating>=4:future[r.user].add(r.item)
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True);audit=Path(audit);audit.mkdir(parents=True,exist_ok=True)
    result={'date':'2026-11-02','status':'running','new_benchmark_training_fits':0,'lambdas':list(LAMBDAS),'seeds':list(SEEDS),
        'input_sha256':sha(raw),'metadata_sha256':sha(items.read_bytes()),'protocol_sha256':sha(Path('experiments/2026-11-02/PROTOCOL.md').read_bytes()),
        'train_rows_sha256':old['train_rows_sha256'],'test_rows_sha256':old['test_rows_sha256'],'cutoff_unix_seconds':old['cutoff_unix_seconds'],
        'exposure':'Exploratory reuse of known19/23/26October benchmark; not independent validation.',
        'data_attribution':old['data_attribution'],'data_license':'GroupLens research terms; no provider data redistribution',
        'similarity':'Binary-genre cosine; static metadata assumed available at cutoff, not historically verified',
        'ild_definition':'Mean of45pairwise genre-cosine distances for each10-item list, averaged over users',
        'normalization':'Minmax raw relevance across same full unseen candidates; constant=0',
        'first_item_rule':'Highest raw relevance including at lambda0','runtime_claim':False,'confidence_interval':None,
        'hardware':{'system':platform.platform(),'processor':platform.processor()},'python':platform.python_version(),
        'dependencies':{n:importlib.metadata.version(n) for n in ['numpy','scipy','torch','matplotlib']},'runs':[]}
    for seed in SEEDS:
        factor=Path(factors)/f'factors-{seed}.npz';assert sha(factor.read_bytes())==old['restricted_audit_sha256'][factor.name]
        model=load_fixed_model(train,factor);engine=DiversityRecommender(model,tags);catalog=set(model.items)
        cohort=[]
        for user in sorted({r.user for r in future_rows}):
            relevant=(future[user]&catalog)-model.seen.get(user,set())
            if relevant:cohort.append((user,relevant,user in model.user_index))
        assert len(cohort)==287 and sum(w for _,_,w in cohort)==95
        result['cohort']={'warm_primary':95,'cold_secondary':192,'all_eligible':287,'catalog_items':len(catalog),
            'excluded_future_users_without_eligible_positive':old['future_users_without_eligible_positive'],
            'future_positive_pairs_outside_train_catalog':old['future_positive_pairs_outside_train_catalog']}
        run={'seed':seed,'factor_sha256':sha(factor.read_bytes()),'conditions':[]};baseline_ids={};candidate_hashes={}
        with (audit/f'predictions-{seed}.jsonl').open('w',encoding='utf-8') as stream:
            for value in LAMBDAS:
                records=[]
                for user,relevant,warm in cohort:
                    got=engine.recommend(user,value);ids=[r['item'] for r in got['ranking']]
                    candidates=catalog-model.seen.get(user,set());h=sha('\n'.join(sorted(candidates)).encode())
                    assert len(ids)==10 and len(set(ids))==10 and set(ids)<=candidates and got['candidate_count']==len(candidates)
                    if value==1:
                        base=model.recommend(method='bpr',user_id=user if warm else None)
                        assert ids==[r['item'] for r in base['ranking']]
                        assert [r['score'] for r in got['ranking']]==[r['score'] for r in base['ranking']]
                        baseline_ids[user]=ids;candidate_hashes[user]=h
                    assert candidate_hashes[user]==h
                    metric=ranking_metrics(ids,relevant);oracle=scalar_oracles(ids,relevant,tags)
                    assert np.allclose([metric['ndcg'],metric['recall'],got['intra_list_diversity']],oracle,rtol=0,atol=1e-12)
                    row={'user':user,'warm':warm,'lambda':value,'ids':ids,'ndcg':metric['ndcg'],'recall':metric['recall'],'ild':got['intra_list_diversity'],'fallback':got['fallback']}
                    records.append(row);stream.write(json.dumps(dict(row,relevant=sorted(relevant),candidate_sha256=h,result=got),separators=(',',':'))+'\n')
                warm_records=[r for r in records if r['warm']]
                condition={'lambda':value,'warm_primary':summarize(warm_records,catalog),'all_eligible_secondary':summarize(records,catalog)}
                if value==1:
                    prior=next(x for x in old['bpr_runs'] if x['seed']==seed)['warm_primary']
                    for metric in ['ndcg_at_10','recall_at_10','catalog_coverage']:assert math.isclose(condition['warm_primary'][metric],prior[metric],abs_tol=1e-12)
                run['conditions'].append(condition);print(json.dumps({'seed':seed,**condition}),flush=True)
                if time.perf_counter()-started>1200:raise TimeoutError('20minute evaluation budget exceeded')
        result['runs'].append(run);out.write_text(json.dumps(result,indent=2)+'\n')
    result['across_seed_descriptive']=[]
    for value in LAMBDAS:
        summaries=[next(c for c in run['conditions'] if c['lambda']==value)['warm_primary'] for run in result['runs']]
        result['across_seed_descriptive'].append({'lambda':value,**{m:{'mean':float(np.mean([s[m] for s in summaries])),'min':min(s[m] for s in summaries),'max':max(s[m] for s in summaries)} for m in ['ndcg_at_10','recall_at_10','intra_list_diversity','catalog_coverage']}})
    result.update(status='completed',candidate_and_no_seen_checks_passed=True,baseline_reproduction_passed=True,
        independent_scalar_oracles_passed=True,finished_at=datetime.now(timezone.utc).isoformat(),wall_seconds_observed=time.perf_counter()-started,
        restricted_audit_sha256={f.name:sha(f.read_bytes()) for f in audit.iterdir() if f.is_file()})
    out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','items','factors','audit','output']:p.add_argument('--'+name,required=True)
    args=p.parse_args();main(args.data,args.items,args.factors,args.audit,args.output)
