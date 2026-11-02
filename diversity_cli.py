"""Local relevance/diversity workflow; default data are independently authored synthetic inputs."""
import argparse
import json
from pathlib import Path
from bpr import BPRRecommender
from recommender import synthetic_rows,read_rows
from catalog_model import synthetic_catalog
from diversity import DiversityRecommender,diversity_csv


def run(args):
    if bool(args.data)!=bool(args.catalog):raise ValueError('Provide both --data and --catalog, or neither')
    rows=read_rows(args.data) if args.data else synthetic_rows()
    tags=json.loads(Path(args.catalog).read_text(encoding='utf-8')) if args.catalog else synthetic_catalog()
    model=BPRRecommender(rows,seed=13)
    engine=DiversityRecommender(model,tags)
    result=engine.recommend(args.user_id,args.lambda_value,args.n)
    result.update(synthetic=not bool(args.data),training=model.training)
    if args.output:
        prefix=Path(args.output);prefix.parent.mkdir(parents=True,exist_ok=True)
        prefix.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        prefix.with_suffix('.csv').write_text(diversity_csv(result),encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2,allow_nan=False))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data');p.add_argument('--catalog',help='JSON mapping each item to a nonempty list of tags')
    p.add_argument('--user-id',default='D00');p.add_argument('--lambda-value',type=float,default=1.0)
    p.add_argument('--n',type=int,default=10);p.add_argument('--output',help='Local JSON/CSV output prefix')
    args=p.parse_args()
    try:run(args)
    except (ValueError,TypeError,KeyError,OSError) as exc:p.error(str(exc))


if __name__=='__main__':main()
