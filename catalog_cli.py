"""Local catalog-cold workflow. Defaults are authored synthetic data only."""
import argparse
import json
from pathlib import Path
from bpr import BPRRecommender
from recommender import synthetic_rows,read_rows
from catalog_model import CatalogModel,synthetic_catalog,catalog_csv


def run(args):
    if bool(args.data)!=bool(args.catalog):
        raise ValueError('Provide both --data and --catalog, or neither for synthetic demo')
    rows=read_rows(args.data) if args.data else synthetic_rows()
    catalog=json.loads(Path(args.catalog).read_text(encoding='utf-8')) if args.catalog else synthetic_catalog()
    model=CatalogModel(BPRRecommender(rows,seed=13),catalog)
    if args.new_item:
        item=json.loads(args.new_item)
        if not isinstance(item,dict) or set(item)!={'id','tags'}:
            raise ValueError('--new-item requires exactly id and tags')
        model.add_item(item['id'],item['tags'])
    result=model.recommend(args.liked,args.seen,args.method,args.n,args.user_id)
    result['synthetic']=not bool(args.data)
    result['metadata_vocabulary']=model.vocabulary
    if args.output:
        root=Path(args.output);root.parent.mkdir(parents=True,exist_ok=True)
        root.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        root.with_suffix('.csv').write_text(catalog_csv(result),encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2,allow_nan=False))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data');p.add_argument('--catalog')
    p.add_argument('--new-item',help='JSON: {"id":"NEW","tags":["space"]}')
    p.add_argument('--liked',nargs='*',default=[]);p.add_argument('--seen',nargs='*',default=[])
    p.add_argument('--user-id');p.add_argument('--n',type=int,default=10)
    p.add_argument('--method',choices=['popularity','collaborative','content','hybrid'],default='hybrid')
    p.add_argument('--output',help='Local output prefix; writes matching .json/.csv')
    args=p.parse_args()
    try:run(args)
    except (ValueError,TypeError,KeyError,OSError) as exc:p.error(str(exc))


if __name__=='__main__':main()
