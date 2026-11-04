"""Batch recommendations with inspectable JSON/CSV evidence; synthetic by default."""
import argparse
import json
from pathlib import Path
from bpr import BPRRecommender
from recommender import synthetic_rows, read_rows
from batch_serving import BatchServing, batch_csv


def run(args):
    prefix = Path(args.output)
    destinations = [prefix.with_suffix('.json'), prefix.with_suffix('.csv')]
    if any(path.exists() for path in destinations):
        raise ValueError('Output already exists; choose a fresh output prefix')
    queries = json.loads(Path(args.queries).read_text(encoding='utf-8')) if args.queries else [
        {'query_id': 'known-00', 'user_id': 'D00'},
        {'query_id': 'known-01', 'user_id': 'D01'},
        {'query_id': 'cold-user', 'user_id': 'NEW'},
        {'query_id': 'anonymous', 'exclude': ['S01']}]
    model = BPRRecommender(read_rows(args.data) if args.data else synthetic_rows(), seed=13)
    result = BatchServing(model).recommend(queries, args.method, args.n, args.batch_size)
    result['input_provenance'] = 'user_supplied_local_data' if args.data else 'authored_synthetic_demo'
    result['training'] = model.training
    prefix.parent.mkdir(parents=True, exist_ok=True)
    destinations[0].write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    destinations[1].write_text(batch_csv(result), encoding='utf-8', newline='\n')
    print(json.dumps({'query_count': result['query_count'], 'input_provenance': result['input_provenance'],
        'model_fingerprint': result['model_fingerprint'], 'json_and_csv_written': True}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', help='Appropriately licensed local CSV; no external upload')
    parser.add_argument('--queries', help='JSON list of query_id/user_id/optional exclude objects')
    parser.add_argument('--method', choices=['bpr', 'popularity'], default='bpr')
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--output', required=True, help='Fresh output prefix for JSON/CSV files')
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
