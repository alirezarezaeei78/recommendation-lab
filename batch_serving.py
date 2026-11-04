"""Bounded, deterministic batch serving over fitted BPR factors; no network I/O."""
import csv
import hashlib
import io
import json
import numpy as np
from recommender import ID

SCHEMA = 'recommendation-batch-v1'


def fixed_order_scores(users, items):
    """Float64 product sums in fixed factor order, independent of query blocking."""
    users, items = np.asarray(users, dtype=np.float64), np.asarray(items, dtype=np.float64)
    if (users.ndim != 2 or items.ndim != 2 or users.shape[1] != items.shape[1]
            or not np.isfinite(users).all() or not np.isfinite(items).all()):
        raise ValueError('Require aligned finite user and item factor matrices')
    scores = np.zeros((len(users), len(items)), dtype=np.float64)
    for dimension in range(users.shape[1]):
        scores += users[:, dimension, None] * items[None, :, dimension]
    if not np.isfinite(scores).all():
        raise ValueError('Factor products overflow the serving score range')
    return scores


class BatchServing:
    """A startup snapshot. Unknown profiles use popularity, never invented factors."""
    def __init__(self, model):
        self.items = tuple(model.items)
        self.index = {item: i for i, item in enumerate(self.items)}
        self.user_index = dict(model.user_index)
        self.trained_users = frozenset(model.trained_users)
        self.seen = {u: frozenset(v) for u, v in model.seen.items()}
        self.likes = {u: frozenset(v) for u, v in model.likes.items()}
        self.user_factors = np.array(model.user_factors, dtype=np.float64, copy=True)
        self.item_factors = np.array(model.item_factors, dtype=np.float64, copy=True)
        self.popularity = np.array(model.popularity, dtype=np.float64, copy=True)
        if (self.user_factors.ndim != 2 or self.item_factors.ndim != 2
                or self.user_factors.shape[0] != len(self.user_index)
                or self.item_factors.shape != (len(self.items), self.user_factors.shape[1])
                or self.popularity.shape != (len(self.items),)
                or not np.isfinite(self.user_factors).all()
                or not np.isfinite(self.item_factors).all()
                or not np.isfinite(self.popularity).all() or (self.popularity < 0).any()):
            raise ValueError('Invalid fitted model dimensions or values')
        digest = hashlib.sha256(json.dumps({'items': self.items,
            'users': sorted(self.user_index.items()), 'trained': sorted(self.trained_users),
            'seen': {u: sorted(v) for u, v in sorted(self.seen.items())}},
            sort_keys=True, separators=(',', ':')).encode())
        for array in (self.user_factors, self.item_factors, self.popularity):
            digest.update(str(array.shape).encode())
            digest.update(array.astype('<f8', copy=False).tobytes())
            array.flags.writeable = False
        self.fingerprint = digest.hexdigest()

    def recommend(self, queries, method='bpr', n=10, batch_size=32):
        if method not in ('bpr', 'popularity'):
            raise ValueError('Batch method must be bpr or popularity')
        if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 100:
            raise ValueError('n must be an integer in1..100')
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 128:
            raise ValueError('batch_size must be an integer in1..128')
        if not isinstance(queries, list) or not 1 <= len(queries) <= 1000:
            raise ValueError('Require1..1000 query objects')
        prepared, query_ids = [], set()
        for query in queries:
            if not isinstance(query, dict) or set(query) - {'query_id', 'user_id', 'exclude'}:
                raise ValueError('Each query allows query_id, user_id and exclude only')
            query_id, user = query.get('query_id'), query.get('user_id')
            if not isinstance(query_id, str) or not ID.fullmatch(query_id) or query_id in query_ids:
                raise ValueError('Query IDs must be valid and unique')
            if user is not None and (not isinstance(user, str) or not ID.fullmatch(user)):
                raise ValueError('Invalid user ID')
            exclude = query.get('exclude', [])
            if (not isinstance(exclude, list) or any(not isinstance(i, str) for i in exclude)
                    or set(exclude) - set(self.items)):
                raise ValueError('Exclusions must be a list of known item IDs')
            query_ids.add(query_id)
            seen = self.seen.get(user, frozenset()) | set(exclude)
            fitted = method == 'bpr' and user in self.trained_users
            reason = None
            if method == 'bpr' and not fitted:
                reason = ('Known profile has no fitted positive factors' if user in self.user_index
                          else 'No fitted user profile; popularity fallback')
            prepared.append({'query_id': query_id, 'user_id': user, 'seen': seen,
                'fitted': fitted, 'fallback_reason': reason})
        results = []
        for start in range(0, len(prepared), batch_size):
            block = prepared[start:start + batch_size]
            fitted = [q for q in block if q['fitted']]
            scores = fixed_order_scores(self.user_factors[
                [self.user_index[q['user_id']] for q in fitted]], self.item_factors)
            row = 0
            for query in block:
                if query['fitted']:
                    values = scores[row]
                    row += 1
                else:
                    values = self.popularity
                candidates = [i for i, item in enumerate(self.items) if item not in query['seen']]
                order = sorted(candidates,
                    key=lambda i: (-values[i], -self.popularity[i], self.items[i]))[:n]
                results.append({'query_id': query['query_id'], 'user_id': query['user_id'],
                    'requested_method': method, 'effective_method': 'bpr' if query['fitted'] else 'popularity',
                    'fallback': method == 'bpr' and not query['fitted'],
                    'fallback_reason': query['fallback_reason'],
                    'status': 'ok' if order else 'no_candidates',
                    'candidate_count': len(candidates),
                    'seen_count': len(query['seen']),
                    'candidate_sha256': hashlib.sha256('\n'.join(self.items[i] for i in candidates).encode()).hexdigest(),
                    'ranking': [{'rank': rank, 'item': self.items[i], 'score': float(values[i]),
                        'positive_users': int(self.popularity[i])} for rank, i in enumerate(order, 1)]})
        return {'schema': SCHEMA, 'model_fingerprint': self.fingerprint,
            'score_policy': 'float64_fixed_factor_order', 'query_count': len(results),
            'results': results}


def batch_csv(result):
    """Status rows preserve empty queries; ranking rows preserve full float precision."""
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    context = ['query_id', 'user_id', 'requested_method', 'effective_method', 'fallback',
               'fallback_reason', 'status', 'candidate_count', 'seen_count', 'candidate_sha256']
    writer.writerow(['row_type', 'schema', 'model_fingerprint'] + context +
                    ['rank', 'item', 'score', 'positive_users'])
    for query in result['results']:
        prefix = [result['schema'], result['model_fingerprint']] + [query[k] for k in context]
        writer.writerow(['query_status'] + prefix + ['', '', '', ''])
        for item in query['ranking']:
            writer.writerow(['ranking'] + prefix + [item['rank'], item['item'],
                format(item['score'], '.17g'), item['positive_users']])
    return stream.getvalue()
