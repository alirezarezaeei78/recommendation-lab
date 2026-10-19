"""Small deterministic binary-positive item-kNN; no network or paid API."""
from dataclasses import dataclass
import csv
import io
import math
import re
import numpy as np
from scipy.sparse import csr_matrix

ID = re.compile(r'^[A-Za-z0-9_.-]{1,64}$')

@dataclass(frozen=True)
class Row:
    user: str
    item: str
    rating: float
    timestamp: int

def read_rows(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['user', 'item', 'rating', 'timestamp']:
            raise ValueError('CSV columns must be user,item,rating,timestamp in that order')
        return validate_rows([Row(r['user'], r['item'], float(r['rating']), int(r['timestamp'])) for r in reader])

def validate_rows(rows):
    if not rows or len(rows) > 500000:
        raise ValueError('Require1..500000 rows')
    pairs = set()
    for r in rows:
        if not ID.fullmatch(r.user) or not ID.fullmatch(r.item):
            raise ValueError('IDs must be1..64 ASCII letters/digits/._-')
        if not math.isfinite(r.rating) or not 1 <= r.rating <= 5 or r.timestamp < 0:
            raise ValueError('Require finite rating1..5 and nonnegative integer timestamp')
        if (r.user, r.item) in pairs:
            raise ValueError('Duplicate user-item pair: resolve repeated-event policy before fitting')
        pairs.add((r.user, r.item))
    return rows

def ranking_metrics(ranking, relevant, k=10):
    if k < 1 or not relevant or len(set(ranking)) != len(ranking):
        raise ValueError('Require positive k, nonempty relevance and unique ranking')
    hits = [1 if x in relevant else 0 for x in ranking[:k]]
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return {'ndcg': dcg / ideal, 'recall': sum(hits) / len(relevant)}

class Recommender:
    def __init__(self, rows, neighbors=20, shrinkage=10.0):
        validate_rows(rows)
        if not isinstance(neighbors, int) or neighbors < 1 or not math.isfinite(shrinkage) or shrinkage < 0:
            raise ValueError('Invalid neighborhood parameters')
        self.items = sorted({r.item for r in rows})
        if len(self.items) > 3000:
            raise ValueError('Local engine supports at most3000 items')
        self.index = {v: i for i, v in enumerate(self.items)}
        self.users = sorted({r.user for r in rows})
        user_index = {v: i for i, v in enumerate(self.users)}
        self.seen = {u: set() for u in self.users}
        self.likes = {u: set() for u in self.users}
        rr, cc = [], []
        for r in rows:
            self.seen[r.user].add(r.item)
            if r.rating >= 4:
                self.likes[r.user].add(r.item)
                rr.append(user_index[r.user]); cc.append(self.index[r.item])
        binary = csr_matrix((np.ones(len(rr)), (rr, cc)), shape=(len(self.users), len(self.items)))
        co = (binary.T @ binary).toarray()
        self.popularity = np.diag(co).copy()
        denominator = np.sqrt(self.popularity[:, None] * self.popularity[None, :])
        sim = np.divide(co, denominator, out=np.zeros_like(co), where=denominator > 0)
        if shrinkage:
            sim *= co / (co + shrinkage)
        np.fill_diagonal(sim, 0)
        self.similarity = np.zeros_like(sim)
        for j in range(len(self.items)):
            sources = sorted(range(len(self.items)), key=lambda i: (-sim[i, j], self.items[i]))[:neighbors]
            self.similarity[sources, j] = sim[sources, j]
        if not np.isfinite(self.similarity).all():
            raise ValueError('Nonfinite similarity')
        self.neighbors, self.shrinkage = neighbors, shrinkage

    def recommend(self, liked=None, seen=None, method='item_knn', n=10):
        if method not in ('item_knn', 'popularity') or not isinstance(n, int) or not 1 <= n <= 100:
            raise ValueError('Method must be item_knn/popularity; n must be1..100')
        liked, seen = set(liked or []), set(seen or [])
        if not liked <= set(self.items) or not seen <= set(self.items):
            raise ValueError('Unknown item ID')
        seen |= liked
        candidates = [i for i, item in enumerate(self.items) if item not in seen]
        indices = [self.index[item] for item in sorted(liked)]
        scores = self.similarity[indices].sum(axis=0) if indices else np.zeros(len(self.items))
        fallback = method == 'item_knn' and not any(scores[i] > 0 for i in candidates)
        effective = 'popularity' if method == 'popularity' or fallback else 'item_knn'
        if effective == 'popularity':
            scores = self.popularity
        order = sorted(candidates, key=lambda i: (-scores[i], -self.popularity[i], self.items[i]))[:n]
        out = []
        for rank, i in enumerate(order, 1):
            contributors = sorted([(x, float(self.similarity[self.index[x], i])) for x in liked if self.similarity[self.index[x], i] > 0], key=lambda x: (-x[1], x[0]))
            out.append({'rank': rank, 'item': self.items[i], 'score': float(scores[i]), 'positive_users': int(self.popularity[i]), 'contributors': contributors if effective == 'item_knn' else []})
        return {'requested_method': method, 'effective_method': effective, 'fallback': fallback, 'candidate_count': len(candidates), 'liked': sorted(liked), 'seen': sorted(seen), 'ranking': out}

def csv_export(result):
    stream = io.StringIO(newline='')
    w = csv.writer(stream)
    w.writerow(['rank', 'item', 'score', 'positive_users', 'effective_method'])
    for r in result['ranking']:
        w.writerow([r['rank'], r['item'], format(r['score'], '.17g'), r['positive_users'], result['effective_method']])
    return stream.getvalue()

def synthetic_rows():
    """Authored example interactions; no real users, products or benchmark records."""
    groups = [('S01','S02','S03','S04'), ('S05','S06','S07','S08'), ('S09','S10','S11','S12')]
    rows = []
    for g, items in enumerate(groups):
        for u in range(12):
            for j, item in enumerate(items):
                if (u+j) % 4 != 0:
                    rows.append(Row(f'D{g*12+u:02}', item, 5 if j%2 else 4, 100+u))
            if u % 3 == 0:
                rows.append(Row(f'D{g*12+u:02}', groups[(g+1)%3][0], 3, 100+u))
    return rows
