"""Authored compact BPR-MF variant; existing item-kNN/popularity stay available."""
import math
import time
import numpy as np
import torch
from recommender import Recommender


def pair_loss(p, qi, qj, regularization=1e-4):
    margin = (p * (qi - qj)).sum(dim=-1)
    penalty = (p.square().sum(-1) + qi.square().sum(-1) + qj.square().sum(-1)).mean()
    return torch.nn.functional.softplus(-margin).mean() + regularization * penalty


def sample_negatives(users, seen_matrix, rng):
    """Uniform training-catalog draws excluding every training-seen item."""
    n_items = seen_matrix.shape[1]
    if any(seen_matrix[u].all() for u in np.unique(users)):
        raise ValueError('A training user has no unseen negative candidates')
    negatives = rng.integers(n_items, size=len(users))
    bad = seen_matrix[users, negatives]
    attempts = 0
    while bad.any() and attempts < 100:
        negatives[bad] = rng.integers(n_items, size=int(bad.sum()))
        bad = seen_matrix[users, negatives]
        attempts += 1
    # Bounded rejection: direct uniform draw also handles nearly full histories.
    for u in np.unique(users[bad]):
        mask = bad & (users == u)
        negatives[mask] = rng.choice(np.flatnonzero(~seen_matrix[u]), size=int(mask.sum()))
    return negatives


class BPRRecommender(Recommender):
    def __init__(self, rows, seed=13, factors=32, epochs=30, batch_size=2048,
                 learning_rate=.01, regularization=1e-4, wall_seconds=1200):
        if len({r.user for r in rows}) > 10000:
            raise ValueError('BPR local runner supports at most10000 training users')
        super().__init__(rows)
        if (not isinstance(seed, int) or not isinstance(factors, int) or factors < 1
                or not isinstance(epochs, int) or epochs < 1 or not isinstance(batch_size, int)
                or batch_size < 1 or not math.isfinite(learning_rate) or learning_rate <= 0
                or not math.isfinite(regularization) or regularization < 0):
            raise ValueError('Invalid BPR training parameters')
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        self.user_index = {u: i for i, u in enumerate(self.users)}
        self.seen_matrix = np.zeros((len(self.users), len(self.items)), dtype=bool)
        pairs = []
        self.trained_users = set()
        for u in self.users:
            row = self.user_index[u]
            self.seen_matrix[row, [self.index[i] for i in self.seen[u]]] = True
            if self.likes[u] and len(self.seen[u]) < len(self.items):
                self.trained_users.add(u)
                pairs.extend((row, self.index[i]) for i in sorted(self.likes[u]))
        p = torch.nn.Parameter(torch.randn(len(self.users), factors) * .05)
        q = torch.nn.Parameter(torch.randn(len(self.items), factors) * .05)
        optimizer = torch.optim.Adam([p, q], lr=learning_rate)
        self.training = {'seed':seed, 'factors':factors, 'epochs_requested':epochs,
            'batch_size':batch_size, 'learning_rate':learning_rate, 'regularization':regularization,
            'positive_pairs':len(pairs), 'trained_users':len(self.trained_users), 'history':[],
            'stop_reason':'epoch_limit', 'device':'cpu', 'torch_threads':torch.get_num_threads()}
        started = time.perf_counter()
        if pairs:
            pair_array = np.asarray(pairs, dtype=np.int64)
            for epoch in range(1, epochs+1):
                shuffled = pair_array[rng.permutation(len(pair_array))]
                negatives = sample_negatives(shuffled[:, 0], self.seen_matrix, rng)
                assert not self.seen_matrix[shuffled[:, 0], negatives].any()
                weighted_loss = 0.0
                for start in range(0, len(shuffled), batch_size):
                    current = shuffled[start:start+batch_size]
                    u, i = torch.from_numpy(current.T.copy())
                    j = torch.from_numpy(negatives[start:start+batch_size])
                    optimizer.zero_grad(set_to_none=True)
                    loss = pair_loss(p[u], q[i], q[j], regularization)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite BPR objective')
                    loss.backward()
                    if not torch.isfinite(p.grad).all() or not torch.isfinite(q.grad).all():
                        raise FloatingPointError('Nonfinite BPR gradient')
                    optimizer.step()
                    weighted_loss += float(loss.detach()) * len(current)
                self.training['history'].append({'epoch':epoch, 'sampled_training_objective':weighted_loss/len(shuffled)})
                if time.perf_counter()-started >= wall_seconds:
                    self.training['stop_reason'] = 'wall_budget'
                    break
        else:
            self.training['stop_reason'] = 'no_eligible_positive_pairs'
        self.user_factors = p.detach().numpy().copy()
        self.item_factors = q.detach().numpy().copy()
        if not np.isfinite(self.user_factors).all() or not np.isfinite(self.item_factors).all():
            raise FloatingPointError('Nonfinite learned factors')
        self.training['epochs_executed'] = len(self.training['history'])
        self.training['wall_seconds_observed'] = time.perf_counter()-started

    def recommend(self, liked=None, seen=None, method='item_knn', n=10, user_id=None):
        if user_id is not None:
            if user_id not in self.user_index:
                raise ValueError('Unknown training-user ID')
            if liked or seen:
                raise ValueError('Choose a training user OR custom item history')
            liked, seen = self.likes[user_id], self.seen[user_id]
        if method != 'bpr':
            return super().recommend(liked, seen, method, n)
        if not isinstance(n, int) or not 1 <= n <= 100:
            raise ValueError('n must be1..100')
        if user_id is None and (liked or seen):
            raise ValueError('BPR requires a selected training-user profile; custom histories use item-kNN/popularity')
        if user_id not in self.trained_users:
            result = super().recommend(liked, seen, 'popularity', n)
            result.update(requested_method='bpr', fallback=True,
                fallback_reason='No fitted BPR user profile; explicit popularity fallback')
            return result
        excluded = set(seen or []) | set(liked or [])
        candidates = [i for i, item in enumerate(self.items) if item not in excluded]
        scores = self.item_factors @ self.user_factors[self.user_index[user_id]]
        order = sorted(candidates, key=lambda i:(-scores[i], -self.popularity[i], self.items[i]))[:n]
        return {'requested_method':'bpr', 'effective_method':'bpr', 'fallback':False,
            'user_id':user_id, 'candidate_count':len(candidates), 'liked':sorted(liked or []),
            'seen':sorted(excluded), 'ranking':[{'rank':rank, 'item':self.items[i],
            'score':float(scores[i]), 'positive_users':int(self.popularity[i]), 'contributors':[]}
            for rank,i in enumerate(order,1)]}
