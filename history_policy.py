"""Explicit history policy over already trained item embeddings; no hidden user factor."""
import numpy as np
from bpr import BPRRecommender
from recommender import Recommender, ID

MIN_POSITIVES = 3  # Declared before evaluation; not an optimized threshold.


class HistoryRecommender(BPRRecommender):
    def recommend(self, liked=None, seen=None, method='item_knn', n=10, user_id=None):
        if method not in ('history_auto', 'factor_centroid'):
            return super().recommend(liked, seen, method, n, user_id)
        if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 100:
            raise ValueError('n must be1..100')
        source = 'custom_history'
        if user_id is not None:
            if not isinstance(user_id, str) or not ID.fullmatch(user_id):
                raise ValueError('Invalid user ID')
            if user_id in self.user_index:
                if liked or seen:
                    raise ValueError('Choose a training user OR custom item history')
                liked, seen = self.likes[user_id], self.seen[user_id]
                source = 'known_profile'
            else:
                source = 'unknown_user_supplied_history' if liked or seen else 'unknown_user_no_history'
        liked, seen = set(liked or []), set(seen or [])
        if not liked <= set(self.items) or not seen <= set(self.items):
            raise ValueError('Unknown item ID')
        seen |= liked
        count = len(liked)
        no_fitted_positives = not self.trained_users
        fallback = no_fitted_positives or count < (MIN_POSITIVES if method == 'history_auto' else 1)
        if fallback:
            result = Recommender.recommend(self, liked, seen, 'popularity', n)
            reason = f'{count} unique positive items; policy requires {MIN_POSITIVES}' if method == 'history_auto' else 'No positive items for an embedding centroid'
            if no_fitted_positives:
                reason = 'No eligible positive profiles were fitted; explicit popularity fallback'
            result.update(requested_method=method, fallback=True, fallback_reason=reason)
        else:
            centroid = np.mean(self.item_factors[[self.index[i] for i in sorted(liked)]], axis=0)
            scores = self.item_factors @ centroid
            if not np.isfinite(scores).all():
                raise FloatingPointError('Nonfinite centroid scores')
            candidates = [i for i, item in enumerate(self.items) if item not in seen]
            order = sorted(candidates, key=lambda i: (-scores[i], -self.popularity[i], self.items[i]))[:n]
            result = {'requested_method':method, 'effective_method':'factor_centroid',
                'fallback':False, 'fallback_reason':None, 'candidate_count':len(candidates),
                'liked':sorted(liked), 'seen':sorted(seen), 'ranking':[
                    {'rank':rank, 'item':self.items[i], 'score':float(scores[i]),
                     'positive_users':int(self.popularity[i]), 'contributors':[]}
                    for rank,i in enumerate(order,1)]}
        result.update(history_positive_count=count, policy_min_positives=MIN_POSITIVES,
            history_source=source, uses_fitted_user_vector=False)
        return result


def load_fixed_model(rows, factor_path):
    """Reload private research factors with canonical train-derived ID mapping."""
    model = HistoryRecommender.__new__(HistoryRecommender)
    Recommender.__init__(model, rows)
    model.user_index = {u:i for i,u in enumerate(model.users)}
    model.trained_users = {u for u in model.users if model.likes[u] and len(model.seen[u]) < len(model.items)}
    with np.load(factor_path, allow_pickle=False) as arrays:
        model.user_factors, model.item_factors = arrays['p'].copy(), arrays['q'].copy()
    if (model.user_factors.ndim != 2 or model.item_factors.ndim != 2 or
            model.user_factors.shape[0] != len(model.users) or
            model.item_factors.shape != (len(model.items),model.user_factors.shape[1]) or
            not np.isfinite(model.user_factors).all() or not np.isfinite(model.item_factors).all()):
        raise ValueError('Fixed factor dimensions/values do not match canonical training data')
    return model
