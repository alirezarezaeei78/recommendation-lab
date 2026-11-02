"""Inspectable greedy MMR over fixed model scores and declared metadata tags."""
import csv
import io
import math
import numpy as np
from recommender import ID


def validate_tags(items, catalog):
    if not isinstance(catalog, dict) or set(items) - set(catalog):
        raise ValueError('Catalog must provide nonempty tag lists for every trained item')
    result = {}
    for item in items:
        tags = catalog[item]
        if (not isinstance(tags, list) or not tags or
                any(not isinstance(t, str) or not t.strip() or len(t) > 64 for t in tags)):
            raise ValueError('Tags must be nonempty strings, at most64 characters each')
        result[item] = sorted(set(t.strip() for t in tags))
    return result


def tag_similarity(items, catalog):
    tags = validate_tags(items, catalog)
    vocab = sorted({t for values in tags.values() for t in values})
    index = {t:i for i,t in enumerate(vocab)}
    vectors = np.zeros((len(items), len(vocab)), dtype=np.float64)
    for i,item in enumerate(items):
        vectors[i, [index[t] for t in tags[item]]] = 1
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.clip(vectors @ vectors.T, 0, 1)


def greedy_mmr(items, scores, popularity, similarity, lambda_value=1.0, n=10):
    """Returns selection diagnostics; all arrays refer to the same unseen candidates.

    Normalize score min/max within this candidate set. Constant scores map to0.
    First selection maximizes relevance (also at lambda0). Subsequent utility is
    lambda*normalized_relevance - (1-lambda)*max_similarity_to_selected.
    Ties use raw relevance, training popularity, then ascending ASCII item ID.
    """
    if (isinstance(lambda_value, bool) or not isinstance(lambda_value, (int,float)) or
            not math.isfinite(lambda_value) or not 0 <= lambda_value <= 1):
        raise ValueError('lambda must be a finite number in0..1')
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 100:
        raise ValueError('n must be1..100')
    size = len(items)
    if len(set(items)) != size or any(not isinstance(i,str) or not ID.fullmatch(i) for i in items):
        raise ValueError('Candidate IDs must be valid and unique')
    scores, popularity = np.asarray(scores,dtype=float), np.asarray(popularity,dtype=float)
    similarity = np.asarray(similarity,dtype=float)
    if (scores.shape != (size,) or popularity.shape != (size,) or
            similarity.shape != (size,size) or not np.isfinite(scores).all() or
            not np.isfinite(popularity).all() or not np.isfinite(similarity).all() or
            (popularity < 0).any() or (similarity < 0).any() or (similarity > 1).any() or
            not np.allclose(similarity,similarity.T,atol=1e-12)):
        raise ValueError('Require finite aligned scores, nonnegative popularity and symmetric0..1 similarities')
    if not size:
        return []
    base = sorted(range(size), key=lambda i:(-scores[i],-popularity[i],items[i]))
    ranks = {i:rank for rank,i in enumerate(base,1)}
    span = float(scores.max()-scores.min())
    if not math.isfinite(span):
        raise ValueError('Score range overflows')
    relevance = (scores-scores.min())/span if span else np.zeros(size)
    remaining = set(range(size)); max_similarity = np.zeros(size); chosen = []
    for rank in range(1,min(n,size)+1):
        utility = lambda_value*relevance-(1-lambda_value)*max_similarity
        j = base[0] if rank==1 else min(remaining,key=lambda i:(-utility[i],-scores[i],-popularity[i],items[i]))
        chosen.append({'rank':rank,'item':items[j],'score':float(scores[j]),
            'base_rank':ranks[j],'normalized_relevance':float(relevance[j]),
            'max_selected_similarity':float(max_similarity[j]),'mmr_utility':float(utility[j]),
            'positive_users':int(popularity[j])})
        remaining.remove(j);max_similarity=np.maximum(max_similarity,similarity[:,j])
    return chosen


def intra_list_diversity(ids, catalog):
    """Mean pairwise cosine distance of declared binary tag vectors; no user metric."""
    if len(ids)<2:return None
    sim=tag_similarity(ids,catalog)
    return float(np.mean((1-sim)[np.triu_indices(len(ids),1)]))


class DiversityRecommender:
    def __init__(self, model, catalog):
        self.model=model
        self.catalog=validate_tags(model.items,catalog)
        self.similarity=tag_similarity(model.items,self.catalog)

    def recommend(self, user_id=None, lambda_value=1.0, n=10):
        if user_id is not None and (not isinstance(user_id,str) or not ID.fullmatch(user_id)):
            raise ValueError('Invalid user ID')
        model=self.model
        fitted=user_id in model.trained_users
        if fitted:
            seen=model.seen[user_id]
            full_scores=model.item_factors @ model.user_factors[model.user_index[user_id]]
        else:
            seen=model.seen.get(user_id,set())
            full_scores=model.popularity
        indices=[i for i,item in enumerate(model.items) if item not in seen]
        ids=[model.items[i] for i in indices]
        ranking=greedy_mmr(ids,full_scores[indices],model.popularity[indices],
            self.similarity[np.ix_(indices,indices)],lambda_value,n)
        return {'requested_method':'bpr_mmr','effective_method':'bpr' if fitted else 'popularity',
            'fallback':not fitted,'fallback_reason':None if fitted else 'No fitted user profile; training-popularity scores reranked',
            'user_id':user_id,'lambda_value':lambda_value,'candidate_count':len(ids),
            'normalization':'minmax over all unseen candidates; constant scores map to0',
            'similarity':'binary-tag cosine','first_item_rule':'highest base relevance',
            'seen':sorted(seen),'ranking':ranking,
            'intra_list_diversity':intra_list_diversity([r['item'] for r in ranking],self.catalog)}


def diversity_csv(result):
    stream=io.StringIO(newline='');writer=csv.writer(stream,lineterminator='\n')
    columns=['rank','item','score','base_rank','normalized_relevance','max_selected_similarity','mmr_utility','positive_users']
    writer.writerow(columns+['lambda_value','effective_method','fallback'])
    for row in result['ranking']:
        writer.writerow([format(row[k],'.17g') if isinstance(row[k],float) else row[k] for k in columns]+
            [result['lambda_value'],result['effective_method'],result['fallback']])
    return stream.getvalue()
