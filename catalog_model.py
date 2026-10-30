"""Catalog-aware genre model and explicit collaborative/content rank fusion."""
import csv
import io
import math
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from recommender import ID


class CatalogModel:
    def __init__(self, collaborative, catalog):
        if not isinstance(catalog, dict) or not 1 <= len(catalog) <= 3000:
            raise ValueError('Catalog requires1..3000 item-to-tag lists')
        for item, tags in catalog.items():
            if (not isinstance(item, str) or not ID.fullmatch(item)
                    or not isinstance(tags, list) or len(tags) > 64
                    or any(not isinstance(t, str) or not ID.fullmatch(t) for t in tags)
                    or len(tags) != len(set(tags))):
                raise ValueError('Valid unique ASCII item IDs/tags required')
        if not set(collaborative.items) <= set(catalog):
            raise ValueError('Catalog must describe every fitted interaction item')
        self.cf = collaborative
        self.items = sorted(catalog)
        self.index = {item:i for i,item in enumerate(self.items)}
        self.vocabulary = sorted({t for tags in catalog.values() for t in tags})
        if not self.vocabulary:
            raise ValueError('At least one metadata tag required')
        self.tag_index = {t:i for i,t in enumerate(self.vocabulary)}
        self.catalog = {i:list(catalog[i]) for i in self.items}
        counts = np.zeros((len(self.items),len(self.vocabulary)))
        for item,tags in self.catalog.items():
            counts[self.index[item], [self.tag_index[t] for t in tags]] = 1
        self.transformer = TfidfTransformer(norm='l2', smooth_idf=True)
        self.vectors = self.transformer.fit_transform(counts).toarray()

    def add_item(self, item, tags):
        """Transform new metadata with frozen vocabulary/IDF, never refit."""
        if not isinstance(item,str) or not ID.fullmatch(item) or item in self.index:
            raise ValueError('New item requires a valid unused ID')
        if (not isinstance(tags,list) or len(tags)>64 or len(tags)!=len(set(tags))
                or any(not isinstance(t,str) or not ID.fullmatch(t) for t in tags)):
            raise ValueError('Valid unique metadata tags required')
        if len(self.items) >= 3000:
            raise ValueError('Catalog cap3000')
        counts = np.zeros((1,len(self.vocabulary)))
        known = [self.tag_index[t] for t in tags if t in self.tag_index]
        counts[0,known] = 1
        self.index[item] = len(self.items)
        self.items.append(item)
        self.catalog[item] = list(tags)
        self.vectors = np.vstack([self.vectors,self.transformer.transform(counts).toarray()])

    def recommend(self, liked=None, seen=None, method='hybrid', n=10, user_id=None):
        if method not in ('popularity','collaborative','content','hybrid'):
            raise ValueError('Unknown catalog method')
        if not isinstance(n,int) or isinstance(n,bool) or not 1 <= n <=100:
            raise ValueError('n must be1..100')
        for values in (liked,seen):
            if values is not None and (not isinstance(values,(list,set,tuple))
                    or any(not isinstance(x,str) for x in values)):
                raise ValueError('History must be a list of item IDs')
        if user_id is not None:
            if not isinstance(user_id,str) or not ID.fullmatch(user_id):
                raise ValueError('Invalid user ID')
            if liked or seen:
                raise ValueError('Choose user_id OR custom history')
            liked = self.cf.likes.get(user_id,set())
            seen = self.cf.seen.get(user_id,set())
        liked,seen = set(liked or []),set(seen or [])
        if not (liked|seen) <= set(self.items):
            raise ValueError('Unknown item ID')
        seen |= liked
        candidates = [i for i in self.items if i not in seen]
        pop = {i:float(self.cf.popularity[self.cf.index[i]]) if i in self.cf.index else 0. for i in self.items}
        content = dict.fromkeys(self.items,0.)
        if liked:
            profile = normalize(np.mean(self.vectors[[self.index[i] for i in sorted(liked)]],axis=0).reshape(1,-1))[0]
            content = dict(zip(self.items,map(float,self.vectors@profile)))
        cf_scores = dict.fromkeys(self.items,0.)
        cf_source = 'none'
        if user_id in self.cf.trained_users:
            query = self.cf.user_factors[self.cf.user_index[user_id]]
            cf_source = 'fitted_user_vector'
        else:
            warm_likes = [i for i in sorted(liked) if i in self.cf.index]
            query = np.mean(self.cf.item_factors[[self.cf.index[i] for i in warm_likes]],axis=0) if warm_likes else None
            if query is not None:
                cf_source = 'custom_warm_factor_centroid'
        if query is not None and self.cf.trained_users:
            cf_scores.update(dict(zip(self.cf.items,map(float,self.cf.item_factors@query))))
        else:
            cf_source='none'
        key = lambda scores: lambda item: (-scores[item],-pop[item],item)
        content_order = sorted([i for i in candidates if content[i]>0],key=key(content))
        cf_order = sorted([i for i in candidates if i in self.cf.index],key=key(cf_scores)) if cf_source!='none' else []
        cr = {i:r for r,i in enumerate(content_order,1)}
        br = {i:r for r,i in enumerate(cf_order,1)}
        effective,reason = method,None
        if method=='content' and not content_order:
            effective,reason='popularity','No positive metadata signal'
        if method=='collaborative' and not cf_order:
            effective,reason='popularity','No collaborative query representation'
        if method=='hybrid':
            if not content_order:
                effective,reason=('collaborative','No positive metadata signal') if cf_order else ('popularity','No metadata or collaborative query signal')
            elif not cf_order:
                effective,reason='content','No collaborative query representation'
        if effective=='popularity':
            scores=pop;order=sorted(candidates,key=key(scores))
        elif effective=='content':
            scores=content;order=sorted(candidates,key=key(scores))
        elif effective=='collaborative':
            scores=cf_scores
            # A cold item has no trained factor, even if a warm score is negative.
            order=sorted(candidates,key=lambda i:(i not in self.cf.index,-scores[i],-pop[i],i))
        else:
            scores={i:(1/(60+cr[i]) if i in cr else 0.)+(1/(60+br[i]) if i in br else 0.) for i in candidates}
            order=sorted(candidates,key=key(scores))
        ranking=[{'rank':r,'item':i,'score':float(scores[i]),'cold_item':i not in self.cf.index,
                  'content_score':content[i],'collaborative_score':cf_scores[i] if i in br else None,
                  'content_rank':cr.get(i),'collaborative_rank':br.get(i),'positive_users':int(pop[i])}
                 for r,i in enumerate(order[:n],1)]
        if any(not math.isfinite(r['score']) for r in ranking):
            raise ValueError('Nonfinite ranking')
        return {'requested_method':method,'effective_method':effective,'fallback_reason':reason,
                'candidate_count':len(candidates),'liked':sorted(liked),'seen':sorted(seen),
                'collaborative_source':cf_source,'fusion_constant':60,'ranking':ranking}


def catalog_csv(result):
    stream=io.StringIO(newline='')
    columns=['rank','item','score','cold_item','content_score','collaborative_score','content_rank','collaborative_rank','positive_users']
    context=['requested_method','effective_method','fallback_reason','candidate_count','collaborative_source']
    writer=csv.DictWriter(stream,fieldnames=columns+context,lineterminator='\n')
    writer.writeheader()
    for row in result['ranking']:
        writer.writerow(dict(row,**{k:result[k] for k in context}))
    return stream.getvalue()


def synthetic_catalog():
    tags=[['space','adventure'],['nature','documentary'],['music','drama']]
    return {f'S{i:02d}':tags[(i-1)//4] for i in range(1,13)}
