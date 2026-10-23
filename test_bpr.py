import copy
import csv
import io
import json
import math
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer
from pathlib import Path
import numpy as np
import torch
from app import handler_for
from bpr import BPRRecommender, pair_loss, sample_negatives
from recommender import Row, csv_export, synthetic_rows


class BPRChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=synthetic_rows()
        cls.model=BPRRecommender(cls.rows,epochs=5)

    def test_independent_pair_loss_and_finite_difference_gradient(self):
        arrays=[np.array([[.2,-.4]]),np.array([[.3,.1]]),np.array([[-.2,.5]])]
        tensors=[torch.tensor(x,dtype=torch.float64,requires_grad=True) for x in arrays]
        loss=pair_loss(*tensors,regularization=.01)
        def objective(values):
            p,a,b=values
            margin=sum(float(p[0,k])*(float(a[0,k])-float(b[0,k])) for k in range(2))
            return math.log1p(math.exp(-margin))+.01*sum(float(v*v) for x in values for v in x.ravel())
        self.assertAlmostEqual(float(loss),objective(arrays),places=12)
        loss.backward()
        for a,tensor in enumerate(tensors):
            for coordinate in range(2):
                plus=copy.deepcopy(arrays);minus=copy.deepcopy(arrays)
                plus[a][0,coordinate]+=1e-6;minus[a][0,coordinate]-=1e-6
                numerical=(objective(plus)-objective(minus))/2e-6
                self.assertAlmostEqual(float(tensor.grad[0,coordinate]),numerical,places=8)

    def test_negatives_exclude_every_seen_rating_and_are_repeatable(self):
        seen=np.array([[True,True,False,False],[False,True,True,False]])
        users=np.tile([0,1],1000)
        a=sample_negatives(users,seen,np.random.default_rng(13))
        b=sample_negatives(users,seen,np.random.default_rng(13))
        np.testing.assert_array_equal(a,b)
        self.assertFalse(seen[users,a].any())
        self.assertEqual(set(a[users==0]),{2,3})
        self.assertEqual(set(a[users==1]),{0,3})

    def test_full_seen_negative_sampling_rejects_without_hanging(self):
        with self.assertRaises(ValueError):
            sample_negatives(np.array([0]),np.ones((1,2),dtype=bool),np.random.default_rng(13))

    def test_same_seed_reordered_rows_exact_factors(self):
        other=BPRRecommender(self.rows[::-1],epochs=5)
        np.testing.assert_array_equal(self.model.user_factors,other.user_factors)
        np.testing.assert_array_equal(self.model.item_factors,other.item_factors)

    def test_same_candidates_for_all_methods_no_seen_or_duplicates(self):
        for method in ['bpr','item_knn','popularity']:
            result=self.model.recommend(user_id='D00',method=method)
            expected=set(self.model.items)-self.model.seen['D00']
            self.assertEqual(result['candidate_count'],len(expected))
            items=[r['item'] for r in result['ranking']]
            self.assertEqual(len(items),len(set(items)))
            self.assertTrue(set(items)<=expected)

    def test_scores_match_independent_scalar_dot_product(self):
        result=self.model.recommend(user_id='D00',method='bpr')
        u=self.model.user_index['D00']
        for item in result['ranking']:
            i=self.model.index[item['item']]
            scalar=sum(float(p)*float(q) for p,q in zip(self.model.user_factors[u],self.model.item_factors[i]))
            self.assertAlmostEqual(scalar,item['score'],places=6)

    def test_explicit_fallback_and_custom_history_validation(self):
        result=self.model.recommend(method='bpr')
        self.assertTrue(result['fallback']);self.assertEqual(result['effective_method'],'popularity')
        with self.assertRaises(ValueError):self.model.recommend(method='bpr',liked=['S01'])
        with self.assertRaises(ValueError):self.model.recommend(method='bpr',user_id='unknown')
        with self.assertRaises(ValueError):self.model.recommend(method='bpr',user_id='D00',seen=['S01'])

    def test_empty_positive_or_all_seen_profiles(self):
        rows=[Row('U','A',2,1),Row('V','A',5,1),Row('V','B',5,1)]
        model=BPRRecommender(rows,epochs=2)
        self.assertEqual(model.training['positive_pairs'],0)
        self.assertTrue(model.recommend(method='bpr',user_id='U')['fallback'])
        self.assertEqual(model.recommend(method='bpr',user_id='V')['ranking'],[])

    def test_export_and_factor_reload_parity(self):
        result=self.model.recommend(user_id='D00',method='bpr')
        exported=list(csv.DictReader(io.StringIO(csv_export(result))))
        for a,b in zip(exported,result['ranking']):
            self.assertEqual(a['item'],b['item']);self.assertEqual(int(a['rank']),b['rank'])
            self.assertEqual(float(a['score']),b['score'])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'factors.npz'
            np.savez(path,p=self.model.user_factors,q=self.model.item_factors)
            with np.load(path) as loaded:
                np.testing.assert_array_equal(loaded['p']@loaded['q'].T,
                    self.model.user_factors@self.model.item_factors.T)

    def test_no_future_interactions_enter_training(self):
        cutoff=110
        past=[r for r in self.rows if r.timestamp<=cutoff]
        future=[Row('NEW','FUTURE',5,999)]
        a=BPRRecommender(past,epochs=2)
        # A later high-rated item is present in the source, absent from all fits.
        source=past+future
        b=BPRRecommender([r for r in source if r.timestamp<=cutoff],epochs=2)
        self.assertNotIn('FUTURE',b.items)
        np.testing.assert_array_equal(a.user_factors,b.user_factors)
        np.testing.assert_array_equal(a.item_factors,b.item_factors)


class BPRAPIWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=BPRRecommender(synthetic_rows(),epochs=5)
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(cls.model,True))
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
        cls.url=f'http://127.0.0.1:{cls.server.server_port}'
    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join()
    def request(self,data):
        return urllib.request.urlopen(urllib.request.Request(self.url+'/api/recommend',
            json.dumps(data).encode(),{'Content-Type':'application/json'}),timeout=5)
    def test_selected_user_api_and_export_parity(self):
        response=json.load(self.request({'method':'bpr','user_id':'D00','n':5}))
        expected=self.model.recommend(method='bpr',user_id='D00',n=5)
        self.assertEqual(response['ranking'],expected['ranking'])
        self.assertEqual(response['csv'],csv_export(expected))
    def test_unknown_and_malformed_user_are_errors(self):
        for user in ['unknown',[],123]:
            with self.assertRaises(urllib.error.HTTPError) as cm:
                self.request({'method':'bpr','user_id':user})
            self.assertEqual(cm.exception.code,400)
    def test_catalog_synthetic_profiles_and_seed(self):
        with urllib.request.urlopen(self.url+'/api/catalog') as response:
            result=json.load(response)
        self.assertTrue(result['synthetic']);self.assertIn('D00',result['users'])
        self.assertEqual(result['bpr_training']['seed'],13)


if __name__=='__main__':unittest.main(verbosity=2)
