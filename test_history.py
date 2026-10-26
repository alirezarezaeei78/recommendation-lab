import csv
import io
import json
import threading
import unittest
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer
import numpy as np
from app import handler_for
from history_policy import HistoryRecommender
from recommender import synthetic_rows, csv_export, Row
from benchmark_history import public_bin_summary


class HistoryChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.model=HistoryRecommender(synthetic_rows(),epochs=5)
    def test_boundary_uses_unique_positive_count(self):
        for liked,expected in [([],True),(['S01','S02','S02'],True),(['S01','S02','S03'],False)]:
            r=self.model.recommend(liked=liked,method='history_auto')
            self.assertEqual(r['fallback'],expected)
            self.assertEqual(r['history_positive_count'],len(set(liked)))
    def test_centroid_matches_independent_scalar_oracle(self):
        liked=['S01','S02','S03'];r=self.model.recommend(liked,method='factor_centroid')
        for row in r['ranking']:
            q=self.model.item_factors[self.model.index[row['item']]]
            expected=sum(float(q[d])*sum(float(self.model.item_factors[self.model.index[i],d]) for i in liked)/len(liked) for d in range(len(q)))
            self.assertAlmostEqual(row['score'],expected,places=6)
    def test_no_hidden_fitted_user_vector(self):
        before=self.model.recommend(method='history_auto',user_id='D00')
        saved=self.model.user_factors.copy()
        try:
            self.model.user_factors[:]=1000
            self.assertEqual(before,self.model.recommend(method='history_auto',user_id='D00'))
        finally:self.model.user_factors[:]=saved
    def test_unknown_user_safe_history_and_unknown_items(self):
        self.assertEqual(self.model.recommend(method='history_auto',user_id='new-user')['effective_method'],'popularity')
        r=self.model.recommend(['S01','S02','S03'],method='history_auto',user_id='new-user')
        self.assertEqual(r['history_source'],'unknown_user_supplied_history');self.assertFalse(r['fallback'])
        with self.assertRaises(ValueError):self.model.recommend(['missing'],method='history_auto')
    def test_seen_exclusion_all_methods_same_candidates(self):
        for method in ['popularity','item_knn','factor_centroid','history_auto']:
            r=self.model.recommend(['S01','S02','S03'],['S05'],method)
            self.assertEqual(r['candidate_count'],len(self.model.items)-4)
            ids=[x['item'] for x in r['ranking']]
            self.assertEqual(len(ids),len(set(ids)));self.assertFalse(set(ids)&{'S01','S02','S03','S05'})
    def test_known_profile_conflict_and_invalid_request(self):
        with self.assertRaises(ValueError):self.model.recommend(['S01'],method='history_auto',user_id='D00')
        for n in [True,0,101,1.5]:
            with self.assertRaises(ValueError):self.model.recommend(method='history_auto',n=n)
    def test_empty_candidates_finite_and_repeatable(self):
        self.assertEqual(self.model.recommend(method='history_auto',seen=self.model.items)['ranking'],[])
        a=self.model.recommend(['S02','S01','S03'],method='history_auto')
        self.assertEqual(a,self.model.recommend(['S03','S01','S02'],method='history_auto'))
    def test_no_learned_positive_pairs_do_not_use_random_vectors(self):
        model=HistoryRecommender([Row('U',i,2,1) for i in ['A','B','C','D']],epochs=2)
        r=model.recommend(['A','B','C'],method='history_auto')
        self.assertTrue(r['fallback']);self.assertEqual(r['effective_method'],'popularity')
        self.assertIn('No eligible positive profiles',r['fallback_reason'])
    def test_policy_csv_metadata_and_full_precision(self):
        for liked in [[],['S01','S02','S03']]:
            r=self.model.recommend(liked,method='history_auto')
            rows=list(csv.DictReader(io.StringIO(csv_export(r))))
            for a,b in zip(rows,r['ranking']):
                self.assertEqual(float(a['score']),b['score'])
                self.assertEqual(a['effective_method'],r['effective_method'])
                self.assertEqual(int(a['history_positive_count']),len(liked))
                self.assertEqual(a['fallback'],str(r['fallback']))
    def test_small_public_bin_suppression(self):
        r=self.model.recommend(method='history_auto')
        one=[{'result':r,'retained':0,'metrics':{'ndcg':.25,'recall':.1}}]
        public=public_bin_summary(one,len(self.model.items))
        self.assertEqual(public['eligible_users'],1)
        self.assertTrue(public['metrics_suppressed']);self.assertNotIn('ndcg_at_10',public)


class HistoryAPIWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=HistoryRecommender(synthetic_rows(),epochs=5)
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(cls.model,True))
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
        cls.url=f'http://127.0.0.1:{cls.server.server_port}/api/recommend'
    @classmethod
    def tearDownClass(cls):cls.server.shutdown();cls.server.server_close();cls.thread.join()
    def request(self,args):
        return urllib.request.urlopen(urllib.request.Request(self.url,json.dumps(args).encode(),{'Content-Type':'application/json'}))
    def test_new_user_and_exact_api_export(self):
        for args in [{'method':'history_auto','user_id':'new-user'},
                     {'method':'history_auto','liked':['S01','S02','S03'],'seen':['S05']}]:
            expected=self.model.recommend(**args);actual=json.load(self.request(args))
            self.assertEqual(actual['ranking'],expected['ranking']);self.assertEqual(actual['csv'],csv_export(expected))
    def test_malformed_unknown_items_and_no_stale_success(self):
        for args in [{'method':'history_auto','liked':['missing']},{'method':'history_auto','user_id':[]}]:
            with self.assertRaises(urllib.error.HTTPError) as cm:self.request(args)
            self.assertEqual(cm.exception.code,400)
