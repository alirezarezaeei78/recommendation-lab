import csv
import io
import json
import math
import threading
import unittest
import urllib.request
import urllib.error
import numpy as np
from http.server import ThreadingHTTPServer
from recommender import Row, Recommender, ranking_metrics, csv_export, synthetic_rows, validate_rows
from app import handler_for

class ModelChecks(unittest.TestCase):
    def setUp(self):
        self.rows=[Row('U1','A',5,1),Row('U1','B',4,1),Row('U2','A',4,1),Row('U2','C',5,1),Row('U3','B',5,1),Row('U3','C',5,1),Row('U4','D',2,1)]
        self.m=Recommender(self.rows)
    def test_independent_cosine_and_shrinkage(self):
        i,j=self.m.index['A'],self.m.index['B']
        self.assertAlmostEqual(self.m.similarity[i,j],(1/math.sqrt(2*2))*(1/11))
        self.assertEqual(self.m.similarity[i,i],0)
    def test_independent_ndcg_and_recall(self):
        x=ranking_metrics(['X','B','A'],{'A','B','C'})
        self.assertAlmostEqual(x['ndcg'],(1/math.log2(3)+1/math.log2(4))/(1+1/math.log2(3)+1/math.log2(4)))
        self.assertAlmostEqual(x['recall'],2/3)
    def test_recall_denominator_not_top_k(self):
        self.assertAlmostEqual(ranking_metrics(['A'],set('ABCDEFGHIJK'),k=1)['recall'],1/11)
    def test_full_candidate_count_and_seen_dislikes(self):
        x=self.m.recommend(['A'],['D'])
        self.assertEqual(x['candidate_count'],2);self.assertEqual({r['item'] for r in x['ranking']},{'B','C'})
    def test_no_repeated_ranking_ids(self):
        x=self.m.recommend(['A','A'],['A']);self.assertEqual(len({r['item'] for r in x['ranking']}),len(x['ranking']))
    def test_cold_history_fallback(self):
        x=self.m.recommend();self.assertTrue(x['fallback']);self.assertEqual(x['effective_method'],'popularity')
    def test_zero_positive_item_is_finite(self):
        self.assertTrue(np.isfinite(self.m.similarity).all());self.assertEqual(self.m.popularity[self.m.index['D']],0)
    def test_all_seen_empty_result(self):
        x=self.m.recommend(seen=list('ABCD'));self.assertEqual(x['ranking'],[]);self.assertEqual(x['candidate_count'],0)
    def test_reorder_invariant(self):
        other=Recommender(self.rows[::-1]);np.testing.assert_allclose(self.m.similarity,other.similarity);self.assertEqual(self.m.recommend(['A']),other.recommend(['A']))
    def test_deterministic_lexical_ties(self):
        self.assertEqual([r['item'] for r in self.m.recommend(['A'])['ranking']][:2],['B','C'])
    def test_top_neighbor_pruning(self):
        m=Recommender(self.rows,neighbors=1);self.assertTrue((np.count_nonzero(m.similarity,axis=0)<=1).all())
    def test_contributor_sum_matches_score(self):
        for r in self.m.recommend(['A','B'])['ranking']:
            self.assertAlmostEqual(r['score'],sum(v for _,v in r['contributors']))
    def test_unknown_input_rejected(self):
        with self.assertRaises(ValueError):self.m.recommend(['missing'])
    def test_invalid_method_rejected(self):
        with self.assertRaises(ValueError):self.m.recommend(method='magic')
    def test_invalid_n_rejected(self):
        for n in (0,101,-1,1.5):
            with self.assertRaises(ValueError):self.m.recommend(n=n)
    def test_duplicate_pairs_rejected(self):
        with self.assertRaises(ValueError):validate_rows(self.rows+[self.rows[0]])
    def test_nan_ratings_rejected(self):
        with self.assertRaises(ValueError):validate_rows([Row('a','b',float('nan'),0)])
    def test_ids_and_negative_time_rejected(self):
        for r in [Row('<script>','x',4,0),Row('a','b',4,-1)]:
            with self.assertRaises(ValueError):validate_rows([r])
    def test_metric_duplicate_ranks_rejected(self):
        with self.assertRaises(ValueError):ranking_metrics(['A','A'],{'A'})
    def test_exports_match_actual_scores(self):
        x=self.m.recommend(['A']);rows=list(csv.DictReader(io.StringIO(csv_export(x))))
        self.assertEqual([r['item'] for r in rows],[r['item'] for r in x['ranking']])
        for a,b in zip(rows,x['ranking']):self.assertEqual(float(a['score']),b['score'])
    def test_repeat_exact(self):
        self.assertEqual(self.m.recommend(['A']),Recommender(self.rows).recommend(['A']))
    def test_authored_demo_has_no_seen_recommendations(self):
        x=Recommender(synthetic_rows()).recommend(['S01','S02'],['S05']);self.assertFalse(set(r['item'] for r in x['ranking']) & {'S01','S02','S05'})

class APIWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=Recommender(synthetic_rows());cls.server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(cls.model,True));cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start();cls.url=f'http://127.0.0.1:{cls.server.server_port}'
    @classmethod
    def tearDownClass(cls):cls.server.shutdown();cls.server.server_close();cls.thread.join()
    def request(self,path='/api/recommend',data=None,origin=None):
        body=json.dumps(data if data is not None else {'liked':['S01','S02'],'seen':['S05']}).encode()
        headers={'Content-Type':'application/json'}
        if origin:headers['Origin']=origin
        return urllib.request.urlopen(urllib.request.Request(self.url+path,body,headers),timeout=5)
    def test_primary_workflow_api_and_engine_parity(self):
        actual=json.load(self.request());expected=self.model.recommend(['S01','S02'],['S05'])
        self.assertEqual(actual['csv'],csv_export(expected));self.assertEqual(actual['ranking'],json.loads(json.dumps(expected))['ranking'])
    def test_api_unknown_item_error(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:self.request(data={'liked':['unknown']})
        self.assertEqual(cm.exception.code,400)
    def test_api_unknown_fields_error(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:self.request(data={'file':'private'})
        self.assertEqual(cm.exception.code,400)
    def test_cross_origin_error(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:self.request(origin='https://example.com')
        self.assertEqual(cm.exception.code,403)
    def test_restricted_files_not_served(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:urllib.request.urlopen(self.url+'/PROTOCOL.md')
        self.assertEqual(cm.exception.code,404)
    def test_catalog_synthetic_flag_and_security_headers(self):
        with urllib.request.urlopen(self.url+'/api/catalog') as response:
            self.assertTrue(json.load(response)['synthetic']);self.assertEqual(response.headers['X-Content-Type-Options'],'nosniff')
    def test_unexpected_host_rejected(self):
        request=urllib.request.Request(self.url+'/api/catalog',headers={'Host':'example.com'})
        with self.assertRaises(urllib.error.HTTPError) as cm:urllib.request.urlopen(request)
        self.assertEqual(cm.exception.code,403)
    def test_oversize_body_rejected(self):
        request=urllib.request.Request(self.url+'/api/recommend',data=b'x'*65537)
        with self.assertRaises(urllib.error.HTTPError) as cm:urllib.request.urlopen(request)
        self.assertEqual(cm.exception.code,413)

if __name__=='__main__':unittest.main(verbosity=2)
