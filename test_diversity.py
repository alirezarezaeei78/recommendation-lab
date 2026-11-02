import argparse
import contextlib
import csv
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
from bpr import BPRRecommender
from recommender import synthetic_rows,Row
from catalog_model import synthetic_catalog
from diversity import greedy_mmr,tag_similarity,intra_list_diversity,DiversityRecommender,diversity_csv
from diversity_cli import run


class MMRChecks(unittest.TestCase):
    def test_hand_calculated_redundancy_and_endpoints(self):
        items=['A','B','C'];scores=[1,.9,.6];sim=tag_similarity(items,{'A':['x'],'B':['x'],'C':['y']})
        for value,expected in [(1,['A','B','C']),(.5,['A','C','B']),(0,['A','C','B'])]:
            out=greedy_mmr(items,scores,[0]*3,sim,value,3)
            self.assertEqual([r['item'] for r in out],expected)
        out=greedy_mmr(items,scores,[0]*3,sim,.5,3)
        self.assertAlmostEqual(out[1]['mmr_utility'],0)
        self.assertEqual(out[1]['base_rank'],3)
        self.assertAlmostEqual(out[2]['mmr_utility'],-.125)

    def test_constant_scores_ties_input_permutation(self):
        ids=['C','B','A'];sim=np.ones((3,3))
        expected=['B','A','C']
        for value in [0,.5,1]:
            out=greedy_mmr(ids,[3]*3,[0,1,0],sim,value,100)
            self.assertEqual([r['item'] for r in out],expected)
            self.assertTrue(all(r['normalized_relevance']==0 for r in out))
        self.assertEqual(greedy_mmr([],[],[],np.empty((0,0)),.5),[])

    def test_validation(self):
        for value in [-1,1.01,float('nan'),float('inf'),True,'x']:
            with self.subTest(value=value),self.assertRaises(ValueError):greedy_mmr(['A'],[0],[0],[[1]],value)
        for n in [0,101,True,1.5]:
            with self.assertRaises(ValueError):greedy_mmr(['A'],[0],[0],[[1]],1,n)
        for ids,scores,pop,sim in [(['A','A'],[1,2],[1,2],np.eye(2)),(['A'],[float('nan')],[0],[[1]]),(['A'],[0],[0],[[2]]),(['A'],[0],[-1],[[1]])]:
            with self.assertRaises(ValueError):greedy_mmr(ids,scores,pop,sim)
        for tags in [{},{'A':[]},{'A':['']},{'A':'x'}]:
            with self.assertRaises(ValueError):tag_similarity(['A'],tags)

    def test_scalar_cosine_and_ild(self):
        tags={'A':['x','y'],'B':['y'],'C':['z']};s=tag_similarity(list(tags),tags)
        self.assertAlmostEqual(s[0,1],1/math.sqrt(2))
        self.assertAlmostEqual(intra_list_diversity(list(tags),tags),(3-1/math.sqrt(2))/3)
        self.assertIsNone(intra_list_diversity(['A'],tags))


class WorkflowChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=BPRRecommender(synthetic_rows(),seed=13,epochs=2)
        cls.engine=DiversityRecommender(cls.model,synthetic_catalog())

    def test_profile_base_parity_seen_exclusion_and_candidate_count(self):
        for user in self.model.users:
            out=self.engine.recommend(user,1)
            base=self.model.recommend(method='bpr',user_id=user)
            self.assertEqual([(r['item'],r['score']) for r in out['ranking']],[(r['item'],r['score']) for r in base['ranking']])
            for value in [0,.5,1]:
                altered=self.engine.recommend(user,value)
                ids=[r['item'] for r in altered['ranking']]
                self.assertFalse(set(ids)&self.model.seen[user]);self.assertEqual(len(ids),len(set(ids)))
                self.assertEqual(altered['candidate_count'],base['candidate_count'])

    def test_unknown_and_no_positive_profiles_are_explicit_fallback(self):
        result=self.engine.recommend('UNKNOWN',.5)
        self.assertTrue(result['fallback']);self.assertEqual(result['effective_method'],'popularity')
        self.assertEqual(result['candidate_count'],len(self.model.items))
        with self.assertRaises(ValueError):self.engine.recommend('invalid id',.5)

    def test_csv_exact_decision_fields(self):
        result=self.engine.recommend('D00',.5)
        rows=list(csv.DictReader(io.StringIO(diversity_csv(result))))
        for got,expected in zip(rows,result['ranking']):
            self.assertEqual(got['item'],expected['item'])
            for field in ['score','normalized_relevance','max_selected_similarity','mmr_utility']:
                self.assertEqual(float(got[field]),expected[field])
        self.assertEqual(len(rows),len(result['ranking']))

    def test_exhausted_candidate_set_and_no_fitted_positive_profile(self):
        model=BPRRecommender([Row('U','A',1,1)],seed=13,epochs=1)
        engine=DiversityRecommender(model,{'A':['x']})
        got=engine.recommend('U',.5)
        self.assertEqual(got['ranking'],[]);self.assertIsNone(got['intra_list_diversity'])
        self.assertTrue(got['fallback']);self.assertEqual(got['candidate_count'],0)
        self.assertEqual(len(diversity_csv(got).splitlines()),1)
        unseen=engine.recommend('UNKNOWN',0)
        self.assertEqual([r['item'] for r in unseen['ranking']],['A'])
        self.assertTrue(unseen['fallback'])

    def test_documented_cli_workflow_export_stdout_and_malformed_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=argparse.Namespace(data=None,catalog=None,user_id='D00',lambda_value=.5,n=5,output=str(Path(tmp)/'rank'))
            stdout=io.StringIO()
            with contextlib.redirect_stdout(stdout):result=run(args)
            self.assertEqual(json.loads(stdout.getvalue()),json.loads(Path(tmp,'rank.json').read_text()))
            self.assertTrue(result['synthetic']);self.assertEqual(len(result['ranking']),5)
            self.assertEqual(Path(tmp,'rank.csv').read_text(),diversity_csv(result))
            args.catalog='missing.json'
            with self.assertRaises(ValueError):run(args)


if __name__=='__main__':unittest.main()
