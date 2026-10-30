import csv
import io
import math
import unittest
import numpy as np
from bpr import BPRRecommender
from recommender import synthetic_rows
from catalog_model import CatalogModel,catalog_csv,synthetic_catalog


class CatalogChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.cf=BPRRecommender(synthetic_rows(),seed=13,epochs=2)
    def model(self):return CatalogModel(self.cf,synthetic_catalog())
    def test_idf_scalar_and_add_frozen(self):
        m=self.model();expected=math.log(13/5)+1
        np.testing.assert_allclose(m.transformer.idf_,expected,rtol=0,atol=1e-14)
        before=m.transformer.idf_.copy();m.add_item('NEW',['space','never_seen'])
        np.testing.assert_array_equal(before,m.transformer.idf_)
        self.assertAlmostEqual(m.vectors[-1,m.tag_index['space']],1.)
    def test_cold_item_content_and_export(self):
        m=self.model();m.add_item('NEW',['space','adventure'])
        r=m.recommend(liked=['S01'],method='content',n=20)
        row=next(x for x in r['ranking'] if x['item']=='NEW')
        self.assertTrue(row['cold_item']);self.assertIsNone(row['collaborative_score'])
        self.assertAlmostEqual(row['content_score'],1.)
        exported=list(csv.DictReader(io.StringIO(catalog_csv(r))))
        self.assertEqual([x['item'] for x in exported],[x['item'] for x in r['ranking']])
        self.assertEqual([float(x['score']) for x in exported],[x['score'] for x in r['ranking']])
    def test_hybrid_scalar(self):
        m=self.model();m.add_item('NEW',['space'])
        r=m.recommend(liked=['S01'],method='hybrid',n=20)
        for row in r['ranking']:
            expected=sum(1/(60+row[k]) for k in ['content_rank','collaborative_rank'] if row[k] is not None)
            self.assertAlmostEqual(row['score'],expected,places=15)
    def test_fallbacks_and_exclusion(self):
        m=self.model();r=m.recommend()
        self.assertEqual(r['effective_method'],'popularity')
        self.assertTrue(r['fallback_reason'])
        r=m.recommend(liked=['S01'],seen=['S02'],method='content')
        self.assertFalse({'S01','S02'} & {x['item'] for x in r['ranking']})
        self.assertEqual(m.recommend(seen=m.items)['ranking'],[])
        self.assertEqual(r,m.recommend(liked=['S01','S01'],seen=['S02'],method='content'))
    def test_unknown_and_invalid(self):
        m=self.model()
        for args in [{'liked':['BAD']},{'n':True},{'n':0},{'method':'bad'},{'liked':'S01'},{'user_id':'D00','liked':['S01']}]:
            with self.assertRaises(ValueError):m.recommend(**args)
        with self.assertRaises(ValueError):m.add_item('S01',['space'])
        with self.assertRaises(ValueError):m.add_item('NEW',['bad tag'])
    def test_cold_query_and_zero_metadata(self):
        m=self.model();m.add_item('NEW',['space']);m.add_item('ZERO',['unseen'])
        self.assertEqual(m.recommend(liked=['NEW'])['effective_method'],'content')
        self.assertEqual(m.recommend(liked=['ZERO'])['effective_method'],'popularity')
    def test_user_profile_and_scalar_cosine(self):
        m=self.model();r=m.recommend(user_id='D00',method='content',n=20)
        self.assertEqual(r['collaborative_source'],'fitted_user_vector')
        ids=sorted(self.cf.likes['D00']);v=m.vectors[[m.index[i] for i in ids]].mean(axis=0)
        norm=math.sqrt(sum(float(x*x) for x in v))
        for row in r['ranking']:
            want=sum(float(a*b) for a,b in zip(m.vectors[m.index[row['item']]],v))/norm
            self.assertAlmostEqual(row['content_score'],want,places=14)


if __name__=='__main__':unittest.main()
