import csv
import io
import math
import unittest
import numpy as np
from benchmark_candidates import candidate_pool, rank_pool, metric, csv_audit


class CandidateChecks(unittest.TestCase):
    def test_retains_all_relevance_and_exact_budget(self):
        pool = candidate_pool(range(100), {4, 8}, 20, 101, 13)
        self.assertEqual(len(pool), 22)
        self.assertTrue({4, 8} <= set(pool) <= set(range(100)))

    def test_nested_reproducible_order_independent(self):
        small = candidate_pool(range(100), {4}, 20, 101, 13)
        large = candidate_pool(reversed(range(100)), {4}, 50, 101, 13)
        self.assertTrue(set(small) <= set(large))
        self.assertEqual(small, candidate_pool(reversed(range(100)), {4}, 20, 101, 13))

    def test_caps_small_pools_and_zero_budget(self):
        self.assertEqual(candidate_pool({'a', 'b'}, {'a'}, 100, 101, 13), ['a', 'b'])
        self.assertEqual(candidate_pool({'a', 'b'}, {'a'}, 0, 101, 13), ['a'])
        self.assertEqual(candidate_pool({'a'}, {'a'}, 20, 101, 13), ['a'])

    def test_invalid_budget_and_relevance(self):
        for budget in (-1, True, 2.5):
            with self.assertRaises(ValueError): candidate_pool({'a'}, {'a'}, budget, 1, 1)
        for rel in (set(), {'b'}):
            with self.assertRaises(ValueError): candidate_pool({'a'}, rel, 20, 1, 1)

    def test_full_candidate_pool_preserved(self):
        self.assertEqual(candidate_pool({'a', 'b', 'c'}, {'b'}, None, None, 1), ['a', 'b', 'c'])

    def test_score_popularity_id_ties(self):
        index = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
        self.assertEqual(rank_pool(list(reversed(index)), np.array([1., 1., 1., 2.]), np.array([4., 5., 5., 0.]), index), ['d', 'b', 'c', 'a'])
        with self.assertRaises(ValueError): rank_pool(['a', 'a'], np.ones(4), np.ones(4), index)
        with self.assertRaises(ValueError): rank_pool(['z'], np.ones(4), np.ones(4), index)

    def test_denominators_and_exact_math(self):
        got = metric(['a', 'x', 'b'], {'a', 'b', 'c'})
        self.assertEqual(got['recall'], 2/3)
        self.assertAlmostEqual(got['ndcg'], (1+1/math.log2(4))/(1+1/math.log2(3)+1/math.log2(4)))
        self.assertEqual(metric(['a'], {'a'})['ndcg'], 1)
        self.assertEqual(metric(['x'], {'a'})['recall'], 0)

    def test_sampling_changes_task_not_model(self):
        # Removing distractors can improve a fixed model's ranking metric.
        self.assertEqual(metric(['x', 'a'], {'a'})['recall'], 1)
        self.assertLess(metric(['x', 'a'], {'a'})['ndcg'], metric(['a'], {'a'})['ndcg'])

    def test_lossless_private_export(self):
        row = dict(model_seed=13, sampling_seed=None, budget=None, method='bpr', user='authored', warm=True,
                   fallback=False, candidate_count=2, candidate_sha256='authoredhash', relevant_count=1,
                   ranking=['a'], scores=[float.fromhex('0x1.123456789abcdp-3')], ndcg=1/3, recall=1/7)
        rows = list(csv.DictReader(io.StringIO(csv_audit([row]))))
        self.assertEqual(float(rows[0]['ndcg']), row['ndcg'])


if __name__ == '__main__':
    unittest.main()
