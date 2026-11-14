"""Independent rational/decimal checks of the teaching arithmetic."""
import math
import unittest
from decimal import Decimal, localcontext
from fractions import Fraction

from tutorial import calculate, rank, sigmoid


class ScoreSemanticsChecks(unittest.TestCase):
    def setUp(self):
        self.result = calculate()

    def test_dot_products_and_affine_values_against_exact_rationals(self):
        user = [Fraction(1), Fraction(2)]
        item_fractions = {"A": [Fraction(2, 5), Fraction(1, 5)],
                          "B": [Fraction(-1, 10), Fraction(1, 5)],
                          "C": [Fraction(1, 5), Fraction(-1, 5)]}
        for row in self.result["rows"]:
            oracle = sum(u * v for u, v in zip(user, item_fractions[row["item"]]))
            self.assertAlmostEqual(row["score"], float(oracle), delta=2e-16)
            self.assertAlmostEqual(row["transformed_score"], float(10 * oracle + 7), delta=2e-15)

    def test_all_pairwise_signs_and_rank_preserved(self):
        rows = self.result["rows"]
        for left in rows:
            for right in rows:
                self.assertEqual(left["score"] > right["score"],
                                 left["transformed_score"] > right["transformed_score"])
        self.assertEqual(self.result["rank_original"], ["A", "B", "C"])
        self.assertEqual(self.result["rank_original"], self.result["rank_transformed"])
        self.assertEqual(self.result["rank_original"], self.result["rank_sigmoid"])

    def test_sigmoids_against_independent_decimal_exponential(self):
        with localcontext() as ctx:
            ctx.prec = 60
            for value in [-15, -5, -0.2, 0, 0.3, 0.5, 0.8, 5, 10, 15]:
                decimal_value = Decimal(str(value))
                oracle = Decimal(1) / (Decimal(1) + (-decimal_value).exp())
                self.assertAlmostEqual(sigmoid(value), float(oracle), delta=2e-16)

    def test_pairwise_shift_cancels_but_positive_scale_changes_likelihood(self):
        pair = self.result["pair_A_over_B"]
        self.assertAlmostEqual(pair["margin"], 0.5)
        self.assertAlmostEqual(pair["shift_only_margin"], pair["margin"])
        self.assertAlmostEqual(pair["transformed_margin"], 5.0)
        self.assertNotEqual(pair["sigmoid_margin"], pair["transformed_sigmoid_margin"])
        self.assertNotEqual(pair["negative_log_sigmoid"], pair["transformed_negative_log_sigmoid"])
        self.assertAlmostEqual(pair["sigmoid_margin"] + sigmoid(-pair["margin"]), 1.0)

    def test_item_sigmoid_is_not_the_pairwise_sigmoid(self):
        item_a = self.result["rows"][0]
        self.assertNotEqual(item_a["single_item_sigmoid"],
                            self.result["pair_A_over_B"]["sigmoid_margin"])

    def test_extremes_ties_and_invalid_scores(self):
        self.assertEqual(sigmoid(-1000), 0.0)
        self.assertEqual(sigmoid(1000), sigmoid(999))  # Real monotonicity is not a float guarantee.
        self.assertEqual(rank({"B": 1.0, "A": 1.0}), ["A", "B"])
        for value in [math.nan, math.inf, -math.inf]:
            with self.assertRaises(ValueError):
                sigmoid(value)
            with self.assertRaises(ValueError):
                rank({"A": value})
        with self.assertRaises(ValueError):
            rank({})


if __name__ == "__main__":
    unittest.main(verbosity=2)
