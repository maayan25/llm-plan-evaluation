# Test the common evaluation functions in the scripts/common.py file

import unittest
from scripts.plan_recovery.common import calculate_semantic_similarity

class TestCommon(unittest.TestCase):
    def test_calculate_semantic_similarity(self):
        same = calculate_semantic_similarity("stack", "stack")
        similar = calculate_semantic_similarity("stack", "load")
        less_similar = calculate_semantic_similarity("stack", "put")
        opposite_sim = calculate_semantic_similarity("unstack", "stack")
        different = calculate_semantic_similarity("unstack", "put-down")
        unrelated = calculate_semantic_similarity("put-down", "sing")
        opposite = calculate_semantic_similarity("put-down", "load")

        self.assertEqual(same, 1)
        self.assertAlmostEqual(similar, 0.85, 1)
        self.assertAlmostEqual(less_similar, 0.25, 1)
        self.assertEqual(opposite_sim, 0)
        self.assertEqual(different, 0)
        self.assertEqual(unrelated, 0)
        self.assertEqual(opposite, 0)