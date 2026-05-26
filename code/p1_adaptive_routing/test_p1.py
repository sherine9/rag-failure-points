"""
Lightweight tests for P1 (no GPU / no model download).

Run: python test_p1.py
"""

import unittest

from failure_labels import (
    gold_in_chunks,
    label_failure,
    enrich_result,
    apply_cross_mode_labels,
    per_query_scores,
)


class TestFailureLabels(unittest.TestCase):

    def test_gold_in_chunks(self):
        self.assertTrue(gold_in_chunks(["built in 1889"], ["1889"]))
        self.assertFalse(gold_in_chunks(["unrelated"], ["1889"]))

    def test_label_retrieval_miss(self):
        r = {
            "mode": "always_retrieve",
            "retrieval_decision": True,
            "chunks": ["no answer here"],
            "prediction": "wrong",
            "gold_answers": ["1889"],
            "exact_match": 0,
            "retrieval_hit": False,
        }
        self.assertEqual(label_failure(r), "retrieval_miss")

    def test_label_none_when_correct(self):
        r = {
            "mode": "selective",
            "retrieval_decision": True,
            "chunks": ["1889"],
            "prediction": "1889",
            "gold_answers": ["1889"],
            "exact_match": 1,
        }
        self.assertEqual(label_failure(r), "none")

    def test_cross_mode_wrong_skip(self):
        selective = {
            "mode": "selective",
            "retrieval_decision": False,
            "chunks": [],
            "prediction": "wrong",
            "gold_answers": ["1889"],
            "exact_match": 0,
        }
        always = {
            "mode": "always_retrieve",
            "retrieval_decision": True,
            "chunks": ["built in 1889"],
            "prediction": "1889",
            "gold_answers": ["1889"],
            "exact_match": 1,
            "retrieval_hit": True,
        }
        no_ret = {
            "mode": "no_retrieval",
            "retrieval_decision": False,
            "chunks": [],
            "prediction": "wrong",
            "gold_answers": ["1889"],
            "exact_match": 0,
        }
        label = label_failure(
            selective,
            refs={"always_retrieve": always, "no_retrieval": no_ret},
        )
        self.assertEqual(label, "wrong_skip_decision")

    def test_apply_cross_mode_labels(self):
        results = {
            "no_retrieval": [
                {
                    "question": "q",
                    "gold_answers": ["a"],
                    "prediction": "a",
                    "chunks": [],
                    "retrieval_decision": False,
                    "mode": "no_retrieval",
                }
            ],
            "always_retrieve": [
                {
                    "question": "q",
                    "gold_answers": ["a"],
                    "prediction": "b",
                    "chunks": ["x"],
                    "retrieval_decision": True,
                    "mode": "always_retrieve",
                }
            ],
            "selective": [
                {
                    "question": "q",
                    "gold_answers": ["a"],
                    "prediction": "b",
                    "chunks": ["x"],
                    "retrieval_decision": True,
                    "mode": "selective",
                }
            ],
        }
        apply_cross_mode_labels(results)
        self.assertEqual(results["no_retrieval"][0]["failure_label"], "none")
        self.assertIn(
            results["selective"][0]["failure_label"],
            ("ignored_evidence", "retrieval_miss", "unnecessary_retrieval"),
        )

    def test_per_query_scores(self):
        s = per_query_scores("1889", ["1889", "in 1889"])
        self.assertEqual(s["exact_match"], 1)
        self.assertGreater(s["rouge_l"], 0.9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
