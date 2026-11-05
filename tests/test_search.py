from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tree_search.fixture import generate, load
from tree_search.retrieval import FEATURE_NAMES, SearchFramework, ndcg, permitted, rank, reciprocal_rank
from tree_search.experiment import clustered_interval, make_ranker, prepare, split_contract, training_matrix


class SearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.documents, cls.queries, cls.judgments = load(ROOT / "data")
        cls.framework = SearchFramework(cls.documents)

    def query(self, tenant="aster", principal="employee"):
        return {"text": "current record retention approved procedure", "tenant": tenant,
                "groups": ["all", "leadership"] if principal == "manager" else ["all"], "as_of_day": 1000}

    def test_fixture_reproduces_exactly_without_external_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / "first", Path(tmp) / "second"
            self.assertEqual(generate(first), generate(second))
            for name in ["documents.json", "queries.json", "qrels.csv"]:
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

    def test_query_families_are_disjoint_including_roles_tenants_paraphrases(self):
        partitions = split_contract(self.queries)
        self.assertEqual([len(partitions[name]) for name in ["train", "validation", "test"]], [8, 3, 3])
        self.assertFalse(set(partitions["train"]) & set(partitions["test"]))
        altered = deepcopy(self.queries)
        altered[0]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "multiple"):
            split_contract(altered)

    def test_tenant_and_acl_are_hard_constraints_before_retrieval(self):
        query = self.query(tenant="birch")
        ids, _, _ = self.framework.candidates(query)
        self.assertTrue(ids)
        lookup = {d["doc_id"]: d for d in self.documents}
        self.assertTrue(all(permitted(lookup[doc_id], query) for doc_id in ids))
        self.assertTrue(all(doc.startswith("birch-") and not doc.endswith("-manager") for doc in ids))
        index = self.framework.scoped_index(query)
        self.assertTrue(all(doc["tenant"] == "birch" and "all" in doc["acl"] for doc in index.documents))

    def test_unknown_principal_or_tenant_fails_closed(self):
        query = self.query()
        query["groups"] = []
        self.assertEqual(self.framework.candidates(query)[0], [])
        query["groups"] = ["all"]
        query["tenant"] = "not-a-tenant"
        self.assertEqual(self.framework.candidates(query)[0], [])

    def test_future_content_and_future_updates_are_excluded(self):
        query = self.query()
        future = deepcopy(self.documents[0])
        future["available_day"] = 1001
        self.assertFalse(permitted(future, query))
        future["available_day"] = 900
        future["updated_day"] = 1001
        self.assertFalse(permitted(future, query))
        self.assertFalse(any(doc.endswith("-future") for doc in self.framework.candidates(query)[0]))

    def test_unknown_nonfinite_or_malformed_days_fail_closed(self):
        document, query = deepcopy(self.documents[0]), self.query()
        self.assertTrue(permitted(document, query))
        for bad in [float("nan"), float("inf"), -float("inf"), -1, True, "970", None, 10 ** 400]:
            for field in ["available_day", "updated_day"]:
                with self.subTest(field=field, value=str(bad)):
                    self.assertFalse(permitted({**document, field: bad}, query))
            self.assertFalse(permitted(document, {**query, "as_of_day": bad}))

    def test_malformed_acl_or_group_context_cannot_grant_access(self):
        document, query = deepcopy(self.documents[0]), self.query()
        self.assertFalse(permitted({**document, "acl": "all"}, {**query, "groups": "all"}))
        self.assertFalse(permitted(document, {**query, "groups": None}))
        self.assertFalse(permitted({**document, "tenant": None}, query))

    def test_manager_context_can_retrieve_restricted_document_employee_cannot(self):
        manager = self.query(principal="manager")
        manager["text"] = "record retention leadership manager exception"
        manager_ids = self.framework.candidates(manager)[0]
        employee = {**manager, "groups": ["all"]}
        self.assertIn("aster-retention-manager", manager_ids)
        self.assertNotIn("aster-retention-manager", self.framework.candidates(employee)[0])

    def test_cache_does_not_reuse_manager_view_for_employee(self):
        framework = SearchFramework(deepcopy(self.documents))
        manager = self.query(principal="manager")
        framework.scoped_index(manager)
        employee = self.query()
        view = framework.scoped_index(employee)
        self.assertFalse(any(d["doc_id"].endswith("-manager") for d in view.documents))

    def test_revocation_invalidates_cached_view(self):
        documents = deepcopy(self.documents)
        framework = SearchFramework(documents)
        query = self.query()
        first = framework.scoped_index(query)
        target = next(d for d in documents if d["doc_id"] == "aster-retention-current")
        self.assertIn(target["doc_id"], [d["doc_id"] for d in first.documents])
        target["acl"] = ["leadership"]
        second = framework.scoped_index(query)
        self.assertIsNot(first, second)
        self.assertNotIn(target["doc_id"], [d["doc_id"] for d in second.documents])

    def test_content_update_invalidates_cached_features(self):
        documents = deepcopy(self.documents)
        framework = SearchFramework(documents)
        query = self.query()
        first = framework.scoped_index(query)
        documents[0]["body"] += " A content update with a new procedure."
        second = framework.scoped_index(query)
        self.assertIsNot(first, second)

    def test_hidden_annotations_never_change_features(self):
        query = self.query()
        original_ids, original_features, _ = self.framework.candidates(query)
        documents = deepcopy(self.documents)
        for doc in documents:
            doc.update({"family": "new-oracle-family", "grade": 3, "expected_answer": "oracle", "target": 1})
        ids, features, _ = SearchFramework(documents).candidates({**query, "query_id": "oracle", "grade": 3, "family": "oracle"})
        self.assertEqual(original_ids, ids)
        np.testing.assert_allclose(original_features, features, rtol=0, atol=0)
        self.assertEqual(features.shape[1], len(FEATURE_NAMES))

    def test_unmatched_tokens_return_no_candidates(self):
        self.assertEqual(self.framework.candidates({**self.query(), "text": "zqxwvp mnrbft"})[0], [])

    def test_ndcg_ideal_includes_relevant_documents_outside_candidate_set(self):
        grades = {"retrieved": 2, "missed": 3}
        metric = ndcg(["retrieved"], grades, ["retrieved", "missed"])
        self.assertAlmostEqual(metric, 3 / (7 + 3 / math.log2(3)))
        self.assertLess(metric, 1)

    def test_metric_rejects_duplicate_or_forbidden_results(self):
        with self.assertRaises(ValueError):
            ndcg(["a", "a"], {"a": 3}, ["a"])
        with self.assertRaises(ValueError):
            ndcg(["forbidden"], {"forbidden": 3}, ["allowed"])

    def test_mrr_threshold_does_not_treat_weak_grade_as_answer(self):
        self.assertEqual(reciprocal_rank(["weak", "good"], {"weak": 1, "good": 2}), .5)
        self.assertEqual(reciprocal_rank(["weak"], {"weak": 1}), 0)

    def test_rank_ties_are_stable_and_invalid_scores_fail(self):
        self.assertEqual(rank(["b", "a"], [1, 1]), ["a", "b"])
        with self.assertRaises(ValueError):
            rank(["a"], [float("nan")])

    def test_grouped_matrix_has_contiguous_nondecreasing_query_ids(self):
        queries = [q for q in self.queries if q["split"] == "train"][:3]
        items = prepare(self.framework, queries, self.judgments)
        matrix, labels, qids = training_matrix(items)
        self.assertEqual(len(matrix), len(labels))
        self.assertTrue(np.all(np.diff(qids) >= 0))
        self.assertEqual(set(qids), {0, 1, 2})

    def test_test_labels_cannot_change_training_matrix(self):
        queries = [q for q in self.queries if q["split"] == "train"][:4]
        first = training_matrix(prepare(self.framework, queries, self.judgments))
        altered = deepcopy(self.judgments)
        for query in [q for q in self.queries if q["split"] == "test"]:
            altered[query["query_id"]] = {"irrelevant-to-training": 3}
        second = training_matrix(prepare(self.framework, queries, altered))
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)

    def test_saved_tree_reloads_with_identical_scores(self):
        queries = [q for q in self.queries if q["split"] == "train"][:8]
        matrix, labels, qids = training_matrix(prepare(self.framework, queries, self.judgments))
        model = make_ranker(2, 5)
        model.fit(matrix, labels, qid=qids)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            model.save_model(path)
            loaded = xgb.XGBRanker()
            loaded.load_model(path)
            np.testing.assert_array_equal(model.predict(matrix), loaded.predict(matrix))

    def test_bootstrap_pairs_queries_and_clusters_families(self):
        first = [{"query_id": "1", "family": "a", "ndcg_at_5": .5},
                 {"query_id": "2", "family": "a", "ndcg_at_5": .7},
                 {"query_id": "3", "family": "b", "ndcg_at_5": .3}]
        second = [{**r, "ndcg_at_5": .1} for r in first]
        interval = clustered_interval(first, second)
        self.assertEqual(interval["families"], 2)
        self.assertAlmostEqual(interval["mean_difference"], .35)
        with self.assertRaises(ValueError):
            clustered_interval(first, list(reversed(second)))


if __name__ == "__main__":
    unittest.main()
