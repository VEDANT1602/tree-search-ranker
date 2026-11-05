"""Grouped tree ranking with family-disjoint evaluation and source receipts."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import sklearn
import xgboost as xgb

from .fixture import load
from .retrieval import FEATURE_NAMES, SearchFramework, ScopedIndex, ndcg, permitted, rank, reciprocal_rank, tokens


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_contract(queries):
    families = {split: sorted({q["family"] for q in queries if q["split"] == split})
                for split in ["train", "validation", "test"]}
    for first in families:
        for second in families:
            if first != second and set(families[first]) & set(families[second]):
                raise ValueError("Query family occurs in multiple partitions")
    if any(not group for group in families.values()):
        raise ValueError("Every partition needs query families")
    return families


def prepare(framework, queries, judgments):
    prepared = []
    for query in sorted(queries, key=lambda q: q["query_id"]):
        ids, features, rrf = framework.candidates(query)
        index = framework.scoped_index(query)
        eligible = [doc["doc_id"] for doc in index.documents] if index else []
        grades = judgments.get(query["query_id"], {})
        prepared.append({"query": query, "ids": ids, "features": features,
                         "rrf": np.asarray(rrf), "grades": grades, "eligible": eligible})
    return prepared


def training_matrix(prepared):
    matrices, labels, qids = [], [], []
    for group, item in enumerate(prepared):
        if not item["ids"]:
            raise ValueError("Training query has no candidates")
        matrices.append(item["features"])
        labels.extend([item["grades"].get(doc, 0) for doc in item["ids"]])
        qids.extend([group] * len(item["ids"]))
    return np.concatenate(matrices), np.asarray(labels, dtype=np.int32), np.asarray(qids, dtype=np.int32)


def score_method(item, method, model=None):
    features = item["features"]
    if method == "bm25":
        return features[:, 0]
    if method == "rrf":
        return item["rrf"]
    if method == "metadata_heuristic":
        lexical = features[:, 0] / max(float(features[:, 0].max()), 1.)
        return .5 * lexical + .25 * features[:, 2] + .2 * features[:, 6] + .05 * np.exp(-features[:, 7] / 5)
    if method == "tree":
        return model.predict(features)
    raise ValueError(method)


def evaluate(prepared, method, model=None):
    rows = []
    for item in prepared:
        query = item["query"]
        ordered = rank(item["ids"], score_method(item, method, model)) if item["ids"] else []
        relevant = {doc for doc in item["eligible"] if item["grades"].get(doc, 0) >= 2}
        row = {"query_id": query["query_id"], "family": query["family"], "tenant": query["tenant"],
               "principal": query["principal"], "intent": "historical" if set(tokens(query["text"])) & {"prior", "previous", "2024", "archived"} else "current",
               "ndcg_at_5": ndcg(ordered, item["grades"], item["eligible"]),
               "mrr_at_10": reciprocal_rank(ordered, item["grades"]),
               "candidate_count": len(item["ids"]), "eligible_documents": len(item["eligible"]),
               "eligible_relevant_documents": len(relevant),
               "candidate_recall": len(relevant & set(item["ids"])) / len(relevant) if relevant else 0.,
               "top_doc": ordered[0] if ordered else None,
               "top_grade": item["grades"].get(ordered[0], 0) if ordered else 0,
               "top_10_doc_ids": json.dumps(ordered[:10]),
               "top_10_grades": json.dumps([item["grades"].get(doc, 0) for doc in ordered[:10]])}
        rows.append(row)
    summary = {"queries": len(rows), "ndcg_at_5": float(np.mean([r["ndcg_at_5"] for r in rows])),
               "mrr_at_10": float(np.mean([r["mrr_at_10"] for r in rows])),
               "candidate_recall": float(np.mean([r["candidate_recall"] for r in rows])),
               "mean_candidates": float(np.mean([r["candidate_count"] for r in rows]))}
    summary["intent_ndcg_at_5"] = {intent: float(np.mean([r["ndcg_at_5"] for r in rows if r["intent"] == intent]))
                                    for intent in ["current", "historical"]}
    return summary, rows


def clustered_interval(tree_rows, baseline_rows, seed=17, draws=2000):
    if [r["query_id"] for r in tree_rows] != [r["query_id"] for r in baseline_rows]:
        raise ValueError("Paired uncertainty needs the same ordered query population")
    family_differences = defaultdict(list)
    for tree, baseline in zip(tree_rows, baseline_rows):
        family_differences[tree["family"]].append(tree["ndcg_at_5"] - baseline["ndcg_at_5"])
    family_means = {family: float(np.mean(values)) for family, values in family_differences.items()}
    values = np.array(list(family_means.values()))
    rng = np.random.default_rng(seed)
    bootstraps = [float(np.mean(rng.choice(values, len(values), replace=True))) for _ in range(draws)]
    return {"mean_difference": float(values.mean()), "lower_95": float(np.quantile(bootstraps, .025)),
            "upper_95": float(np.quantile(bootstraps, .975)), "unit": "query family",
            "families": len(values), "draws": draws, "seed": seed, "family_means": family_means}


def csv_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_ranker(depth, estimators):
    return xgb.XGBRanker(objective="rank:ndcg", tree_method="hist", max_depth=depth,
                         n_estimators=estimators, learning_rate=.08, n_jobs=2, random_state=17,
                         subsample=1., colsample_bytree=1., reg_lambda=1.,
                         lambdarank_pair_method="topk", lambdarank_num_pair_per_sample=8)


def benchmark(root: Path):
    started = time.perf_counter()
    documents, queries, judgments = load(root / "data")
    partitions = split_contract(queries)
    framework = SearchFramework(documents)
    prepared = {split: prepare(framework, [q for q in queries if q["split"] == split], judgments)
                for split in partitions}
    matrix, labels, qids = training_matrix(prepared["train"])
    experiments, models = [], []
    for depth, estimators in [(2, 40), (2, 80), (3, 40), (3, 80)]:
        model = make_ranker(depth, estimators)
        model.fit(matrix, labels, qid=qids)
        validation, _ = evaluate(prepared["validation"], "tree", model)
        experiments.append({"max_depth": depth, "n_estimators": estimators, "validation": validation})
        models.append(model)
    selected = max(range(len(experiments)), key=lambda i: (experiments[i]["validation"]["ndcg_at_5"],
                                                          -experiments[i]["max_depth"], -experiments[i]["n_estimators"]))
    model = models[selected]
    baseline_validation = {method: evaluate(prepared["validation"], method)[0]
                           for method in ["bm25", "rrf", "metadata_heuristic"]}
    best_baseline = max(baseline_validation, key=lambda method: baseline_validation[method]["ndcg_at_5"])
    test, raw = {}, {}
    for method in ["bm25", "rrf", "metadata_heuristic", "tree"]:
        test[method], raw[method] = evaluate(prepared["test"], method, model)
    out = root / "reports"
    out.mkdir(exist_ok=True)
    model_path = out / "ranker.json"
    model.save_model(model_path)
    # Demonstrate the dangerous alternative, never use it for training or serving.
    unsafe = ScopedIndex(documents)
    failure_demos = []
    for query in [q for q in queries if q["split"] == "test" and q["tenant"] == "birch"][:4]:
        scores = unsafe.bm25(query["text"])
        ids = [d["doc_id"] for d in unsafe.documents]
        unsafe_top = rank(ids, scores)[:5]
        by_id = {d["doc_id"]: d for d in documents}
        blocked = [doc_id for doc_id in unsafe_top if not permitted(by_id[doc_id], query)]
        safe_ids, _, _ = framework.candidates(query)
        failure_demos.append({"query_id": query["query_id"], "unsafe_global_top_5": unsafe_top,
                              "forbidden_hits": blocked, "safe_candidate_count": len(safe_ids),
                              "safe_forbidden_hits": sum(not permitted(by_id[doc_id], query) for doc_id in safe_ids)})
    paired = clustered_interval(raw["tree"], raw[best_baseline])
    source_paths = sorted((root / "src").rglob("*.py")) + [root / "run.py", root / "requirements.txt"]
    source_hashes = {str(path.relative_to(root)).replace("\\", "/"): digest(path) for path in source_paths}
    data_hashes = {path.name: digest(path) for path in sorted((root / "data").glob("*")) if path.is_file()}
    result = {"schema_version": 1, "run_kind": "executed_synthetic_ranking_experiment",
              "data": {"documents": len(documents), "queries": len(queries), "judgment_origin": "original deterministic fixture rules, not human or LLM judgments",
                       "families": partitions, "sha256": data_hashes},
              "features": FEATURE_NAMES, "training": {"rows": len(labels), "query_groups": int(qids.max() + 1),
                  "feature_matrix_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
                  "labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
                  "qid_sha256": hashlib.sha256(qids.tobytes()).hexdigest(),
                  "seed": 17, "candidate_experiments": experiments, "selection": experiments[selected],
                  "selection_rule": "validation full-eligible-corpus NDCG@5; smaller depth/tree count on exact tie"},
              "baselines_validation": baseline_validation, "best_baseline_by_validation": best_baseline,
              "test": test, "paired_tree_minus_baseline_ndcg": paired,
              "access": {"prefilter": "tenant AND ACL intersection AND availability/update <= as_of",
                         "forbidden_candidates_across_all_queries": sum(not permitted(next(d for d in documents if d["doc_id"] == doc_id), item["query"])
                             for group in prepared.values() for item in group for doc_id in item["ids"])},
              "model_sha256": digest(model_path), "source_sha256": source_hashes,
              "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "sklearn": sklearn.__version__,
                              "xgboost": xgb.__version__, "platform": platform.platform(), "threads": 2},
              "runtime_seconds": time.perf_counter() - started}
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    (out / "failure_demos.json").write_text(json.dumps(failure_demos, indent=2) + "\n", encoding="utf-8", newline="\n")
    rows = []
    for method in raw:
        rows.extend([{**r, "method": method} for r in raw[method]])
    csv_rows(out / "per_query_test.csv", rows)
    (out / "feature_importance.json").write_text(json.dumps(dict(zip(FEATURE_NAMES, map(float, model.feature_importances_))), indent=2) + "\n", encoding="utf-8", newline="\n")
    from .report import render
    render(root)
    print(json.dumps({"selection": experiments[selected], "best_baseline": best_baseline, "test": test,
                      "paired_difference": paired, "runtime_seconds": result["runtime_seconds"]}, indent=2))
    return result
