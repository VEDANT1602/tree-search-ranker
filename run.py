"""Reproduce a permission-scoped search ranking experiment."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from tree_search.fixture import generate, load
from tree_search.retrieval import SearchFramework, rank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("generate", help="Regenerate original fictional data exactly")
    sub.add_parser("benchmark", help="Train, choose on validation and evaluate held-out query families")
    sub.add_parser("report", help="Regenerate prose/chart from saved numerical evidence")
    search = sub.add_parser("search", help="Rank permitted fictional documents; no answer generation")
    search.add_argument("text")
    search.add_argument("--tenant", choices=["aster", "birch", "cedar"], default="aster")
    search.add_argument("--principal", choices=["employee", "manager"], default="employee")
    search.add_argument("--as-of", type=int, default=1000)
    search.add_argument("--method", choices=["tree", "bm25", "rrf", "metadata_heuristic"], default="tree")
    args = parser.parse_args()
    if args.command == "generate":
        print(json.dumps(generate(ROOT / "data"), indent=2))
    elif args.command == "benchmark":
        from tree_search.experiment import benchmark
        benchmark(ROOT)
    elif args.command == "report":
        from tree_search.report import render
        render(ROOT)
    else:
        from tree_search.experiment import score_method
        import xgboost as xgb
        documents, _, _ = load(ROOT / "data")
        query = {"text": args.text, "tenant": args.tenant, "as_of_day": args.as_of,
                 "groups": ["all", "leadership"] if args.principal == "manager" else ["all"]}
        framework = SearchFramework(documents)
        ids, features, rrf = framework.candidates(query)
        model = None
        if ids and args.method == "tree":
            if not (ROOT / "reports" / "ranker.json").exists():
                parser.error("Run benchmark first to create the tree artifact")
            model = xgb.XGBRanker()
            model.load_model(ROOT / "reports" / "ranker.json")
        ordered = rank(ids, score_method({"features": features, "rrf": rrf}, args.method, model)) if ids else []
        lookup = {doc["doc_id"]: doc for doc in documents}
        print(json.dumps({"answer_generated": False, "status": "ranked_documents" if ids else "no_lexical_match",
                          "method": args.method, "tenant": args.tenant, "principal": args.principal,
                          "query": args.text, "results": [{"doc_id": doc_id, "title": lookup[doc_id]["title"],
                              "source": lookup[doc_id]["source"], "status": lookup[doc_id]["status"]}
                              for doc_id in ordered[:5]]}, indent=2))


if __name__ == "__main__":
    main()
