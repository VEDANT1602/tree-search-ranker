"""Render the search evaluation and chart from saved metrics."""
from __future__ import annotations
import json
from pathlib import Path


def render(root: Path):
    reports = root / "reports"
    result = json.loads((reports / "metrics.json").read_text(encoding="utf-8"))
    test = result["test"]
    comparison = result["paired_tree_minus_baseline_ndcg"]
    baseline = result["best_baseline_by_validation"]
    selection = result["training"]["selection"]
    labels = {"bm25": "BM25", "rrf": "BM25 + SVD fusion", "metadata_heuristic": "Metadata heuristic", "tree": "XGBRanker"}
    decision = ("Keep the tree as the next evaluation candidate, then collect independent judgments before any deployment."
                if comparison["lower_95"] > 0 else
                "Keep the simpler baseline until independent judgments justify the tree's extra complexity.")
    table = "\n".join(f"| {labels[name]} | {values['ndcg_at_5']:.4f} | {values['mrr_at_10']:.4f} | {values['candidate_recall']:.4f} |"
                       for name, values in test.items())
    story = f'''# Ranking current and archived procedures under access constraints

**An original independent reference implementation by Vedant Thaker, November 5, 2026.** It uses fictional documents and rule-authored judgments, not employer data or a reconstructed customer deployment.

**{decision}**

## Which version of a procedure belongs first?

Two people can type almost identical words and need different versions of a policy. The comparison tests whether a small tree ranker can use authority, document status and current or historical intent alongside text similarity.

BM25, TF-IDF cosine and a locally fitted truncated-SVD projection supply the text signals. The system returns ranked document IDs. It uses no generative model or LLM judge.

## When an archive is the correct match

A lexical ranker rewards overlapping words. A semantic-style channel can add candidate documents whose vocabulary differs. But neither similarity score, by itself, says whether an old procedure is being requested, whether a draft is authoritative, or whether a user is permitted to see the document.

The fictional catalog contains approved procedures, keyword-heavy archives, drafts, discussions, manager-only exceptions and future documents. For a current query, a matching archive can be wrong. For a historical query, the same archive can be exactly the right result. A blanket freshness rule is therefore also an incomplete solution.

## How candidates and relevance grades stay separate

**Scope before indexing.** The document must belong to the query tenant, intersect the trusted principal's groups, and have both its availability and latest update no later than the query time. TF-IDF/SVD statistics and BM25 document frequencies are then fitted only within that permitted view. The cache key hashes that view's permitted contents and metadata; revocation and content updates change the key. In the fixture run, the framework produced **{result['access']['forbidden_candidates_across_all_queries']} forbidden candidates** across all train, validation and test queries. This checks the implemented policy; it is not an authentication or security certification.

**Retrieve a common candidate set.** Each query takes up to 20 BM25 and 20 SVD-channel candidates, deduplicated. Every compared method ranks that same union. Ideal NDCG uses the full eligible corpus. Missing relevant candidates therefore still count against the ranking.

**Represent observable evidence.** The ranker sees {len(result['features'])} named features derived from raw text and document metadata: lexical/TF-IDF/SVD similarity, title/body overlap, source authority, document age/status, and historical/manager words in the request. An explicit field allowlist excludes query IDs, family IDs, relevance grades and qrels. A mutation test changes hidden annotations without changing the feature matrix.

**Rank within query groups.** XGBoost learns a grouped `rank:ndcg` model, rather than pretending query-document rows are independent regression targets. Validation compared four predeclared configurations with depths 2/3 and 40/80 trees. Validation chose **depth {selection['max_depth']}, {selection['n_estimators']} trees** using full-eligible-corpus NDCG@5. The held-out test families were not used to choose hyperparameters. The best non-tree baseline on validation was **{labels[baseline]}**.

**Audit the claim.** The split keeps complete query families apart: {len(result['data']['families']['train'])} train, {len(result['data']['families']['validation'])} validation and {len(result['data']['families']['test'])} test. All tenants, roles and paraphrases of a topic stay in its partition. Documents are a known static catalog; this is a held-out-query-family test, not an unseen-document or unseen-tenant test.

```mermaid
flowchart LR
  A[Trusted tenant, groups, as-of time] --> B[Filter documents before index/cache]
  B --> C[BM25 + TF-IDF/SVD candidate union]
  C --> D[Observable features only]
  D --> E[Grouped XGBRanker]
  E --> F[Ranked document IDs]
  G[Separate graded judgments] --> H[Family-disjoint evaluation]
  F --> H
```

## Ranking on held-out query families

All methods evaluated the same **{test['tree']['queries']} test queries** from three held-out families. NDCG@5 uses gain `2^grade - 1`; MRR@10 treats grade 2 or 3 as relevant. Candidate recall is the fraction of all eligible grade>=2 documents present in the candidate union.

| Method | Mean NDCG@5 | Mean MRR@10 | Candidate recall |
|---|---:|---:|---:|
{table}

![Held-out ranking comparison](ranking.svg)

The paired tree-minus-{labels[baseline]} NDCG difference is **{comparison['mean_difference']:+.4f}**, with a family-bootstrap 95% interval **[{comparison['lower_95']:+.4f}, {comparison['upper_95']:+.4f}]**. Resampling the {comparison['families']} held-out families avoids treating paraphrases and tenant replicas as independent evidence. Three synthetic families are still a very weak basis for external generalization; the interval describes only this constructed fixture.

Across splits the baseline ordering can change. BM25 reached test NDCG@5 {test['bm25']['ndcg_at_5']:.4f}, and fusion reached {test['rrf']['ndcg_at_5']:.4f}. The paired comparison keeps **{labels[baseline]}**, the baseline selected on validation, even if another method scores higher on test.

The most important limit is in the labels themselves. This fixture was designed to make authority, status and historical intent matter, and its judgments were generated from that explicit rule contract. A successful tree fit demonstrates the implemented mechanism under those rules. It is not independent evidence that the ranker will improve real enterprise search.

## Inspect access failures and candidate misses

`failure_demos.json` deliberately compares a dangerous global BM25 index with the scoped candidate path. The unsafe path is used only for these demonstrations, never for training or the search CLI. Inspect its wrong-tenant hits beside the scoped path's forbidden-hit count. The tests separately exercise restricted/future documents, cache reuse after an ACL revocation, and hidden-annotation mutation.

`per_query_test.csv` preserves the scores, candidate denominators and top document/grade for each method. `feature_importance.json` is a model diagnostic, not a causal explanation. `metrics.json` records the training matrix, label, group, dataset, source and model SHA-256 digests, as well as the executed library versions.

## What independent judgments would add

For a relevance study with actual users, the missing evidence is independently reviewed judgments on failed searches, ambiguous acronyms, answerless requests and stale but popular documents. A family/time split declared before inspecting its holdout would let candidate recall and reranking quality be measured separately.

The implemented impact is a reproducible way to challenge a ranking decision while preserving permission boundaries. There are no measured customer accuracy, latency-at-scale, CTR, revenue, or generated-answer claims. The CLI returns ranked fictional documents; it does not answer the user's business question or validate an LLM's response.

## Inspiration and primary references

An [industry discussion of tree-based models](https://www.linkedin.com/posts/andresvourakis_xgboost-and-similar-tree-based-models-are-share-7523776256539029505-auas/) prompted the project topic. The design references [Dropbox's account of Dash relevance labeling and XGBoost ranking](https://dropbox.tech/machine-learning/llm-human-labeling-improving-search-relevance-dropbox-dash) and [XGBoost's learning-to-rank documentation](https://xgboost.readthedocs.io/en/release_3.1.0/tutorials/learning_to_rank.html). Those sources motivate the architecture; no source code, customer data, scale claims or published results were copied. In particular, Dropbox's LLM-assisted label workflow is not implemented or evaluated here.
'''
    (reports / "case-study.md").write_text(story, encoding="utf-8", newline="\n")
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="860" height="420" viewBox="0 0 860 420" role="img" aria-labelledby="title desc">',
           '<title id="title">Held-out query-family ranking comparison</title>',
           '<desc id="desc">Zero-baseline NDCG at five for four methods on the same fictional test queries. This is not a customer performance benchmark.</desc>',
           '<rect width="860" height="420" fill="#f6f8fc"/>',
           '<text x="30" y="34" font-family="sans-serif" font-size="23" fill="#17334a">Similarity and document context need separate checks</text>',
           '<text x="30" y="64" font-family="sans-serif" font-size="15" fill="#496477">NDCG@5 on held-out fictional query families; same permitted candidates</text>']
    for index, (method, values) in enumerate(test.items()):
        y = 100 + index * 65
        width = values["ndcg_at_5"] * 460
        svg.extend([f'<text x="30" y="{y + 23}" font-family="sans-serif" font-size="16">{labels[method]}</text>',
                    f'<rect x="255" y="{y}" width="{width:.2f}" height="34" rx="4" fill="{"#157e7a" if method == "tree" else "#6a8095"}"/>',
                    f'<text x="{265 + width:.2f}" y="{y + 23}" font-family="sans-serif" font-size="17">{values["ndcg_at_5"]:.4f}</text>'])
    svg.extend(['<text x="30" y="391" font-family="sans-serif" font-size="14" fill="#496477">Source: metrics.json. Judgments are original fixture rules, not independent relevance labels.</text>', '</svg>'])
    (reports / "ranking.svg").write_text("\n".join(svg), encoding="utf-8", newline="\n")
