# Ranking current and archived procedures under access constraints

**An original independent reference implementation by Vedant Thaker, November 5, 2026.** It uses fictional documents and rule-authored judgments, not employer data or a reconstructed customer deployment.

**Keep the tree as the next evaluation candidate, then collect independent judgments before any deployment.**

## Which version of a procedure belongs first?

Two people can type almost identical words and need different versions of a policy. The comparison tests whether a small tree ranker can use authority, document status and current or historical intent alongside text similarity.

BM25, TF-IDF cosine and a locally fitted truncated-SVD projection supply the text signals. The system returns ranked document IDs. It uses no generative model or LLM judge.

## When an archive is the correct match

A lexical ranker rewards overlapping words. A semantic-style channel can add candidate documents whose vocabulary differs. But neither similarity score, by itself, says whether an old procedure is being requested, whether a draft is authoritative, or whether a user is permitted to see the document.

The fictional catalog contains approved procedures, keyword-heavy archives, drafts, discussions, manager-only exceptions and future documents. For a current query, a matching archive can be wrong. For a historical query, the same archive can be exactly the right result. A blanket freshness rule is therefore also an incomplete solution.

## How candidates and relevance grades stay separate

**Scope before indexing.** The document must belong to the query tenant, intersect the trusted principal's groups, and have both its availability and latest update no later than the query time. TF-IDF/SVD statistics and BM25 document frequencies are then fitted only within that permitted view. The cache key hashes that view's permitted contents and metadata; revocation and content updates change the key. In the fixture run, the framework produced **0 forbidden candidates** across all train, validation and test queries. This checks the implemented policy; it is not an authentication or security certification.

**Retrieve a common candidate set.** Each query takes up to 20 BM25 and 20 SVD-channel candidates, deduplicated. Every compared method ranks that same union. Ideal NDCG uses the full eligible corpus. Missing relevant candidates therefore still count against the ranking.

**Represent observable evidence.** The ranker sees 17 named features derived from raw text and document metadata: lexical/TF-IDF/SVD similarity, title/body overlap, source authority, document age/status, and historical/manager words in the request. An explicit field allowlist excludes query IDs, family IDs, relevance grades and qrels. A mutation test changes hidden annotations without changing the feature matrix.

**Rank within query groups.** XGBoost learns a grouped `rank:ndcg` model, rather than pretending query-document rows are independent regression targets. Validation compared four predeclared configurations with depths 2/3 and 40/80 trees. Validation chose **depth 3, 80 trees** using full-eligible-corpus NDCG@5. The held-out test families were not used to choose hyperparameters. The best non-tree baseline on validation was **BM25 + SVD fusion**.

**Audit the claim.** The split keeps complete query families apart: 8 train, 3 validation and 3 test. All tenants, roles and paraphrases of a topic stay in its partition. Documents are a known static catalog; this is a held-out-query-family test, not an unseen-document or unseen-tenant test.

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

All methods evaluated the same **72 test queries** from three held-out families. NDCG@5 uses gain `2^grade - 1`; MRR@10 treats grade 2 or 3 as relevant. Candidate recall is the fraction of all eligible grade>=2 documents present in the candidate union.

| Method | Mean NDCG@5 | Mean MRR@10 | Candidate recall |
|---|---:|---:|---:|
| BM25 | 0.8110 | 0.7882 | 1.0000 |
| BM25 + SVD fusion | 0.7706 | 0.7795 | 1.0000 |
| Metadata heuristic | 0.8047 | 0.8438 | 1.0000 |
| XGBRanker | 0.9699 | 1.0000 | 1.0000 |

![Held-out ranking comparison](ranking.svg)

The paired tree-minus-BM25 + SVD fusion NDCG difference is **+0.1993**, with a family-bootstrap 95% interval **[+0.1377, +0.2578]**. Resampling the 3 held-out families avoids treating paraphrases and tenant replicas as independent evidence. Three synthetic families are still a very weak basis for external generalization; the interval describes only this constructed fixture.

Across splits the baseline ordering can change. BM25 reached test NDCG@5 0.8110, and fusion reached 0.7706. The paired comparison keeps **BM25 + SVD fusion**, the baseline selected on validation, even if another method scores higher on test.

The most important limit is in the labels themselves. This fixture was designed to make authority, status and historical intent matter, and its judgments were generated from that explicit rule contract. A successful tree fit demonstrates the implemented mechanism under those rules. It is not independent evidence that the ranker will improve real enterprise search.

## Inspect access failures and candidate misses

`failure_demos.json` deliberately compares a dangerous global BM25 index with the scoped candidate path. The unsafe path is used only for these demonstrations, never for training or the search CLI. Inspect its wrong-tenant hits beside the scoped path's forbidden-hit count. The tests separately exercise restricted/future documents, cache reuse after an ACL revocation, and hidden-annotation mutation.

`per_query_test.csv` preserves the scores, candidate denominators and top document/grade for each method. `feature_importance.json` is a model diagnostic, not a causal explanation. `metrics.json` records the training matrix, label, group, dataset, source and model SHA-256 digests, as well as the executed library versions.

## What independent judgments would add

For a relevance study with actual users, the missing evidence is independently reviewed judgments on failed searches, ambiguous acronyms, answerless requests and stale but popular documents. A family/time split declared before inspecting its holdout would let candidate recall and reranking quality be measured separately.

The implemented impact is a reproducible way to challenge a ranking decision while preserving permission boundaries. There are no measured customer accuracy, latency-at-scale, CTR, revenue, or generated-answer claims. The CLI returns ranked fictional documents; it does not answer the user's business question or validate an LLM's response.

## Inspiration and primary references

An [industry discussion of tree-based models](https://www.linkedin.com/posts/andresvourakis_xgboost-and-similar-tree-based-models-are-share-7523776256539029505-auas/) prompted the project topic. The design references [Dropbox's account of Dash relevance labeling and XGBoost ranking](https://dropbox.tech/machine-learning/llm-human-labeling-improving-search-relevance-dropbox-dash) and [XGBoost's learning-to-rank documentation](https://xgboost.readthedocs.io/en/release_3.1.0/tutorials/learning_to_rank.html). Those sources motivate the architecture; no source code, customer data, scale claims or published results were copied. In particular, Dropbox's LLM-assisted label workflow is not implemented or evaluated here.
