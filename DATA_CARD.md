# Data card: fictional enterprise search fixture

## Origin and ownership

All documents, query texts and judgment rules were authored for this independent portfolio reference implementation. The organizations `aster`, `birch` and `cedar` are fictional tenants. No employer/customer content, source-company dataset, LLM-generated judgment service or external text corpus is used. The fixture is covered by the repository's MIT license.

Regenerate the committed bytes with `python run.py generate`. The executable source is `src/tree_search/fixture.py`. `reports/metrics.json` records the SHA-256 of each input file used in the executed experiment.

## Grain and fields

- `documents.json`: 294 fictional documents. One row is a document version with tenant, ACL groups, title/body, source/status, updated day and available day. The `family` annotation is used to describe the fixture and is explicitly excluded from ranker features.
- `queries.json`: 336 search requests across 14 topic families, three tenants, two principal scenarios, current/historical intent and two wordings. `family`, `split` and `query_id` describe the evaluation; none enters the model feature matrix.
- `qrels.csv`: 1,176 non-zero graded query/document judgments. Every unlisted pair has grade 0 by the fixture's exhaustive construction, not because an unjudged real-world document is assumed irrelevant. Judgments are separate from the feature path.

Time is a fictional day counter, not a calendar date. The fixture uses integers and query time is day 1000. The access predicate accepts only finite, nonnegative real numeric day values; booleans, strings, missing values, NaN and infinities fail closed. Tenant and group context must also have the expected string/list types. Documents can be current, archived, draft, discussion, manager-restricted or not yet available. A document must satisfy both availability and latest-update time. The implementation has no version-history store, so an updated-after-query document is excluded rather than exposing its future content.

## Judgment contract

Relevance grades are ordinal 0-3. These are rule-authored fixture labels, not independent human evaluations or an LLM judge's assessment.

| Request | Grade 3 | Grade 2 | Grade 1 |
|---|---|---|---|
| Current employee procedure | Approved current policy | Current discussion pointing to the policy | Draft and archived procedure |
| Current manager exception | Permitted manager exception policy | Current general policy | Discussion and draft |
| Historical/2024 procedure | Archived procedure | None | Current general policy and discussion |

Other-family, other-tenant, future, tangential and unlisted pairs receive grade 0. Permission/time filtering occurs independently of relevance labels and before indexing/candidate creation. For an employee, manager-only documents cannot become candidates. For a manager, the relevant restricted document can be retrieved. An inaccessible positive judgment would not enter the eligible-corpus ideal metric denominator.

Because the labels were designed around authority/status/intent compatibility, high ranking scores partially reflect those construction rules. This is intentional for inspecting feature interactions and failure boundaries. It is not evidence of independent search relevance or a deployable accuracy estimate.

## Declared split and candidate policy

Train topics: deployment rollback, expense reimbursement, incident escalation, access requests, leave approval, purchase approval, customer handoff and backup restoration.

Validation topics: vendor security, employee onboarding and data room access.

Test topics: record retention, travel approval and release handoff.

Every query belonging to a topic stays in the same split across tenants, roles and paraphrases. The permitted static document catalog is known at indexing time and includes documents associated with held-out query themes. This is a query-family holdout, not an inductive unseen-document benchmark. Unsupervised TF-IDF/SVD fitting uses no qrels or query labels.

Each method ranks the same candidate union: at most 20 BM25 and 20 SVD-channel results after access/time filtering. NDCG@5's ideal ranking uses all eligible documents rather than the candidate union, retaining penalties for missed relevant candidates. MRR@10 regards grades 2/3 as relevant. Candidate recall is eligible grade>=2 documents present divided by all eligible grade>=2 documents.

## Boundaries

- No generated answer, LLM quality, real customer relevance, online CTR, revenue or serving-scale latency is measured.
- Three test families are too few to support broad performance claims. Bootstrap intervals remain conditional on a small constructed fixture.
- The role flag in the CLI is a fictional scenario selector. There is no authentication system, production policy engine or security certification.
- Repeated tenant corpora and templated paraphrases are correlated. The uncertainty calculation resamples families instead of treating 72 test requests as independent.
- Source metadata is assumed available and correct. A real deployment must validate provenance, source freshness and permission propagation.
