"""Construct original fictional documents, queries and independent judgment files."""
from __future__ import annotations

import csv
import json
from pathlib import Path

TOPICS = [
    ("deploy", "deployment rollback", "release recovery", "train"),
    ("expenses", "expense reimbursement", "spending claims", "train"),
    ("incident", "incident escalation", "outage response", "train"),
    ("access", "access request", "account permission", "train"),
    ("leave", "leave approval", "time off", "train"),
    ("procurement", "purchase approval", "supplier purchase", "train"),
    ("handoff", "customer handoff", "account transition", "train"),
    ("backups", "backup restoration", "data recovery", "train"),
    ("vendor", "vendor security", "supplier assurance", "validation"),
    ("onboarding", "employee onboarding", "new starter", "validation"),
    ("dataroom", "data room access", "deal documents", "validation"),
    ("retention", "record retention", "document lifecycle", "test"),
    ("travel", "travel approval", "business trip", "test"),
    ("release", "release handoff", "launch transition", "test"),
]
TENANTS = ["aster", "birch", "cedar"]


def generate(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    documents, queries, judgments = [], [], []
    for tenant_index, tenant in enumerate(TENANTS):
        for family_index, (family, label, alias, split) in enumerate(TOPICS):
            prefix = f"{tenant}-{family}"
            facts = f"Route {label} through the service owner. Record the decision and review within {3 + family_index} days."
            common = {"tenant": tenant, "family": family}
            variants = [
                ("current", "Approved", f"{label} approved procedure", f"{alias}. {facts} Current approved guidance for staff, effective now.", "policy", "current", 970, 970, ["all"]),
                ("archive", "Archived", f"{label} prior 2024 procedure", f"{label}. {alias}. Prior 2024 archived procedure: use the earlier owner and a 30 day review. {' '.join([label] * 4)}", "policy", "archived", 300, 300, ["all"]),
                ("draft", "Draft", f"{label} current proposed draft", f"{label}. {alias}. Current draft procedure proposes automatic approval. Not approved. {' '.join([label] * 3)}", "draft", "draft", 998, 998, ["all"]),
                ("discussion", "Discussion", f"Discussion of {alias}", f"Staff discussion about {label}. The current approved procedure is linked from the policy. This thread is not the authoritative decision.", "chat", "current", 990, 990, ["all"]),
                ("manager", "Manager", f"{alias} leadership exceptions", f"Manager exception for {label}: leadership must record an exception before overriding the current approved procedure. {facts}", "policy", "current", 975, 975, ["leadership"]),
                ("future", "Scheduled", f"{label} future approved procedure", f"{label} next procedure would allow a 90 day window. Not available at the query time.", "policy", "current", 1020, 1020, ["all"]),
                ("tangent", "Glossary", f"{label} vocabulary and definitions", f"Definitions of {label} and {alias}; background terms only, no current or prior procedure.", "wiki", "current", 940, 940, ["all"]),
            ]
            for suffix, _, title, body, source, status, updated, available, acl in variants:
                documents.append({**common, "doc_id": f"{prefix}-{suffix}", "title": title,
                                  "body": body, "source": source, "status": status,
                                  "updated_day": updated + tenant_index, "available_day": available,
                                  "acl": acl})
            for principal in ["employee", "manager"]:
                for intent in ["current", "historical"]:
                    for wording in [0, 1]:
                        role_words = "manager exception " if principal == "manager" else "staff "
                        if intent == "current":
                            text = (f"{label} {role_words}current approved procedure" if not wording else
                                    f"where is the active {alias} guidance for {role_words.strip()} today")
                            grades = {"manager": 3, "current": 2, "discussion": 1, "draft": 1} if principal == "manager" else {
                                "current": 3, "discussion": 2, "draft": 1, "archive": 1}
                        else:
                            text = (f"{label} {role_words}prior 2024 archived procedure" if not wording else
                                    f"what did the previous {alias} guidance say in 2024 for {role_words.strip()}")
                            grades = {"archive": 3, "current": 1, "discussion": 1}
                        qid = f"{prefix}-{principal}-{intent}-{wording}"
                        queries.append({"query_id": qid, "family": family, "split": split,
                                        "tenant": tenant, "principal": principal,
                                        "groups": ["all", "leadership"] if principal == "manager" else ["all"],
                                        "text": text, "as_of_day": 1000})
                        for suffix, grade in grades.items():
                            judgments.append({"query_id": qid, "doc_id": f"{prefix}-{suffix}", "grade": grade})
    (directory / "documents.json").write_text(json.dumps(documents, indent=2) + "\n", encoding="utf-8", newline="\n")
    (directory / "queries.json").write_text(json.dumps(queries, indent=2) + "\n", encoding="utf-8", newline="\n")
    with (directory / "qrels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, ["query_id", "doc_id", "grade"])
        writer.writeheader()
        writer.writerows(judgments)
    return {"documents": len(documents), "queries": len(queries), "positive_judgments": len(judgments)}


def load(directory: Path):
    docs = json.loads((directory / "documents.json").read_text(encoding="utf-8"))
    queries = json.loads((directory / "queries.json").read_text(encoding="utf-8"))
    if len({d["doc_id"] for d in docs}) != len(docs) or len({q["query_id"] for q in queries}) != len(queries):
        raise ValueError("Document and query identifiers must be unique")
    doc_ids, query_ids = {d["doc_id"] for d in docs}, {q["query_id"] for q in queries}
    judgments = {}
    with (directory / "qrels.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["query_id"] not in query_ids or row["doc_id"] not in doc_ids or not 0 <= int(row["grade"]) <= 3:
                raise ValueError("Judgment references an unknown identifier or invalid grade")
            if row["doc_id"] in judgments.get(row["query_id"], {}):
                raise ValueError("Duplicate judgment")
            judgments.setdefault(row["query_id"], {})[row["doc_id"]] = int(row["grade"])
    return docs, queries, judgments
