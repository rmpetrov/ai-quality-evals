"""Offline orchestration and reproducible JSON report serialization."""

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from ai_quality_evals import __version__
from ai_quality_evals.metrics import (
    Aggregate,
    CaseResult,
    ThresholdFailure,
    aggregate,
    check_thresholds,
    evaluate_case,
)
from ai_quality_evals.retrieval import BM25Retriever
from ai_quality_evals.validation import load_cases, load_corpus, load_records

DEFAULT_DATA = Path(__file__).parent / "data"


@dataclass(frozen=True, slots=True)
class Report:
    schema_version: int
    evaluator_version: str
    mode: Literal["retrieval", "replay"]
    response_origin: Literal["synthetic_fixture"] | None
    evaluation_k: int
    retrieval_config: dict[str, str | float] | None
    input_sha256: dict[str, str | None]
    dataset_case_count: int
    omitted_case_ids: tuple[str, ...]
    results: tuple[CaseResult, ...]
    aggregate: Aggregate
    by_category: dict[str, Aggregate]
    minimums: dict[str, float]
    threshold_failures: tuple[ThresholdFailure, ...]

    @property
    def gate_status(self) -> Literal["not_evaluated", "passed", "failed"]:
        if not self.minimums:
            return "not_evaluated"
        return "failed" if self.threshold_failures else "passed"

    @property
    def passed(self) -> bool | None:
        status = self.gate_status
        return None if status == "not_evaluated" else status == "passed"


def _corpus_hash(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.md")):
        digest.update(path.name.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def run_evaluation(
    *,
    kb: Path = DEFAULT_DATA / "kb",
    cases_path: Path = DEFAULT_DATA / "cases.jsonl",
    traces_path: Path | None = None,
    top_k: int = 3,
    minimums: Mapping[str, float] | None = None,
) -> Report:
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    documents = load_corpus(kb)
    cases = load_cases(cases_path, documents)
    evaluated: list[CaseResult] = []
    config: dict[str, str | float] | None = None
    if traces_path is None:
        retriever = BM25Retriever(documents.values())
        config = {
            "algorithm": "bm25",
            "k1": retriever.k1,
            "b": retriever.b,
            "tokenizer": "ascii_casefold_v1",
            "query_terms": "distinct_sorted",
        }
        for case in cases:
            retrieval = retriever.retrieve(case.question, top_k=top_k)
            evaluated.append(evaluate_case(case, retrieval, documents, top_k))
    else:
        case_map = {case.case_id: case for case in cases}
        records = load_records(traces_path, cases, documents)
        for record in records:
            evaluated.append(
                evaluate_case(
                    case_map[record.case_id], record.retrieval, documents, top_k, record.response
                )
            )
    # Stable case order for equivalent trace files; raw file hash still captures input order.
    results = tuple(sorted(evaluated, key=lambda result: result.case.case_id))
    summary = aggregate(results)
    thresholds = dict(minimums or {})
    evaluated_ids = {result.case.case_id for result in results}
    return Report(
        1,
        __version__,
        "retrieval" if traces_path is None else "replay",
        None if traces_path is None else "synthetic_fixture",
        top_k,
        config,
        {
            "corpus": _corpus_hash(kb),
            "cases": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
            "traces": hashlib.sha256(traces_path.read_bytes()).hexdigest() if traces_path else None,
        },
        len(cases),
        tuple(sorted(case.case_id for case in cases if case.case_id not in evaluated_ids)),
        results,
        summary,
        {
            category: aggregate([result for result in results if result.case.category == category])
            for category in sorted({result.case.category for result in results})
        },
        thresholds,
        check_thresholds(summary, thresholds),
    )


def report_json(report: Report) -> str:
    payload = asdict(report)
    payload["gate_status"] = report.gate_status
    payload["passed"] = report.passed
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def write_report(report: Report, path: Path) -> None:
    """Atomically replace a report; a failure never leaves partial JSON at the destination."""
    serialized = report_json(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as out:
        temporary_path = Path(out.name)
        try:
            out.write(serialized)
            out.close()
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
