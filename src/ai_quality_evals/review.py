"""Strict reviewer-owned annotations and separate deterministic semantic review reports."""

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Literal, cast

from ai_quality_evals.consistency import RULES, RULES_VERSION, check_consistency
from ai_quality_evals.runner import Report, report_json
from ai_quality_evals.validation import (
    ValidationError,
    array,
    object_fields,
    parse_json,
    strings,
    text,
)

DIMENSIONS = (
    "answer_correctness",
    "groundedness",
    "necessary_fact_coverage",
    "appropriate_abstention",
    "citation_support",
)
DEFECT_LABELS = frozenset(
    {
        "evidence_contradiction",
        "unsupported_personal_claim",
        "unsupported_claim",
        "incorrect_abstention",
        "missing_evidence",
        "inadequate_abstention_explanation",
    }
)
type ReviewStatus = Literal["unreviewed", "unreviewable", "reviewed"]
type AnnotationOrigin = Literal["human_review", "synthetic_fixture"]


@dataclass(frozen=True, slots=True)
class CaseReview:
    case_id: str
    status: ReviewStatus
    reviewer: str | None
    reviewed_at: str | None
    scores: dict[str, int | None]
    justifications: dict[str, str | None]
    defect_labels: tuple[str, ...]
    notes: str | None


@dataclass(frozen=True, slots=True)
class ReviewSet:
    schema_version: Literal[1]
    annotation_origin: AnnotationOrigin
    capture_sha256: str
    reviews: tuple[CaseReview, ...]


def prepare_reviews(report: Report) -> ReviewSet:
    """Create unscored rows only; never create a reviewer, timestamp or rating."""
    capture_hash = report.input_sha256["traces"]
    if capture_hash is None:
        raise ValidationError("review preparation requires captured or replayed responses")
    return ReviewSet(
        1,
        "human_review",
        capture_hash,
        tuple(
            CaseReview(
                row.case.case_id,
                "unreviewed" if row.response is not None else "unreviewable",
                None,
                None,
                dict.fromkeys(DIMENSIONS),
                dict.fromkeys(DIMENSIONS),
                (),
                None if row.response is not None else "No completed, parsed model response.",
            )
            for row in report.results
        ),
    )


def _nullable_text(value: object) -> str | None:
    return None if value is None else text(value)


def _timestamp(value: object) -> str | None:
    if value is None:
        return None
    result = text(value)
    if not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|[+-]\d\d:\d\d)", result):
        raise ValidationError("reviewed_at must be an ISO 8601 timestamp with timezone")
    if not result.endswith("Z") and (int(result[-5:-3]) > 23 or int(result[-2:]) > 59):
        raise ValidationError("invalid reviewed_at timezone offset")
    try:
        datetime.fromisoformat(result)
    except ValueError as exc:
        raise ValidationError("invalid reviewed_at timestamp") from exc
    return result


def parse_reviews(value: object, report: Report) -> ReviewSet:
    obj = object_fields(value, {"schema_version", "annotation_origin", "capture_sha256", "reviews"})
    if type(obj["schema_version"]) is not int or obj["schema_version"] != 1:
        raise ValidationError("unsupported review schema_version; expected integer 1")
    origin = text(obj["annotation_origin"])
    if origin not in {"human_review", "synthetic_fixture"}:
        raise ValidationError("annotation_origin must be human_review or synthetic_fixture")
    capture_hash = text(obj["capture_sha256"])
    if not re.fullmatch(r"[0-9a-f]{64}", capture_hash):
        raise ValidationError("capture_sha256 must be a lowercase SHA-256 hash")
    if capture_hash != report.input_sha256["traces"]:
        raise ValidationError("review capture hash mismatch")
    rows = {row.case.case_id: row for row in report.results}
    reviews: dict[str, CaseReview] = {}
    for value in array(obj["reviews"]):
        entry = object_fields(
            value,
            {
                "case_id",
                "status",
                "reviewer",
                "reviewed_at",
                "scores",
                "justifications",
                "defect_labels",
                "notes",
            },
        )
        case_id = text(entry["case_id"], identifier=True)
        if case_id not in rows:
            raise ValidationError(f"unknown reviewed case ID: {case_id}")
        if case_id in reviews:
            raise ValidationError(f"duplicate case review: {case_id}")
        status = text(entry["status"])
        if status not in {"unreviewed", "unreviewable", "reviewed"}:
            raise ValidationError(f"unknown review status: {status}")
        reviewer = _nullable_text(entry["reviewer"])
        timestamp = _timestamp(entry["reviewed_at"])
        ratings = object_fields(entry["scores"], set(DIMENSIONS))
        scores: dict[str, int | None] = {}
        for dimension, rating in ratings.items():
            if rating is not None and (type(rating) is not int or rating not in {0, 1, 2}):
                raise ValidationError(
                    f"{case_id}: invalid rating for {dimension}; use 0, 1, 2 or null"
                )
            scores[dimension] = rating
        rationales = object_fields(entry["justifications"], set(DIMENSIONS))
        justifications = {name: _nullable_text(rationale) for name, rationale in rationales.items()}
        labels = strings(entry["defect_labels"])
        if set(labels) - DEFECT_LABELS:
            raise ValidationError(f"unknown defect labels: {sorted(set(labels) - DEFECT_LABELS)}")
        notes = _nullable_text(entry["notes"])
        reviewable = rows[case_id].response is not None
        if status == "reviewed":
            if not reviewable:
                raise ValidationError(f"{case_id}: cannot rate a failed or unreviewable generation")
            if reviewer is None or timestamp is None:
                raise ValidationError(f"{case_id}: reviewed rows require reviewer and reviewed_at")
            if any(rationale is None for rationale in justifications.values()):
                raise ValidationError(
                    f"{case_id}: every dimension needs a justification, including N/A"
                )
        else:
            if (
                any(rating is not None for rating in scores.values())
                or any(rationale is not None for rationale in justifications.values())
                or labels
            ):
                raise ValidationError(
                    f"{case_id}: {status} rows cannot contain ratings or defect labels"
                )
            if status == "unreviewed":
                if (
                    not reviewable
                    or reviewer is not None
                    or timestamp is not None
                    or notes is not None
                ):
                    raise ValidationError(
                        f"{case_id}: unreviewed requires a parsed response and no annotations"
                    )
            elif (
                notes is None
                or (reviewer is None) != (timestamp is None)
                or (reviewable and reviewer is None)
            ):
                raise ValidationError(
                    f"{case_id}: unreviewable needs an explanation and reviewer identity "
                    "for parsed responses"
                )
        reviews[case_id] = CaseReview(
            case_id,
            cast(ReviewStatus, status),
            reviewer,
            timestamp,
            scores,
            justifications,
            tuple(sorted(labels)),
            notes,
        )
    return ReviewSet(
        1,
        cast(AnnotationOrigin, origin),
        capture_hash,
        tuple(reviews[key] for key in sorted(reviews)),
    )


def load_reviews(path: Path, report: Report) -> tuple[ReviewSet, str]:
    raw = path.read_bytes()
    try:
        return parse_reviews(parse_json(raw.decode("utf-8")), report), hashlib.sha256(
            raw
        ).hexdigest()
    except (ValueError, UnicodeError) as exc:
        raise ValidationError(f"{path}: {exc}") from exc


def summarize_reviews(report: Report, reviews: ReviewSet) -> dict[str, object]:
    by_id = {review.case_id: review for review in reviews.reviews}
    statuses = {
        row.case.case_id: by_id[row.case.case_id].status
        if row.case.case_id in by_id
        else ("unreviewed" if row.response is not None else "unreviewable")
        for row in report.results
    }
    reviewed = [review for review in reviews.reviews if review.status == "reviewed"]
    dimensions: dict[str, object] = {}
    for name in DIMENSIONS:
        ratings = [review.scores[name] for review in reviewed if review.scores[name] is not None]
        applicable = cast(list[int], ratings)
        dimensions[name] = {
            "mean": fmean(applicable) if applicable else None,
            "applicable_denominator": len(applicable),
            "rating_counts": {str(rating): applicable.count(rating) for rating in (0, 1, 2)},
            "inapplicable_case_ids": [
                review.case_id for review in reviewed if review.scores[name] is None
            ],
        }
    issues = [
        {
            "case_id": review.case_id,
            "zero_score_dimensions": [name for name in DIMENSIONS if review.scores[name] == 0],
            "defect_labels": review.defect_labels,
            "reviewer": review.reviewer,
        }
        for review in reviewed
        if 0 in review.scores.values() or review.defect_labels
    ]
    return {
        "annotation_origin": reviews.annotation_origin,
        "semantic_status": (
            "synthetic_annotations_recorded"
            if reviews.annotation_origin == "synthetic_fixture"
            else "human_judgment_recorded"
        )
        if reviewed
        else "unreviewed",
        "case_count": len(statuses),
        "status_counts": {
            status: sum(value == status for value in statuses.values())
            for status in ("reviewed", "unreviewed", "unreviewable")
        },
        "missing_review_case_ids": sorted(set(statuses) - set(by_id)),
        "unreviewed_case_ids": [key for key, status in statuses.items() if status == "unreviewed"],
        "unreviewable_case_ids": [
            key for key, status in statuses.items() if status == "unreviewable"
        ],
        "dimensions": dimensions,
        "defect_label_counts": dict(
            sorted(Counter(label for review in reviewed for label in review.defect_labels).items())
        ),
        "reviewer_findings": issues,
        "material_review_findings": [issue for issue in issues if issue["zero_score_dimensions"]],
        "high_impact_unsupported_claim_case_ids": [
            review.case_id
            for review in reviewed
            if "unsupported_personal_claim" in review.defect_labels
        ],
        "cases": [
            {
                "case_id": key,
                "status": status,
                "annotation": asdict(by_id[key]) if key in by_id else None,
            }
            for key, status in statuses.items()
        ],
    }


def compare_reviews(
    left: ReviewSet, right: ReviewSet, case_ids: tuple[str, ...] = ()
) -> dict[str, object]:
    """Compare paired authored ratings only; never infer defects from response text."""
    if left.annotation_origin != right.annotation_origin:
        raise ValidationError("cannot compare human_review and synthetic_fixture annotations")
    first = {review.case_id: review for review in left.reviews if review.status == "reviewed"}
    second = {review.case_id: review for review in right.reviews if review.status == "reviewed"}
    paired = sorted(set(first) & set(second))
    all_ids = set(case_ids) | {review.case_id for review in left.reviews + right.reviews}
    dimensions: dict[str, object] = {}
    regressions: list[dict[str, object]] = []
    for name in DIMENSIONS:
        pairs = [
            (case_id, first[case_id].scores[name], second[case_id].scores[name])
            for case_id in paired
            if first[case_id].scores[name] is not None and second[case_id].scores[name] is not None
        ]
        before = [cast(int, old) for _, old, _ in pairs]
        after = [cast(int, new) for _, _, new in pairs]
        dimensions[name] = {
            "paired_applicable_denominator": len(pairs),
            "left_mean": fmean(before) if pairs else None,
            "right_mean": fmean(after) if pairs else None,
            "mean_delta": fmean(new - old for old, new in zip(before, after, strict=True))
            if pairs
            else None,
            "unpaired_or_inapplicable_case_ids": sorted(all_ids - {key for key, _, _ in pairs}),
        }
        for case_id, old, new in pairs:
            if cast(int, new) < cast(int, old):
                regressions.append(
                    {
                        "case_id": case_id,
                        "dimension": name,
                        "left_score": old,
                        "right_score": new,
                        "material": new == 0,
                    }
                )
    return {
        "direction": "right_minus_left",
        "paired_reviewed_case_ids": paired,
        "not_paired_case_ids": sorted(all_ids - set(paired)),
        "dimensions": dimensions,
        "score_regressions": regressions,
        "material_regressions": [row for row in regressions if row["material"]],
        "new_defect_labels": [
            {
                "case_id": case_id,
                "defect_labels": sorted(
                    set(second[case_id].defect_labels) - set(first[case_id].defect_labels)
                ),
            }
            for case_id in paired
            if set(second[case_id].defect_labels) - set(first[case_id].defect_labels)
        ],
        "reviewer_changed_case_ids": [
            case_id for case_id in paired if first[case_id].reviewer != second[case_id].reviewer
        ],
    }


def semantic_report(report: Report, reviews: ReviewSet, review_sha256: str) -> dict[str, object]:
    findings = [
        asdict(finding)
        for row in report.results
        if row.response is not None
        for finding in check_consistency(row.case.case_id, row.response)
    ]
    return {
        "capture_sha256": reviews.capture_sha256,
        "review_sha256": review_sha256,
        "capture_origin": report.response_origin,
        "prompt_versions": sorted({record.prompt.version for record in report.captures}),
        "automatic_evaluation": parse_json(report_json(report)),
        "response_consistency": {
            "rules_version": RULES_VERSION,
            "rules": RULES,
            "checked_case_ids": [
                row.case.case_id for row in report.results if row.response is not None
            ],
            "uncheckable_case_ids": [
                row.case.case_id for row in report.results if row.response is None
            ],
            "findings": findings,
        },
        "human_semantic_review": summarize_reviews(report, reviews),
    }


def artifact_json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def write_private_artifact(path: Path, value: object, inputs: tuple[Path, ...]) -> None:
    if not {"live-captures", "human-reviews"}.intersection(path.resolve().parts):
        raise ValidationError(
            "review artifacts must be under Git-ignored live-captures/ or human-reviews/"
        )
    if any(path.resolve() == source.resolve() for source in inputs):
        raise ValidationError("review output must not overwrite captures or annotations")
    serialized = artifact_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
