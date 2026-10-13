from dataclasses import replace

import pytest

from ai_quality_evals.models import Document
from ai_quality_evals.retrieval import BM25Retriever, tokenize


def _doc(doc_id: str, content: str) -> Document:
    return Document(doc_id, "Ignored title", "1.0", "2026-01-01", "current", content)


def test_bm25_independent_numeric_score_and_term_frequency() -> None:
    retriever = BM25Retriever([_doc("a", "alpha alpha"), _doc("b", "beta")], k1=1, b=0)
    hits = retriever.retrieve("alpha").hits
    # N=2, df=1 => IDF=ln(2); tf=2, k1=1, b=0 => multiplier 4/3.
    assert len(hits) == 1
    assert hits[0].document_id == "a"
    assert hits[0].score == pytest.approx(0.924196240747, abs=1e-12)
    assert hits[0].score > retriever.retrieve("beta").hits[0].score


def test_query_changes_actual_rank() -> None:
    retriever = BM25Retriever([_doc("a", "enrollment enrollment"), _doc("b", "dental dental")])
    assert [hit.document_id for hit in retriever.retrieve("enrollment").hits] == ["a"]
    assert [hit.document_id for hit in retriever.retrieve("dental").hits] == ["b"]


def test_length_normalization_prefers_concentrated_match() -> None:
    retriever = BM25Retriever([_doc("a", "alpha filler filler filler"), _doc("b", "alpha")])
    assert [hit.document_id for hit in retriever.retrieve("alpha").hits] == ["b", "a"]


def test_ties_use_id_order_and_ignore_input_order() -> None:
    docs = [_doc("z", "alpha"), _doc("a", "alpha"), _doc("m", "alpha")]
    expected = ["a", "m", "z"]
    for order in (docs, list(reversed(docs))):
        result = BM25Retriever(order).retrieve("alpha", top_k=10)
        assert [hit.document_id for hit in result.hits] == expected
        assert [hit.score for hit in result.hits] == pytest.approx([0.133531392625] * 3)


def test_empty_no_match_and_repeated_query_terms() -> None:
    retriever = BM25Retriever([_doc("a", "alpha"), _doc("b", "beta")])
    assert retriever.retrieve("").hits == ()
    assert retriever.retrieve("! ?").hits == ()
    assert retriever.retrieve("unknown").hits == ()
    assert retriever.retrieve("alpha alpha").hits == retriever.retrieve("alpha").hits


def test_top_k_and_query_are_exposed() -> None:
    retriever = BM25Retriever([_doc("a", "alpha"), _doc("b", "alpha")])
    result = retriever.retrieve("ALPHA", top_k=1)
    assert result.query == "ALPHA"
    assert result.top_k == 1
    assert [hit.document_id for hit in result.hits] == ["a"]


@pytest.mark.parametrize("top_k", [0, -1, True])
def test_invalid_k(top_k: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        BM25Retriever([_doc("a", "alpha")]).retrieve("alpha", top_k=top_k)


@pytest.mark.parametrize(("k1", "b"), [(0, 0.75), (1.5, -1), (1.5, 2), (float("nan"), 0.75)])
def test_invalid_parameters(k1: float, b: float) -> None:
    with pytest.raises(ValueError, match="BM25 requires"):
        BM25Retriever([_doc("a", "alpha")], k1=k1, b=b)


@pytest.mark.parametrize("docs", [[], [_doc("a", "alpha"), _doc("a", "beta")], [_doc("a", "")]])
def test_invalid_index(docs: list[Document]) -> None:
    with pytest.raises(ValueError, match="index requires"):
        BM25Retriever(docs)


def test_tokenizer_definition() -> None:
    assert tokenize("In-NETWORK: $1,500; 2026.") == ("in", "network", "1", "500", "2026")


def test_metadata_and_gold_are_not_retrieval_hints() -> None:
    doc = _doc("private-id", "alpha")
    modified = replace(doc, title="goldterm", status="archived", version="goldterm")
    assert BM25Retriever([modified]).retrieve("goldterm private-id").hits == ()
    assert (
        BM25Retriever([modified]).retrieve("alpha").hits
        == BM25Retriever([doc]).retrieve("alpha").hits
    )
