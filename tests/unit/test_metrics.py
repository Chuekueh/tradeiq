from src.evaluation.metrics import RAGMetrics


def test_context_precision():
    retrieved = ["doc1", "doc2", "doc3"]
    relevant = ["doc1", "doc3"]
    assert RAGMetrics.context_precision(retrieved, relevant) == 2 / 3


def test_context_precision_empty():
    assert RAGMetrics.context_precision([], ["doc1"]) == 0.0


def test_context_recall():
    retrieved = ["doc1", "doc2"]
    relevant = ["doc1", "doc3"]
    assert RAGMetrics.context_recall(retrieved, relevant) == 0.5


def test_context_recall_all_retrieved():
    retrieved = ["doc1", "doc2", "doc3"]
    relevant = ["doc1", "doc2"]
    assert RAGMetrics.context_recall(retrieved, relevant) == 1.0


def test_mrr_first_position():
    assert RAGMetrics.mrr(["doc1", "doc2"], ["doc1"]) == 1.0


def test_mrr_second_position():
    assert RAGMetrics.mrr(["doc2", "doc1"], ["doc1"]) == 0.5


def test_mrr_not_found():
    assert RAGMetrics.mrr(["doc2", "doc3"], ["doc1"]) == 0.0


def test_ndcg_perfect():
    retrieved = ["doc1", "doc2"]
    relevant = ["doc1", "doc2"]
    assert RAGMetrics.ndcg_at_k(retrieved, relevant, k=2) == 1.0


def test_ndcg_no_relevant():
    retrieved = ["doc3", "doc4"]
    relevant = ["doc1", "doc2"]
    assert RAGMetrics.ndcg_at_k(retrieved, relevant, k=2) == 0.0
