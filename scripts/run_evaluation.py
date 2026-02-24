"""Run the TradeIQ RAG evaluation pipeline."""

import argparse

from src.config import get_settings
from src.embeddings.embedding_service import EmbeddingService
from src.evaluation.dataset import EvaluationDataset
from src.evaluation.evaluator import RAGEvaluator
from src.generation.chain import RAGChain
from src.generation.llm_service import LLMService
from src.generation.response_parser import ResponseParser
from src.memory.conversation_memory import ConversationMemory
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.reranker import Reranker
from src.vectorstore.chroma_store import ChromaVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TradeIQ RAG evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/evaluation/golden_qa.json",
        help="Path to evaluation dataset",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="docs/evaluation-report.md",
        help="Output path for markdown report",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="data/evaluation/results.json",
        help="Output path for JSON results",
    )
    args = parser.parse_args()

    settings = get_settings()
    print("Initializing components...")

    embedding_service = EmbeddingService(model_name=settings.embedding_model.value)
    vector_store = ChromaVectorStore(settings)
    memory = ConversationMemory(db_path="./data/eval_conversations.db")
    llm = LLMService(settings)

    searcher = HybridSearcher(
        vector_store=vector_store,
        embedding_service=embedding_service,
        alpha=settings.hybrid_search_alpha,
    )
    searcher.build_bm25_index()

    reranker = Reranker()
    response_parser = ResponseParser()

    chain = RAGChain(
        searcher=searcher,
        reranker=reranker,
        llm_service=llm,
        memory=memory,
        response_parser=response_parser,
        retrieval_top_k=settings.retrieval_top_k,
        rerank_top_k=settings.rerank_top_k,
    )

    # Load dataset
    dataset = EvaluationDataset.from_json(args.dataset)
    print(f"Loaded {len(dataset)} evaluation samples")

    # Run evaluation
    evaluator = RAGEvaluator(rag_chain=chain, llm=llm)
    report = evaluator.evaluate(dataset)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Context Precision:  {report.avg_context_precision:.4f}")
    print(f"  Context Recall:     {report.avg_context_recall:.4f}")
    print(f"  MRR:                {report.avg_mrr:.4f}")
    print(f"  nDCG@5:             {report.avg_ndcg:.4f}")
    print(f"  Faithfulness:       {report.avg_faithfulness:.4f}")
    print(f"  Answer Relevancy:   {report.avg_answer_relevancy:.4f}")
    print(f"  Avg Latency:        {report.avg_latency_ms:.0f}ms")
    print(f"  Total Time:         {report.total_time_s:.1f}s")
    print("=" * 60)

    # Export
    RAGEvaluator.export_report(report, args.output_md)
    RAGEvaluator.export_json(report, args.output_json)
    print(f"\nReport saved to: {args.output_md}")
    print(f"JSON saved to:   {args.output_json}")

    memory.close()


if __name__ == "__main__":
    main()
