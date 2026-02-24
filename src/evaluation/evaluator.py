import json
import time
from dataclasses import dataclass
from pathlib import Path

import structlog

from src.evaluation.dataset import EvalSample, EvaluationDataset
from src.evaluation.metrics import RAGMetrics
from src.generation.chain import RAGChain, RAGResponse
from src.generation.llm_service import LLMService

logger = structlog.get_logger()


@dataclass
class EvalResult:
    question: str
    answer: str
    ground_truth: str
    context_precision: float
    context_recall: float
    mrr: float
    ndcg: float
    faithfulness: float
    answer_relevancy: float
    latency_ms: float


@dataclass
class EvaluationReport:
    results: list[EvalResult]
    avg_context_precision: float
    avg_context_recall: float
    avg_mrr: float
    avg_ndcg: float
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_latency_ms: float
    total_time_s: float


class RAGEvaluator:
    def __init__(self, rag_chain: RAGChain, llm: LLMService):
        self._chain = rag_chain
        self._llm = llm

    def evaluate(self, dataset: EvaluationDataset) -> EvaluationReport:
        """Run full evaluation over the dataset."""
        start = time.perf_counter()
        results: list[EvalResult] = []

        for i, sample in enumerate(dataset.samples):
            logger.info("evaluating", sample=i + 1, total=len(dataset))
            result = self._evaluate_sample(sample)
            results.append(result)

        total_time = time.perf_counter() - start

        report = EvaluationReport(
            results=results,
            avg_context_precision=self._avg(r.context_precision for r in results),
            avg_context_recall=self._avg(r.context_recall for r in results),
            avg_mrr=self._avg(r.mrr for r in results),
            avg_ndcg=self._avg(r.ndcg for r in results),
            avg_faithfulness=self._avg(r.faithfulness for r in results),
            avg_answer_relevancy=self._avg(r.answer_relevancy for r in results),
            avg_latency_ms=self._avg(r.latency_ms for r in results),
            total_time_s=round(total_time, 1),
        )

        logger.info(
            "evaluation_complete",
            samples=len(results),
            precision=report.avg_context_precision,
            recall=report.avg_context_recall,
            faithfulness=report.avg_faithfulness,
        )
        return report

    def _evaluate_sample(self, sample: EvalSample) -> EvalResult:
        response: RAGResponse = self._chain.invoke(sample.question, session_id="eval")

        retrieved_sources = [s.source for s in response.sources]
        contexts = [s.content for s in response.sources]

        gt_sources = sample.ground_truth_sources
        return EvalResult(
            question=sample.question,
            answer=response.answer,
            ground_truth=sample.ground_truth_answer,
            context_precision=RAGMetrics.context_precision(retrieved_sources, gt_sources),
            context_recall=RAGMetrics.context_recall(retrieved_sources, gt_sources),
            mrr=RAGMetrics.mrr(retrieved_sources, gt_sources),
            ndcg=RAGMetrics.ndcg_at_k(retrieved_sources, gt_sources),
            faithfulness=RAGMetrics.faithfulness(response.answer, contexts, self._llm),
            answer_relevancy=RAGMetrics.answer_relevancy(
                sample.question, response.answer, self._llm
            ),
            latency_ms=response.query_time_ms,
        )

    @staticmethod
    def _avg(values) -> float:  # type: ignore[no-untyped-def]
        vals = list(values)
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    @staticmethod
    def export_report(report: EvaluationReport, output_path: str) -> None:
        """Export evaluation report as markdown."""
        md = f"""# TradeIQ RAG Evaluation Report

## Summary

| Metric | Score |
|--------|-------|
| Context Precision | {report.avg_context_precision:.4f} |
| Context Recall | {report.avg_context_recall:.4f} |
| MRR | {report.avg_mrr:.4f} |
| nDCG@5 | {report.avg_ndcg:.4f} |
| Faithfulness | {report.avg_faithfulness:.4f} |
| Answer Relevancy | {report.avg_answer_relevancy:.4f} |
| Avg Latency | {report.avg_latency_ms:.0f}ms |
| Total Eval Time | {report.total_time_s:.1f}s |
| Samples | {len(report.results)} |

## Per-Question Results

| # | Question | Precision | Recall | Faithfulness | Relevancy | Latency |
|---|----------|-----------|--------|-------------|-----------|---------|
"""
        for i, r in enumerate(report.results, 1):
            q = r.question[:50] + "..." if len(r.question) > 50 else r.question
            md += (
                f"| {i} | {q} | {r.context_precision:.2f} "
                f"| {r.context_recall:.2f} | {r.faithfulness:.2f} "
                f"| {r.answer_relevancy:.2f} | {r.latency_ms:.0f}ms |\n"
            )

        Path(output_path).write_text(md, encoding="utf-8")

    @staticmethod
    def export_json(report: EvaluationReport, output_path: str) -> None:
        """Export evaluation results as JSON."""
        data = {
            "summary": {
                "context_precision": report.avg_context_precision,
                "context_recall": report.avg_context_recall,
                "mrr": report.avg_mrr,
                "ndcg": report.avg_ndcg,
                "faithfulness": report.avg_faithfulness,
                "answer_relevancy": report.avg_answer_relevancy,
                "avg_latency_ms": report.avg_latency_ms,
                "total_time_s": report.total_time_s,
                "num_samples": len(report.results),
            },
            "results": [
                {
                    "question": r.question,
                    "context_precision": r.context_precision,
                    "context_recall": r.context_recall,
                    "mrr": r.mrr,
                    "faithfulness": r.faithfulness,
                    "answer_relevancy": r.answer_relevancy,
                    "latency_ms": r.latency_ms,
                }
                for r in report.results
            ],
        }
        Path(output_path).write_text(json.dumps(data, indent=2), encoding="utf-8")
