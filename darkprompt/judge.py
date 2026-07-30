from __future__ import annotations

from typing import Optional

from .adapter import TargetAdapter
from .evaluator import RuleEvaluator
from .models import EvaluationStatus, ExecutionTrace, TestCase
from .mutator import PromptMutator
from .runner import Runner


class JudgeFeedbackLoop:
    """Bounded adaptive retry loop using deterministic mutation strategies."""

    def __init__(
        self,
        target_adapter: TargetAdapter,
        mutator: Optional[PromptMutator] = None,
        evaluator: Optional[RuleEvaluator] = None,
    ):
        self.target = target_adapter
        self.mutator = mutator or PromptMutator()
        self.evaluator = evaluator or RuleEvaluator()

    def run_adaptive_case(self, case: TestCase, max_retries: int = 2) -> ExecutionTrace:
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")

        aggregate = ExecutionTrace(
            test_case_id=case.id,
            metadata={"adaptive": True, "attempts": []},
        )
        runner = Runner(adapter=self.target, evaluator=self.evaluator)
        variants = self.mutator.named_variants(case.prompt)
        attempt_limit = min(len(variants), max_retries + 1)

        for index, (label, prompt) in enumerate(variants[:attempt_limit]):
            attempt_case = case.model_copy(deep=True)
            attempt_case.prompt = prompt
            if label == "OCR":
                attempt_case.allow_media_root(self.mutator.out_dir)
            attempt = runner.run_case(attempt_case)

            aggregate.prompts.extend(attempt.prompts)
            aggregate.responses.extend(attempt.responses)
            aggregate.tool_calls.extend(attempt.tool_calls)
            aggregate.metadata["attempts"].append(
                {
                    "index": index,
                    "label": label,
                    "status": attempt.evaluation.status.value
                    if attempt.evaluation
                    else EvaluationStatus.INCONCLUSIVE.value,
                    "reason": attempt.evaluation.reason if attempt.evaluation else "",
                    "error": attempt.error.model_dump() if attempt.error else None,
                }
            )

            if attempt.error:
                aggregate.error = attempt.error
                break

            if attempt.evaluation and attempt.evaluation.status == EvaluationStatus.FAIL:
                break

        aggregate.metadata["retries"] = max(0, len(aggregate.metadata["attempts"]) - 1)
        aggregate.evaluation = self.evaluator.evaluate(case, aggregate)
        return aggregate
