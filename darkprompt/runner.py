from __future__ import annotations

import copy
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import ValidationError

from .adapter import TargetAdapter
from .evaluator import RuleEvaluator
from .models import ExecutionTrace, TestCase, TestPack
from .redactor import RegexRedactor


class PackLoadError(ValueError):
    pass


class Runner:
    def __init__(
        self,
        adapter: TargetAdapter,
        redactor: Optional[RegexRedactor] = None,
        evaluator: Optional[RuleEvaluator] = None,
    ):
        self.adapter = adapter
        self.redactor = redactor
        self.evaluator = evaluator or RuleEvaluator()

    def load_pack(self, pack_dir: Path) -> TestPack:
        pack_root = pack_dir.expanduser().resolve()
        pack_yaml = pack_dir / "pack.yaml"
        if not pack_yaml.is_file():
            raise PackLoadError(f"Missing pack file: {pack_yaml}")

        try:
            with pack_yaml.open("r", encoding="utf-8") as handle:
                pack_data = yaml.safe_load(handle) or {}
            pack = TestPack(**pack_data)
        except (OSError, yaml.YAMLError, ValidationError) as exc:
            raise PackLoadError(f"Invalid pack definition {pack_yaml}: {exc}") from exc

        seen_ids = {case.id for case in pack.cases}
        cases_dir = pack_dir / "cases"
        if cases_dir.exists():
            for case_file in sorted(cases_dir.glob("*.yaml")):
                try:
                    with case_file.open("r", encoding="utf-8") as handle:
                        raw = yaml.safe_load(handle)
                    case_rows = raw if isinstance(raw, list) else [raw]
                    for row in case_rows:
                        case = TestCase(**(row or {}))
                        if case.id in seen_ids:
                            raise PackLoadError(f"Duplicate test case id {case.id!r} in {case_file}")
                        seen_ids.add(case.id)
                        pack.cases.append(case)
                except PackLoadError:
                    raise
                except (OSError, yaml.YAMLError, ValidationError, TypeError) as exc:
                    raise PackLoadError(f"Invalid case file {case_file}: {exc}") from exc

        for case in pack.cases:
            case.allow_media_root(pack_root)
        return pack

    def finalize_trace(self, case: TestCase, trace: ExecutionTrace) -> ExecutionTrace:
        trace.metadata.setdefault("category", case.category)
        trace.evaluation = self.evaluator.evaluate(case, trace)
        if self.redactor:
            trace = self.redactor.redact(trace)
        return trace

    def run_case(self, case: TestCase) -> ExecutionTrace:
        context = {"history": []}
        trace = ExecutionTrace(test_case_id=case.id)

        prompts = [case.prompt, *case.chain]
        for prompt in prompts:
            current_case = copy.deepcopy(case)
            current_case.prompt = prompt
            turn = self.adapter.execute(current_case, context)

            trace.prompts.append(prompt)
            if turn.error:
                trace.error = turn.error
                trace.metadata.update(turn.metadata)
                break

            response = turn.responses[0] if turn.responses else ""
            trace.responses.append(response)
            trace.tool_calls.extend(turn.tool_calls)
            trace.metadata.update(turn.metadata)

            context["history"].append({"role": "user", "content": prompt})
            context["history"].append({"role": "assistant", "content": response})

        return self.finalize_trace(case, trace)

    def run(self, pack: TestPack) -> List[ExecutionTrace]:
        return [self.run_case(case) for case in pack.cases]
