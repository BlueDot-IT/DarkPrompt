from __future__ import annotations

import copy
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import typer
from rich.console import Console

from . import __version__
from .adapters.anthropic import AnthropicAdapter
from .adapters.exploitrank import ExploitRankBridge
from .adapters.gemini import GeminiAdapter
from .adapters.huggingface import HuggingFaceAdapter
from .adapters.mistral import MistralAdapter
from .adapters.ollama import OllamaAdapter
from .adapters.openai import OpenAIAdapter
from .evaluator import RuleEvaluator
from .judge import JudgeFeedbackLoop
from .models import EvaluationStatus, ExecutionTrace, TestPack
from .mutator import PromptMutator
from .redactor import RedactionPatternError, RegexRedactor
from .reporter import Reporter
from .runner import PackLoadError, Runner

app = typer.Typer(
    help="DarkPrompt: repeatable adversarial security testing for LLM targets.",
    no_args_is_help=True,
)
console = Console()

ADAPTERS = {
    "openai": OpenAIAdapter,
    "ollama": OllamaAdapter,
    "anthropic": AnthropicAdapter,
    "huggingface": HuggingFaceAdapter,
    "gemini": GeminiAdapter,
    "mistral": MistralAdapter,
}

DEFAULT_MODELS = {
    "openai": "gpt-5.6-luna",
    "anthropic": "claude-sonnet-5",
    "ollama": "mistral",
    "huggingface": "meta-llama/Llama-3.2-3B-Instruct",
    "gemini": "gemini-3.5-flash",
    "mistral": "mistral-medium-latest",
}


def _run_parallel(
    jobs: Iterable[Callable[[], ExecutionTrace]],
    max_workers: int,
) -> List[ExecutionTrace]:
    functions = list(jobs)
    if max_workers <= 1:
        return [function() for function in functions]

    results: list[Optional[ExecutionTrace]] = [None] * len(functions)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(function): index
            for index, function in enumerate(functions)
        }
        for future in as_completed(future_map):
            results[future_map[future]] = future.result()
    return [result for result in results if result is not None]


def _redaction_patterns(values: List[str]) -> List[str]:
    patterns: list[str] = []
    for value in values:
        patterns.extend(item.strip() for item in value.split(",") if item.strip())
    return patterns


def _normalize_provider(value: Optional[str]) -> Optional[str]:
    return value.lower().strip() if value else None


def _validate_provider(value: str, option_name: str) -> None:
    if value not in ADAPTERS:
        raise typer.BadParameter(
            f"Unknown provider {value!r}. Choose from: {', '.join(sorted(ADAPTERS))}.",
            param_hint=option_name,
        )


@app.command()
def run(
    target: str = typer.Option("openai", "--target", "-t", help="Target adapter."),
    pack_dir: Optional[Path] = typer.Option(
        None, "--pack", "-p", help="Directory containing the test pack."
    ),
    out_dir: Path = typer.Option(
        Path("./out"), "--out", "-o", help="Output directory."
    ),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name."),
    redact: List[str] = typer.Option(
        [], "--redact", help="Regex to redact. May be repeated."
    ),
    base_url: Optional[str] = typer.Option(
        None, "--base-url", help="Override the target provider API base URL."
    ),
    judge_target: Optional[str] = typer.Option(
        None,
        "--judge-target",
        help="Separate provider used for semantic_judge assertions.",
    ),
    judge_model: Optional[str] = typer.Option(
        None,
        "--judge-model",
        help="Model used by the semantic judge provider.",
    ),
    judge_base_url: Optional[str] = typer.Option(
        None,
        "--judge-base-url",
        help="Override the semantic judge provider API base URL.",
    ),
    mutate: bool = typer.Option(
        False, "--mutate", help="Apply one deterministic random mutation per case."
    ),
    report_format: str = typer.Option(
        "markdown", "--format", "-f", help="Report format: markdown or json."
    ),
    sensitivity: bool = typer.Option(
        False, "--sensitivity", "-s", help="Run every supported mutation."
    ),
    adaptive: bool = typer.Option(
        False, "--adaptive", "-a", help="Retry refused cases with bounded mutations."
    ),
    exploit_rank: bool = typer.Option(
        False, "--exploit-rank", "--er", help="Load cases from an ExploitRank database."
    ),
    exploit_db: Optional[Path] = typer.Option(
        None, "--exploit-db", help="Path to the ExploitRank SQLite database."
    ),
    exploit_limit: int = typer.Option(
        3, "--exploit-limit", min=1, help="Maximum ExploitRank cases to load."
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed", help="Seed used for reproducible mutations."
    ),
    max_workers: int = typer.Option(
        1, "--max-workers", min=1, help="Maximum concurrent requests."
    ),
    max_retries: int = typer.Option(
        2, "--max-retries", min=0, help="Maximum adaptive retries."
    ),
    fail_on_findings: bool = typer.Option(
        False,
        "--fail-on-findings",
        help="Exit with status 2 when FAIL or PARTIAL findings are present.",
    ),
):
    """Run an adversarial test pack against one target provider."""
    target = _normalize_provider(target) or "openai"
    judge_target = _normalize_provider(judge_target)
    report_format = report_format.lower().strip()
    _validate_provider(target, "--target")
    if judge_target:
        _validate_provider(judge_target, "--judge-target")
    elif judge_model or judge_base_url:
        raise typer.BadParameter(
            "--judge-model and --judge-base-url require --judge-target.",
            param_hint="--judge-target",
        )
    if report_format not in {"markdown", "json"}:
        raise typer.BadParameter("--format must be either markdown or json.")

    model = model or DEFAULT_MODELS[target]
    console.print(
        f"[bold blue]Starting DarkPrompt v{__version__}[/bold blue] "
        f"target=[green]{target}[/green] model=[green]{model}[/green]"
    )

    try:
        redactor = RegexRedactor(_redaction_patterns(redact)) if redact else None
        adapter = ADAPTERS[target](model=model, base_url=base_url)
        judge_adapter = None
        if judge_target:
            resolved_judge_model = judge_model or DEFAULT_MODELS[judge_target]
            judge_adapter = ADAPTERS[judge_target](
                model=resolved_judge_model,
                base_url=judge_base_url,
            )
            console.print(
                f"Semantic judge: provider=[green]{judge_target}[/green] "
                f"model=[green]{resolved_judge_model}[/green]"
            )
        evaluator = RuleEvaluator(judge_adapter=judge_adapter)
        runner = Runner(adapter=adapter, redactor=redactor, evaluator=evaluator)
        mutator = PromptMutator(seed=seed, out_dir=str(out_dir / "media"))

        if pack_dir:
            pack = runner.load_pack(pack_dir)
        else:
            pack = TestPack(
                name="Generated DarkPrompt Run",
                description="Cases generated from configured external sources.",
                version=__version__,
            )

        if exploit_rank:
            bridge = ExploitRankBridge(str(exploit_db) if exploit_db else None)
            for exploit in bridge.get_latest_exploits(limit=exploit_limit):
                pack.cases.append(bridge.generate_case_from_exploit(exploit))

        if not pack.cases:
            raise ValueError(
                "No test cases loaded. Provide --pack or configure --exploit-rank."
            )

        traces: List[ExecutionTrace]
        if sensitivity:
            jobs: list[Callable[[], ExecutionTrace]] = []
            for original_case in pack.cases:
                for mutation_name, mutated_prompt in mutator.named_variants(
                    original_case.prompt
                ):
                    mutated_case = original_case.model_copy(deep=True)
                    mutated_case.prompt = mutated_prompt
                    if mutation_name == "OCR":
                        mutated_case.allow_media_root(mutator.out_dir)
                    suffix = re.sub(r"[^A-Za-z0-9]+", "-", mutation_name).strip("-")
                    mutated_case.id = f"{original_case.id}-{suffix}"

                    def execute(
                        case=mutated_case,
                        name=mutation_name,
                        category=original_case.category,
                    ) -> ExecutionTrace:
                        trace = runner.run_case(case)
                        trace.metadata["mutation"] = name
                        trace.metadata["category"] = category
                        return trace

                    jobs.append(execute)
            traces = _run_parallel(jobs, max_workers)
        elif adaptive:
            judge = JudgeFeedbackLoop(
                target_adapter=adapter,
                mutator=mutator,
                evaluator=evaluator,
            )
            jobs = [
                lambda case=case: runner.finalize_trace(
                    case, judge.run_adaptive_case(case, max_retries=max_retries)
                )
                for case in pack.cases
            ]
            traces = _run_parallel(jobs, max_workers)
        else:
            rng = random.Random(seed)
            cases = []
            for original_case in pack.cases:
                case = original_case.model_copy(deep=True)
                if mutate:
                    variants = mutator.named_variants(case.prompt)[1:]
                    mutation_name, case.prompt = rng.choice(variants)
                    if mutation_name == "OCR":
                        case.allow_media_root(mutator.out_dir)
                else:
                    mutation_name = "Original"
                cases.append((case, mutation_name))

            jobs = []
            for case, mutation_name in cases:

                def execute(
                    current_case=case,
                    name=mutation_name,
                ) -> ExecutionTrace:
                    trace = runner.run_case(current_case)
                    trace.metadata["mutation"] = name
                    return trace

                jobs.append(execute)
            traces = _run_parallel(jobs, max_workers)

        reporter = Reporter(redactor=redactor)
        report_path = (
            reporter.generate_json(pack, traces, out_dir)
            if report_format == "json"
            else reporter.generate_markdown(pack, traces, out_dir)
        )
        console.print(f"[bold green]Audit complete[/bold green]: {report_path}")

        has_findings = any(
            trace.evaluation
            and trace.evaluation.status
            in {EvaluationStatus.FAIL, EvaluationStatus.PARTIAL}
            for trace in traces
        )
        if fail_on_findings and has_findings:
            raise typer.Exit(2)
    except (PackLoadError, RedactionPatternError, FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Configuration error: {exc}[/red]")
        raise typer.Exit(1) from exc


@app.command("validate-pack")
def validate_pack(
    pack_dir: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
):
    """Validate a test pack without sending any provider requests."""
    adapter = OllamaAdapter()
    runner = Runner(adapter=adapter)
    try:
        pack = runner.load_pack(pack_dir)
    except PackLoadError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"[green]Valid pack[/green]: {pack.name} ({len(pack.cases)} cases)")


@app.command("list-providers")
def list_providers():
    """List built-in providers and capabilities."""
    for name, adapter in sorted(ADAPTERS.items()):
        capabilities = adapter.capabilities
        console.print(
            f"{name}: multi_turn={capabilities.multi_turn}, "
            f"images={capabilities.images}, tools={capabilities.tools}"
        )


@app.command()
def version():
    console.print(f"DarkPrompt v{__version__}")


if __name__ == "__main__":
    app()
