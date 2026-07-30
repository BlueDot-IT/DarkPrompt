# Changelog

All notable changes to DarkPrompt are documented here.

## Unreleased

### Security

- Confined media payloads to approved roots, image MIME types, and a 10 MiB limit.
- Made raw report evidence operator-opt-in and expanded recursive redaction coverage.
- Made incomplete evaluations fail `--fail-on-findings` unless explicitly allowed.
- Added bounded regex evaluation and hardened refusal heuristics against mixed refusal/compliance.
- Sanitized HTTP status errors before persistence.
- Added a locked CI dependency graph and corrected organization repository links.

## 1.2.0

### Added

- Typed assertions for refusal, content, regex, JSON Schema, tool-call, and semantic checks.
- Assertion scopes for final response, any response, all responses, and a specific one-based turn.
- Weighted assertion scores and per-assertion confidence, reasons, and evidence.
- Optional isolated semantic judge provider through `--judge-target`, `--judge-model`, and `--judge-base-url`.
- Draft 2020-12 JSON Schema validation during pack loading and response evaluation.
- Detailed assertion tables and aggregate assertion metrics in Markdown and JSON reports.

### Changed

- Assertions now take precedence over legacy free-form `expected_outcome` evaluation.
- Semantic judge failures and malformed judge output are marked inconclusive instead of being guessed.
- Redaction now covers assertion-level reasons and evidence.
- JSON report schema version increased to 1.2.

### Compatibility

- Existing test packs using only `expected_outcome` continue to work.

## 1.1.0

### Added

- Structured PASS, FAIL, PARTIAL, ERROR, SKIPPED, and INCONCLUSIVE evaluation statuses.
- Structured provider errors that no longer count as model responses.
- Real multi-turn history propagation across built-in chat adapters.
- Actual image payload construction for OpenAI, Anthropic, Gemini, Mistral, and Ollama.
- Deterministic mutation seeds and named mutation metadata.
- Concurrent sensitivity execution with a configurable worker limit.
- `validate-pack`, `list-providers`, `--fail-on-findings`, and configurable ExploitRank database options.
- Python 3.9, 3.11, and 3.13 CI coverage, package builds, static checks, and coverage enforcement.

### Changed

- Replaced refusal-count reporting with explicit per-trace evaluation results.
- Switched Ollama to the chat endpoint so conversation history is preserved.
- Made Hugging Face use its OpenAI-compatible chat router.
- Made package version metadata derive from `darkprompt.__version__`.
- Improved pack validation, redaction validation, Markdown escaping, and JSON report metadata.

### Fixed

- Multi-turn cases no longer send isolated prompts.
- OCR mutations are no longer passed to capable providers as plain path markers.
- Empty reports no longer divide by zero.
- Missing credentials and HTTP failures no longer appear as exposed-model findings.
- ExploitRank no longer depends on a developer-specific absolute filesystem path.
