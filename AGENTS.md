# AGENTS.md — τ-bench / τ³ StageGate Research Fork

> Instructions for AI coding agents working on this fork of the τ-bench codebase.
>
> This file preserves the core upstream τ-bench operating instructions and adds a StageGate-specific research overlay. Treat it as a map. For deep StageGate context, read `docs/stagegate/` and the active ExecPlans in `docs/stagegate/exec-plans/active/`.

## Project overview

τ-bench is a simulation framework for evaluating conversational customer-service agents. It supports text and voice interactions in half-duplex, turn-based mode and full-duplex, simultaneous/streaming mode.

Domains include:

- `mock`
- `airline`
- `retail`
- `telecom`
- `banking_knowledge`

This fork is used to test a custom realtime voice-agent scaffold named **StageGate** on τ³ / τ-Voice.

StageGate conditions:

- **V0 baseline**: official OpenAI audio-native GPT-Realtime-2 scaffold.
- **V1 StageOnly**: baseline plus `advance_stage` staged workflow packets.
- **V2 StageGate**: StageOnly plus typed entity ledger and pre-write validator.

The final research claim is the paired delta between conditions on the same tasks, not a raw leaderboard brag.

## StageGate benchmark-validity rules

Do **not** modify:

- task files
- evaluator logic
- user simulator prompts or hidden user state
- domain policy files
- domain tools
- scoring logic

StageGate may use only:

- domain name
- public domain policy and public tool schemas
- conversation history visible to the agent
- agent-visible model tool calls
- official domain tool outputs
- transcript/audio events available to the agent path
- server-side state derived from the above

StageGate must never use:

- hidden task objective
- task ID as a rule selector
- expected final database state
- evaluator output
- reward/failure signal
- user simulator private plan
- clean simulator text unavailable to the voice agent
- rerunning only failed tasks for reported final numbers

If a proposed change could affect benchmark validity, stop and ask for human review before proceeding.

## StageGate docs to read before implementation

Read these before StageGate implementation work:

- `docs/stagegate/00-project-brief.md`
- `docs/stagegate/01-research-thesis.md`
- `docs/stagegate/02-architecture.md`
- `docs/stagegate/03-experiment-protocol.md`
- `docs/stagegate/04-modal-execution.md`
- `docs/stagegate/05-tracing-and-visualization.md`
- `docs/stagegate/06-validity-and-leakage-rules.md`
- `docs/stagegate/07-sierra-submission.md`
- `docs/stagegate/08-codex-operating-model.md`
- `.agent/PLANS.md`

For complex features or refactors, use an ExecPlan from `docs/stagegate/exec-plans/active/` and the template in `.agent/PLANS.md`.

## Setup

```bash
uv sync                        # core only: airline, retail, telecom, mock
uv sync --extra voice          # + voice/audio-native features
uv sync --extra knowledge      # + banking_knowledge domain and retrieval pipeline
uv sync --extra gym            # + gymnasium RL interface
uv sync --extra dev            # + pytest, ruff, pre-commit; required for committing
uv sync --extra experiments    # + plotting libs for src/experiments/
uv sync --all-extras           # everything
uv run tau2 check-data         # verify installation
```

For StageGate development, default to:

```bash
uv sync --extra voice --extra dev --extra experiments
uv run tau2 check-data
uv run tau2 intro
```

`langfuse` and `redis` are not declared dependencies but may be needed if `USE_LANGFUSE=True` or if `LLM_CACHE_ENABLED=True` with redis cache type in `config.py`. Install manually only when needed:

```bash
uv pip install langfuse redis
```

Environment variables: copy `.env.example` to `.env` and set API keys. The codebase uses LiteLLM for LLM provider abstraction where applicable.

Required keys depend on the task:

- `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` — LLM-based agents and user simulators
- `ELEVENLABS_API_KEY` — voice synthesis
- `DEEPGRAM_API_KEY` — voice transcription

Never commit `.env` or raw secrets. For final StageGate runs, benchmark API keys should live in Modal secrets, not in ordinary Codex context.

## Common commands

| Command | What it does | Required install |
|---|---|---|
| `make test` | Run core tests; skips voice, streaming, gym, banking_knowledge | `uv sync --extra dev` |
| `make test-voice` | Run voice + streaming tests | `uv sync --extra voice --extra dev` |
| `make test-knowledge` | Run banking_knowledge tests | `uv sync --extra knowledge --extra dev` |
| `make test-gym` | Run gymnasium tests | `uv sync --extra gym --extra dev` |
| `make test-all` | Run all tests | `uv sync --all-extras` |
| `make lint` | Lint with ruff | `uv sync --extra dev` |
| `make format` | Format with ruff | `uv sync --extra dev` |
| `make lint-fix` | Lint and auto-fix | `uv sync --extra dev` |
| `make check-all` | Run lint + format checks; same as pre-commit hook | `uv sync --extra dev` |
| `make clean` | Remove venv, caches, build artifacts | — |
| `make env-cli` | Interactive environment CLI for testing domain tools | — |

`make test` is the safe default. Always run `make check-all` before committing. A pre-commit hook enforces this.

## Running evaluations

Text half-duplex example:

```bash
tau2 run \
  --domain airline \
  --agent-llm gpt-4.1 \
  --user-llm gpt-4.1 \
  --num-trials 1 \
  --num-tasks 5
```

Voice full-duplex / audio-native smoke example:

```bash
tau2 run \
  --domain retail \
  --audio-native \
  --audio-native-provider openai \
  --audio-native-model gpt-realtime-2 \
  --reasoning-effort high \
  --speech-complexity control \
  --tick-duration 0.2 \
  --max-steps-seconds 300 \
  --max-concurrency 1 \
  --seed 300 \
  --num-tasks 1 \
  --verbose-logs \
  --audio-taps \
  --save-to smoke_stagegate_retail_control
```

Knowledge-domain example:

```bash
tau2 run \
  --domain banking_knowledge \
  --retrieval-config qwen_embeddings \
  --agent-llm gpt-4.1 \
  --user-llm gpt-4.1 \
  --num-tasks 5
```

Results go to `data/simulations/`. Use `tau2 view` to browse them:

```bash
tau2 view --dir data/simulations/smoke_stagegate_retail_control --expanded-ticks
```

## StageGate final-run constants

Final benchmark runs must use:

- domains: `retail`, `airline`, `telecom`
- conditions: `baseline`, `stage_only`, `stagegate`
- model: `gpt-realtime-2`
- reasoning effort: `high`
- speech complexity: `regular`
- tick duration: `0.2`
- max steps seconds: `1200`
- max concurrency: `1`
- seed: `300`
- no `--num-tasks`
- no `--task-ids`

Do not report final claims from control-speech runs, filtered task runs, failed-task-only reruns, or runs with different constants across conditions.

## Architecture

```text
src/tau2/
├── agent/             # Agent implementations: half-duplex and full-duplex
├── api_service/       # FastAPI-based API service
├── config.py          # Central configuration; single source of truth for defaults
├── cli.py             # CLI entry point: tau2 command
├── data_model/        # Pydantic data models: messages, trajectories, etc.
├── domains/           # Domain definitions: airline, mock, retail, telecom, banking_knowledge
├── environment/       # Environment, DB, server, toolkit base classes
├── evaluator/         # Task evaluation logic
├── gym/               # Gymnasium-compatible RL interface
├── knowledge/         # Knowledge retrieval pipeline
├── metrics/           # Metrics computation
├── orchestrator/      # Simulation orchestrators: half-duplex and full-duplex
├── registry.py        # Global registry for agents, domains, tasks, users
├── runner/            # Simulation runner: batch execution, checkpointing, builders
├── scripts/           # CLI command implementations
├── user/              # User simulator implementations
├── utils/             # Shared utilities
└── voice/             # Voice synthesis, transcription, audio-native providers
    └── audio_native/  # Realtime voice providers: openai, gemini, nova, xai, deepgram, qwen, livekit
```

Top-level directories:

- `data/` — domain data and simulation outputs
- `tests/` — pytest test suite
- `scripts/` — standalone utility scripts
- `src/experiments/` — self-contained research/experimental code
- `docs/` — user-facing documentation

## Key patterns

### Registry system

All agents, domains, tasks, and user simulators are registered in `src/tau2/registry.py`. To add a new component, register it there:

```python
registry.register_agent_factory(create_my_agent, "my_agent")
registry.register_domain(get_environment, "my_domain")
registry.register_tasks(get_tasks, "my_domain", get_task_splits=get_tasks_split)
```

For StageGate, prefer an env-gated wrapper or localized OpenAI audio-native scaffold change unless an ExecPlan explicitly chooses a new registered component. If a new agent/provider/factory is added, register it and document why.

### Agent architecture

Two base classes are determined by communication mode:

| Mode | Base class | Key method | Used by |
|---|---|---|---|
| Half-duplex, turn-based | `HalfDuplexAgent` | `generate_next_message()` | `LLMAgent` |
| Full-duplex, streaming | `FullDuplexAgent` | `get_next_chunk()` | `DiscreteTimeAudioNativeAgent` |

Both share constructor signature:

```python
__init__(self, tools: list[Tool], domain_policy: str)
```

For LLM-based agents, mix in `LLMConfigMixin` to add `llm` and `llm_args` parameters.

### Domain structure

Each standard domain in `src/tau2/domains/<name>/` contains:

- `data_model.py` — DB subclass with domain data models
- `tools.py` — `ToolKitBase` subclass with domain tools
- `environment.py` — `get_environment()`, `get_tasks()`, `get_tasks_split()`
- `user_tools.py` — optional user-facing tools
- `utils.py` — data paths and helpers

Domain data lives in `data/tau2/domains/<name>/`, including files such as `tasks.json`, `policy.md`, and `db.json` / `db.toml`.

The `banking_knowledge` domain extends this pattern with `retrieval.py`, `retrieval_mixins.py`, `retrieval_toolkits.py`, `db_query.py`, dynamic tools and policy that vary by `--retrieval-config`, and the separate `knowledge/` retrieval pipeline module. Its data directory also includes `documents/`, `prompts/`, and `tasks/` subdirectories. See `src/tau2/knowledge/README.md` for details.

### Orchestrators

- `Orchestrator` — half-duplex, turn-based, synchronous tool execution
- `FullDuplexOrchestrator` — full-duplex, tick-based, simultaneous agent/user activity

### Audio-native providers

Each audio-native provider has its own WebSocket protocol and event format. Always verify against provider documentation and existing provider tests before changing provider behavior. If `.cursor/rules/audio-native-provider.md` exists, read it before provider-level implementation.

StageGate should stay localized to the OpenAI audio-native agent/scaffold path unless the active ExecPlan requires broader changes.

## StageGate implementation expectations

When implementing StageGate:

1. Read enough code before editing. Use `rg`, `rg --files`, and targeted file reads.
2. Make additive, testable changes first.
3. Keep changes localized to the OpenAI audio-native agent/scaffold path unless the ExecPlan says otherwise.
4. Add or update tests before any large benchmark run.
5. Update the active ExecPlan after every meaningful step.
6. Record design decisions in the ExecPlan Decision Log.
7. Record surprising repo behavior or benchmark behavior in Surprises & Discoveries.
8. Prefer explicit failure and logging over silent fallback.
9. Keep baseline behavior unchanged when StageGate env flags are disabled.
10. Keep JSONL trace events stable and versioned with `schema_version`.

Expected StageGate implementation pieces:

- `advance_stage` orchestration tool
- compact stage packet generator
- typed entity ledger
- pre-write validator
- JSONL trace writer controlled by env var
- leakage tests
- Modal execution support
- analysis scripts and trace viewer

## Testing

Tests are split into tiers matching optional dependency groups:

```bash
make test
make test-voice
make test-knowledge
make test-gym
make test-all
```

Domain-specific examples:

```bash
pytest tests/test_domains/test_<domain_name>
pytest tests/test_agent.py
pytest -m "not full_duplex_integration"
```

Test layout mirrors source:

- `tests/test_domains/` — per-domain tool and user-tool tests, except `test_banking_knowledge/` which requires the `knowledge` extra
- `tests/test_streaming/` — streaming/full-duplex tests; requires `voice` extra
- `tests/test_voice/` — audio-native provider tests; requires `voice` extra; individual providers gated by `{PROVIDER}_TEST_ENABLED=1`
- `tests/test_gym/` — gymnasium RL interface tests; requires `gym` extra

For StageGate, add or update tests for:

- stage packet schema
- entity ledger state transitions
- validator allow/block decisions
- no hidden task/evaluator/simulator state access
- trace JSONL event format
- smoke path through the OpenAI audio-native adapter
- baseline path unchanged when StageGate is disabled

Suggested validation sequence, adjusted after Codex maps exact test paths:

```bash
make test
make test-voice
make check-all
```

When live provider tests require API keys, run the narrowest non-live tests first and record what remains unverified.

## Code style

- Formatter/linter: Ruff, configured in `pyproject.toml`
- Line length: 88 characters
- Python: `>=3.12`, `<3.14`
- Type hints: encouraged, especially for public APIs
- Docstrings: required for public APIs and complex functions
- Import sorting: handled by Ruff
- Models: use Pydantic `BaseModel` for data classes where appropriate

Ruff rules include `E4`, `E7`, `E9`, `F`, and `I`, with `E501` and `F541` ignored.

StageGate-specific style:

- Prefer typed dataclasses or Pydantic models for StageGate state.
- Do not use `Any` to avoid typing important state.
- Do not add broad `try`/`except` blocks that swallow errors.
- Keep stage packets compact and easy to inspect.
- Keep implementation transparent enough for a skeptical benchmark maintainer to audit.
- Do not add hidden retries or task-specific branches.

## Commit conventions

Use conventional commit-style messages:

```text
feat: add memory system to agent base class
fix: resolve environment tool timeout issues
docs: update domain contribution guidelines
test: add integration tests for retail domain
```

For StageGate, examples:

```text
feat: add StageOnly advance_stage scaffold
test: add StageGate leakage checks
feat: add StageGate entity ledger
feat: add pre-write validator for StageGate
```

## Things to watch out for

- `.env`: never commit it. It contains API keys. Use `.env.example` as reference.
- `data/`: contains domain data and simulation outputs. Be careful modifying JSON/TOML data files. StageGate should not modify domain task, policy, DB seed, or scoring data.
- `config.py`: single source of truth for default configuration values. Import constants from here rather than defining local duplicates.
- `registry.py`: all new agents, domains, and user simulators must be registered here to be usable via CLI.
- Audio-native providers: each has its own WebSocket protocol and event format. Verify against provider documentation and tests.
- Task splits: `base` is the default for evaluation. `train`/`test` splits are for RL experiments.
- Pre-commit hook: runs `make check-all`. Fix issues before committing.
- Notebooks: excluded from Ruff via pyproject configuration.
- `banking_knowledge`: uses `--retrieval-config` to specify knowledge access. If omitted, it defaults to `bm25`, which is offline and needs no API keys.

Offline `banking_knowledge` configs include:

- `no_knowledge`
- `full_kb`
- `golden_retrieval`
- `bm25`
- `bm25_grep`
- `grep_only`

Configs requiring external keys:

- `openai_embeddings*` requires `OPENAI_API_KEY`
- `qwen_embeddings*` requires `OPENROUTER_API_KEY`
- `*_reranker` additionally requires `OPENAI_API_KEY` for the LLM reranker
- `terminal_use*` requires `sandbox-runtime` via `npm install -g @anthropic-ai/sandbox-runtime@0.0.23`

Embedding cache lives in `data/.embeddings_cache`, which is gitignored. See `src/tau2/knowledge/README.md` for full details.

## Review expectations

Before claiming a milestone is complete, Codex must report:

- files changed
- tests run
- exact command results
- what remains unverified
- any changes to benchmark-validity assumptions
- whether baseline behavior is unchanged
- whether secrets or local-only paths were introduced

If the work affects benchmark validity, stop and ask for human review before proceeding.
