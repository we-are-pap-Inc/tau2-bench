# ExecPlan 01 - StageGate Implementation

## Purpose

Implement StageGate for the OpenAI audio-native scaffold without changing
benchmark content or scoring. V1 StageOnly provides the `advance_stage`
orchestration tool, deterministic stage packets, and JSONL tracing. V2
StageGate adds a typed entity ledger for retail, airline, and telecom that
updates only from agent-visible function-call arguments and official domain
tool outputs. It also adds a pre-write validator that blocks unsafe
side-effecting domain tool calls before `Environment.get_response()` can mutate
domain state, and returns corrective `stagegate.stage_packet.v1` packets to the
model.

The implementation does not edit evaluator logic, task files, user-simulator
state, scoring, domain policy files, or domain tool definitions.

## Benchmark Validity Constraints

Do not modify task files, evaluator logic, user simulator prompts or hidden
state, domain policy files, domain tools, scoring logic, or database
final-state comparison logic.

StageGate may use only domain name, public domain policy and tool schemas,
conversation-visible tool calls, official domain tool outputs, and server-side
state derived from the agent-visible path.

StageGate must not read hidden task objectives, task IDs as rule selectors,
expected final database state, evaluator output, reward or failure signals,
user simulator private plans, or clean simulator text unavailable to the voice
agent.

## Current Runtime Map

- OpenAI realtime session setup:
  `src/tau2/voice/audio_native/openai/discrete_time_adapter.py`
  `DiscreteTimeOpenAIAdapter.connect()` and
  `DiscreteTimeOpenAIAdapter._async_connect()` call
  `OpenAIRealtimeProvider.connect()` and
  `OpenAIRealtimeProvider.configure_session()`.
- OpenAI tool registration:
  `src/tau2/voice/audio_native/openai/provider.py`
  `OpenAIRealtimeProvider.configure_session()` includes `"tools"` in the
  `session.update` payload, and
  `OpenAIRealtimeProvider._format_tools_for_api()` converts
  `Tool.openai_schema` into Realtime function schema objects.
- Agent session insertion point:
  `src/tau2/agent/discrete_time_audio_native_agent.py`
  `DiscreteTimeAudioNativeAgent.__init__()` creates
  `StageGateController.from_env()`;
  `DiscreteTimeAudioNativeAgent._build_system_prompt()` appends the prompt
  addendum only when enabled; and
  `DiscreteTimeAudioNativeAgent.get_init_state()` passes
  `StageGateController.session_tools()` to `adapter.connect()`.
- Tool-call receipt:
  `src/tau2/voice/audio_native/openai/discrete_time_adapter.py`
  `DiscreteTimeOpenAIAdapter._process_event()` handles
  `FunctionCallArgumentsDoneEvent`, parses arguments, creates `ToolCall`, and
  appends it to `TickResult.tool_calls`.
- Tool-call dispatch:
  `src/tau2/orchestrator/full_duplex_orchestrator.py`
  `FullDuplexOrchestrator._process_participant_turn()` routes agent tool
  calls through `FullDuplexOrchestrator._execute_stagegate_tool_call()` when a
  controller is active.
- Domain tool execution:
  `src/tau2/environment/environment.py` `Environment.get_response()` calls
  `make_tool_call()`, which dispatches to
  `src/tau2/environment/toolkit.py` `ToolKitBase.use_tool()`.
- Tool result return:
  `src/tau2/agent/discrete_time_audio_native_agent.py`
  `DiscreteTimeAudioNativeAgent._handle_tool_result()` queues tool results;
  `src/tau2/voice/audio_native/openai/discrete_time_adapter.py`
  `DiscreteTimeOpenAIAdapter._flush_pending_tool_results()` calls
  `src/tau2/voice/audio_native/openai/provider.py`
  `OpenAIRealtimeProvider.send_tool_result()`.
- Trajectory and run-output writing:
  `src/tau2/orchestrator/full_duplex_orchestrator.py`
  `FullDuplexOrchestrator._finalize()` builds `SimulationRun`;
  `src/tau2/runner/batch.py` `run_single_task()` writes task logs and audio;
  `src/tau2/runner/checkpoint.py` `create_checkpoint_fns()` writes
  `results.json` and per-simulation JSON.
- JSONL trace insertion point:
  `src/tau2/voice/audio_native/openai/stagegate/controller.py`
  `StageGateController._trace()` is the single StageOnly facade over
  `src/tau2/voice/audio_native/openai/stagegate/trace.py`
  `JsonlTraceWriter.write()`.
- Relevant existing tests:
  `tests/test_streaming/test_stagegate.py`,
  `tests/test_streaming/test_discrete_time_audio_native_agent.py`,
  `tests/test_streaming/test_tool_call_flow.py`,
  `tests/test_voice/test_audio_native/test_provider_suite.py`, and
  `tests/test_environment.py`.

## Implemented StageGate Design

- `src/tau2/voice/audio_native/openai/stagegate/stage_schema.py`
  defines `StagePacket` and `TraceEvent` schemas. `StagePacket` now includes
  `ambiguous_facts` alongside `known_facts`, `missing_facts`, and `ask_next`.
- `src/tau2/voice/audio_native/openai/stagegate/trace.py`
  provides `JsonlTraceWriter.from_env()` controlled by `TAU2_TRACE_JSONL`.
- `src/tau2/voice/audio_native/openai/stagegate/orchestrator.py`
  provides `StagePacketOrchestrator`, which builds deterministic packets from
  `advance_stage` arguments, public tool names, and, for StageGate only, the
  typed entity ledger.
- `src/tau2/voice/audio_native/openai/stagegate/ledger.py`
  defines `LedgerStatus`, `LedgerEvidence`, `LedgerSlot`, and `EntityLedger`,
  with domain slots for retail, airline, and telecom.
- `src/tau2/voice/audio_native/openai/stagegate/validator.py`
  provides `ValidatorDecision` and `PreWriteValidator`, classifies side-effecting
  tools from reviewed domain tool names plus public tool names/descriptions,
  tracks visible read inspections and confirmations, validates write/action
  tool calls, and builds corrective `StagePacket` blocks without calling domain
  tools.
- `src/tau2/voice/audio_native/openai/stagegate/controller.py`
  provides `StageGateController` with env parsing, prompt addendum,
  `session_tools()`, `is_advance_stage()`, `handle_advance_stage()`, ledger
  ownership when `condition="stagegate"`, ledger updates from visible events,
  validator ownership when `condition="stagegate"`, visible message recording,
  and `ledger_update` / validator trace events.
- `src/tau2/agent/discrete_time_audio_native_agent.py` wires the controller
  into OpenAI audio-native session setup without changing baseline behavior.
- `src/tau2/orchestrator/full_duplex_orchestrator.py` records visible
  participant text for the validator, intercepts `advance_stage`, and validates
  non-`advance_stage` domain tool calls before domain execution. Allowed calls
  continue through `Environment.get_response()` unchanged; blocked calls return
  an error `ToolMessage` containing a corrective stage packet and do not touch
  domain state.
- `src/tau2/voice/audio_native/openai/__init__.py` lazy-loads provider and
  adapter classes so importing the StageOnly package does not make core
  τ-bench imports require voice-only dependencies.

The active package exports the ledger and validator models. `stage_only`
remains validator-free; the pre-write validator is active only for
`TAU2_STAGEGATE_CONDITION=stagegate`.

## Implementation Progress

- [x] Keep StageOnly localized to
  `src/tau2/voice/audio_native/openai/stagegate/` plus the existing
  audio-native agent and full-duplex orchestrator insertion points.
- [x] Add `advance_stage` session tool only when
  `TAU2_STAGEGATE_CONDITION` is `stage_only` or `stagegate`.
- [x] Treat `stagegate` as StageOnly-compatible for this step.
- [x] Generate `stagegate.stage_packet.v1` tool results without calling
  `Environment.get_response()`.
- [x] Emit canonical `stagegate.trace.v1` JSONL events when
  `TAU2_TRACE_JSONL` is set: `run_start`, `run_end`,
  `model_function_call`, `domain_tool_call`, `domain_tool_result`,
  `advance_stage_call`, and `stage_packet_returned`. Runtime StageGate traces
  do not include evaluator-derived outcome, reward, or pass/fail fields.
- [x] Leave normal domain tools on the existing execution path.
- [x] Remove active V2 ledger and validator behavior from this scope.
- [x] Preserve batch `trial` metadata on trace rows by attaching the trial to
  the orchestrator before simulation execution.
- [x] Emit `run_end` trace rows for full-duplex run exceptions so started trace
  runs do not remain open-ended after retryable failures.
- [x] Add a typed StageGate entity ledger with required statuses, evidence
  metadata, and retail/airline/telecom slots.
- [x] Keep `stage_only` ledger-free while `stagegate` owns an `EntityLedger`.
- [x] Update the ledger from model tool arguments as `heard_not_confirmed` and
  successful official domain tool outputs as `tool_verified`.
- [x] Emit `ledger_update` JSONL trace rows with source, tool, tick, evidence,
  and per-slot `ledger_delta`.
- [x] Feed StageGate packets from ledger `known_facts`, `missing_facts`,
  `ambiguous_facts`, and `ask_next`; keep StageOnly packet behavior compatible.
- [x] Add `PreWriteValidator` with `ValidatorDecision`, side-effecting tool
  classification, read-tool pass-through, exact identifier checks, policy-state
  inspection checks, action summary checks, user confirmation checks, and
  required argument checks.
- [x] Return corrective `stagegate.stage_packet.v1` tool messages when blocking
  unsafe write/action calls.
- [x] Emit `validator_check`, `validator_allow`, and `validator_block` trace
  rows.
- [x] Keep blocked calls from invoking `Environment.get_response()` or mutating
  domain toolkit state.
- [x] Restore the full-duplex visibility boundary: newly emitted user chunks are
  not recorded into StageGate validator state until the next agent turn receives
  them as `incoming_for_agent`.
- [x] Move evaluator-derived final outcome rows out of StageGate runtime and
  into `scripts/stagegate_posthoc_outcomes.py`, which writes
  `oracle_analysis.jsonl` after result files exist.
- [x] Add `scripts/stagegate_prohibited_diff_guard.py` to prevent StageGate
  branches from modifying benchmark-controlled task, domain, evaluator, user,
  metrics, or scoring files.

## Tests

Focused tests in `tests/test_streaming/test_stagegate.py` cover:

- baseline/unset environment leaves prompt and tool list unchanged;
- `TAU2_STAGEGATE_CONDITION=stage_only` adds exactly one `advance_stage` tool;
- `TAU2_STAGEGATE_CONDITION=stagegate` enables the entity ledger and
  pre-write validator;
- `advance_stage` returns a `stagegate.stage_packet.v1` packet;
- `advance_stage` does not call `Environment.get_response()` and does not
  mutate a fake domain toolkit counter;
- normal non-`advance_stage` domain tools still call
  `Environment.get_response()` unchanged;
- `TAU2_TRACE_JSONL` writes canonical runtime `stagegate.trace.v1` rows for
  StageOnly packets, model function calls, and domain tool calls/results;
- entity ledger domain-slot initialization and JSON serialization;
- model tool-argument updates with source/event/tick evidence;
- successful official tool-result verification;
- errored tool-result no-op behavior;
- contradiction handling and `ambiguous_facts` in StageGate packets;
- `ledger_update` trace event shape and metadata.
- regression coverage that retail product names and telecom plan names do not
  get misclassified as customer names.
- validator blocking never mutates domain state;
- missing user confirmation blocks a write and returns a corrective packet;
- confirmed exact identifiers allow a read and then a policy-valid write;
- action summaries must mention the exact mutable identifier as a distinct
  value before the user confirmation can satisfy the write gate;
- same-tick user confirmations do not satisfy validator checks before delivery
  to the agent;
- next-tick user confirmations become usable only after delivery as
  `incoming_for_agent`;
- StageGate runtime package code does not read evaluator-derived reward fields;
- prohibited-path guard classification for benchmark-controlled files.
- action summaries must state a consequence, not only an intended action;
- read-only tools are not overblocked;
- validator leakage guards show no task objective, expected final DB,
  user-simulator private state, evaluator result, or task-ID routing inputs.

## Validation Evidence

- `uv run pytest tests/test_streaming/test_stagegate.py -q` initially could
  not collect in the freshly created core-only environment. Direct import
  investigation showed the active package path then failed on missing voice
  dependency `websockets`.
- `uv sync --extra voice --extra dev` completed successfully and installed the
  voice/dev dependencies needed for the audio-native import path.
- `uv run python - <<'PY' ... import tau2 ... PY` with a meta-path guard that
  raises `ModuleNotFoundError` for `websockets` succeeded and printed
  `import tau2 ok without importing websockets`, confirming the StageOnly
  import path does not eagerly require the voice extra.
- `uv run pytest tests/test_streaming/test_stagegate.py -q`
  result: `6 passed, 2 warnings in 0.02s`.
- `uv run pytest tests/test_streaming/test_discrete_time_audio_native_agent.py -q`
  result: `33 passed, 2 warnings in 0.06s`.
- `uv sync --extra voice --extra dev --extra experiments`
  result: completed successfully after the fresh worktree environment resolved
  `uv run pytest` to a global Python 3.13 pytest before the dev extra was
  installed.
- `uv run pytest tests/test_streaming/test_stagegate.py tests/test_stagegate_trace_viewer.py -q`
  result: `12 passed, 2 warnings in 0.04s`.
- `uv run pytest tests/test_streaming/test_stagegate.py tests/test_stagegate_trace_viewer.py -q`
  after review fixes result: `15 passed, 2 warnings in 0.04s`.
- `make test`
  result: `164 passed, 17 failed, 1 xfailed, 14 warnings`; failures are
  LLM-backed core tests failing with `litellm.AuthenticationError` because
  `OPENAI_API_KEY` is not set in this environment.
- `make test-voice`
  result: `255 passed, 3 skipped, 83 deselected, 2 warnings in 0.72s`.
- `make test-voice` after review fixes
  result: `258 passed, 3 skipped, 83 deselected, 2 warnings in 0.49s`.
- `make check-all`
  result: Ruff check passed and Ruff format reformatted 3 files.
- `make check-all` after review fixes
  result: Ruff check passed and Ruff format left 320 files unchanged.
- `git diff --check`
  result: passed with no whitespace errors.
- `uv run ruff check .`
  result: `All checks passed!`.
- 2026-05-09: `uv sync --extra voice --extra dev`
  result: completed successfully in this fresh worktree and installed the
  voice/dev dependency set.
- 2026-05-09: `uv run pytest tests/test_streaming/test_stagegate.py tests/test_stagegate_trace_viewer.py -q`
  result before formatting: `22 passed, 2 warnings in 0.12s`.
- 2026-05-09: `make test-voice`
  result: `265 passed, 3 skipped, 83 deselected, 2 warnings in 0.69s`.
- 2026-05-09: `make check-all`
  result: Ruff check passed and Ruff format reformatted 2 files.
- 2026-05-09: `uv run pytest tests/test_streaming/test_stagegate.py tests/test_stagegate_trace_viewer.py -q`
  result after formatting: `22 passed, 2 warnings in 0.05s`.
- 2026-05-09: `make test`
  result: `164 passed, 17 failed, 1 xfailed, 14 warnings`; failures are
  credential-dependent LLM tests failing with `litellm.AuthenticationError`
  because `OPENAI_API_KEY` is not set in this environment, plus one downstream
  assertion from an LLM-backed run producing no results.
- 2026-05-09: `npm run format`, `npm run check`, and `npm run lint`
  result: all failed with npm `ENOENT` because this repository has no root
  `package.json`.
- 2026-05-09 review pass: `uv run pytest tests/test_streaming/test_stagegate.py tests/test_stagegate_trace_viewer.py -q`
  result after tightening name extraction: `23 passed, 2 warnings in 0.05s`.
- 2026-05-09 review pass: `make check-all`
  result: Ruff check passed and Ruff format left 321 files unchanged.
- 2026-05-09 review pass: `make test-voice`
  result: `266 passed, 3 skipped, 83 deselected, 2 warnings in 0.45s`.
- 2026-05-09 pre-write validator pass: initial
  `uv run pytest tests/test_streaming/test_stagegate.py -q`
  failed during collection with `ModuleNotFoundError: No module named 'tau2'`
  in the fresh `.venv`; `uv sync --extra voice --extra dev` completed
  successfully.
- 2026-05-09 pre-write validator pass:
  `uv run pytest tests/test_streaming/test_stagegate.py -q`
  result before formatting: `30 passed, 2 warnings in 0.06s`.
- 2026-05-09 pre-write validator pass:
  `uv run ruff check src/tau2/voice/audio_native/openai/stagegate/validator.py src/tau2/voice/audio_native/openai/stagegate/controller.py src/tau2/orchestrator/full_duplex_orchestrator.py tests/test_streaming/test_stagegate.py`
  result: `All checks passed!`.
- 2026-05-09 pre-write validator pass: `make test-voice`
  result before formatting: `275 passed, 3 skipped, 83 deselected, 2 warnings
  in 0.74s`.
- 2026-05-09 pre-write validator pass: `make check-all`
  result: Ruff check passed and Ruff format reformatted 1 file.
- 2026-05-09 pre-write validator pass:
  `uv run pytest tests/test_streaming/test_stagegate.py -q`
  result after formatting: `30 passed, 2 warnings in 0.04s`.
- 2026-05-09 pre-write validator pass: `make test`
  result: `164 passed, 17 failed, 1 xfailed, 14 warnings`; failures are
  credential-dependent LLM tests failing with `litellm.AuthenticationError`
  because `OPENAI_API_KEY` is not set in this environment, plus one downstream
  assertion from an LLM-backed run producing no results.
- 2026-05-09 pre-write validator pass:
  `npm run format`, `npm run check`, and `npm run lint`
  result: all failed with npm `ENOENT` because this repository has no root
  `package.json`.
- 2026-05-09 pre-write validator pass: `make test-voice`
  result after formatting: `275 passed, 3 skipped, 83 deselected, 2 warnings
  in 0.48s`.
- 2026-05-09 cleanup review pass:
  `uv run ruff check src/tau2/voice/audio_native/openai/stagegate/validator.py tests/test_streaming/test_stagegate.py`
  result: `All checks passed!`.
- 2026-05-09 cleanup review pass:
  `uv run pytest tests/test_streaming/test_stagegate.py -q`
  result after tightening exact identifier and consequence checks and final
  formatting: `33 passed, 2 warnings in 0.05s`.
- 2026-05-09 cleanup review pass: `make test-voice`
  result after final formatting: `278 passed, 3 skipped, 83 deselected,
  2 warnings in 0.53s`.
- 2026-05-09 cleanup review pass: `make check-all`
  result after final formatting: Ruff check passed and Ruff format left
  322 files unchanged.
- 2026-05-09 cleanup review pass: `make test`
  result: `164 passed, 17 failed, 1 xfailed, 14 warnings`; failures are
  credential-dependent LLM tests failing with `litellm.AuthenticationError`
  because `OPENAI_API_KEY` is not set in this environment, plus one downstream
  assertion from an LLM-backed run producing no results.
- 2026-05-09 cleanup review pass:
  `npm run format`, `npm run check`, and `npm run lint`
  result: all failed with npm `ENOENT` because this repository has no root
  `package.json`.
- 2026-05-09 visibility-boundary fix:
  `uv sync --extra voice --extra dev`
  result: completed successfully after the first focused pytest run failed
  during collection with `ModuleNotFoundError: No module named 'tau2'` in the
  fresh `.venv`.
- 2026-05-09 visibility-boundary fix:
  `uv run pytest tests/test_streaming/test_stagegate.py tests/test_stagegate_trace_viewer.py tests/test_stagegate_prohibited_diff_guard.py -q`
  result after fixing test chronology and formatting: `41 passed, 2 warnings
  in 0.07s`.
- 2026-05-09 visibility-boundary fix:
  `uv run ruff check .`
  result: `All checks passed!`.
- 2026-05-09 visibility-boundary fix:
  `make check-all`
  result: Ruff check passed and Ruff format left 325 files unchanged.
- 2026-05-09 visibility-boundary fix: `git diff --check`
  result: passed with no whitespace errors.
- 2026-05-09 visibility-boundary fix:
  `python scripts/stagegate_prohibited_diff_guard.py --base origin/main --head HEAD`
  result: failed because no `python` executable exists on this shell's `PATH`.
  Equivalent project-Python command
  `uv run python scripts/stagegate_prohibited_diff_guard.py --base origin/main --head HEAD`
  passed; `python3 scripts/stagegate_prohibited_diff_guard.py --base origin/main --head HEAD`
  also passed after the guard script was kept stdlib-only.
- 2026-05-09 visibility-boundary fix:
  `npm run format`, `npm run check`, and `npm run lint`
  result: all failed with npm `ENOENT` because this repository has no root
  `package.json`.

Warnings observed in the passing focused and voice test commands:

- `audioop` deprecation warning from
  `src/tau2/voice/utils/audio_preprocessing.py`.
- unknown pytest config option `asyncio_default_fixture_loop_scope`.

## Decision Log

- `TAU2_STAGEGATE_CONDITION=stagegate` owns the entity ledger and pre-write
  validator. `stage_only` remains a StageOnly control condition with no ledger
  or validator.
- `FullDuplexOrchestrator._execute_stagegate_tool_call()` is the right boundary
  for `advance_stage` interception because it can return a normal
  `ToolMessage` without touching domain state.
- `JsonlTraceWriter` remains independent from τ-bench checkpointing so traces
  can be enabled without changing `SimulationRun`, evaluator inputs, scoring,
  or result files.
- 2026-05-09 visibility-boundary fix: StageGate records user text only when it
  is delivered to the agent as `incoming_for_agent`; newly emitted simulator
  chunks are not validator-visible in the same tick.
- 2026-05-09 visibility-boundary fix: Evaluator-derived final outcome rows are
  produced only by posthoc analysis scripts outside
  `src/tau2/voice/audio_native/openai/stagegate/`.
- 2026-05-09: `run_end` on exception uses
  `termination_reason="exception"` because retry infrastructure owns the final
  failed `SimulationRun` object for exhausted attempts.
- 2026-05-09: StageGate ledger state is keyed only by public domain name, never
  by task ID, expected final database state, evaluator output, simulator
  private state, or clean simulator text.
- 2026-05-09: StageOnly remains ledger-free to preserve V1 behavior; only
  `TAU2_STAGEGATE_CONDITION=stagegate` owns an `EntityLedger`.
- 2026-05-09: The first ledger implementation uses deterministic structured
  extraction from visible tool arguments/results. It does not attempt
  free-text parsing of simulator speech or `advance_stage.observed_facts`.
- 2026-05-09 review pass: String fields named `name` are not treated as person
  names because product and plan tool outputs also expose `name`; person names
  come from structured `full_name`, `first_name`/`last_name`, or `name` objects,
  while telecom plan display names are captured only from plan-shaped payloads.
- 2026-05-09 pre-write validator pass: The validator uses only public domain
  name, public tool schemas/names/descriptions, visible participant text,
  model tool-call arguments, official domain tool results, and ledger state
  derived from those visible events.
- 2026-05-09 pre-write validator pass: Blocked writes return an error
  `ToolMessage` with a corrective stage packet and increment the existing
  orchestrator tool-error counter; the domain environment is not called.
- 2026-05-09 pre-write validator pass: Task IDs remain trace metadata only and
  are not passed into validator decision logic.
- 2026-05-09 cleanup review pass: Policy preconditions now require every
  required visible field for the relevant read inspection, avoiding underblocks
  where one field such as bill status was present but another such as amount
  due was missing.
- 2026-05-09 cleanup review pass: Nested list identifiers from read-tool
  payloads, such as `bill_ids`, are recorded under equivalent singular names
  such as `bill_id` so successful official lookups can verify exact write
  arguments without using hidden state.
- 2026-05-09 cleanup review pass: Exact identifier mentions in the assistant's
  visible action summary use token-boundary matching so a neighboring ID cannot
  satisfy confirmation for the requested mutable record.

## Remaining Work

- Run a later one-task OpenAI audio-native smoke with credentials and runtime
  budget available.

## Outcomes & Retrospective

StageOnly, the V2 entity ledger, and the pre-write validator are implemented
and narrowly tested. Baseline remains disabled unless
`TAU2_STAGEGATE_CONDITION` is set to `stage_only` or `stagegate`; `stage_only`
keeps the prior orchestration-only behavior, while `stagegate` now blocks
unsafe write/action calls before domain-state mutation and returns corrective
stage packets to the model.
