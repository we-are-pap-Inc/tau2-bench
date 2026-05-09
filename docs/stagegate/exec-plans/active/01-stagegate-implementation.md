# ExecPlan 01 - StageGate Implementation

## Purpose

Implement StageGate for the OpenAI audio-native scaffold without changing
benchmark content or scoring. V1 StageOnly provides the `advance_stage`
orchestration tool, deterministic stage packets, and JSONL tracing. V2
StageGate adds a typed entity ledger for retail, airline, and telecom that
updates only from agent-visible function-call arguments and official domain
tool outputs.

Pre-write validation remains out of scope for this implementation: no
corrective packet flow, no domain-tool blocking, no evaluator edits, no task
edits, no user-simulator edits, no domain-policy edits, and no domain-tool
edits.

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
- `src/tau2/voice/audio_native/openai/stagegate/controller.py`
  provides `StageGateController` with env parsing, prompt addendum,
  `session_tools()`, `is_advance_stage()`, `handle_advance_stage()`, ledger
  ownership when `condition="stagegate"`, ledger updates from visible events,
  and `ledger_update` trace events.
- `src/tau2/agent/discrete_time_audio_native_agent.py` wires the controller
  into OpenAI audio-native session setup without changing baseline behavior.
- `src/tau2/orchestrator/full_duplex_orchestrator.py` intercepts only
  `advance_stage`; all other tool calls continue through
  `Environment.get_response()` unchanged.
- `src/tau2/voice/audio_native/openai/__init__.py` lazy-loads provider and
  adapter classes so importing the StageOnly package does not make core
  τ-bench imports require voice-only dependencies.

The active package exports the ledger models. It intentionally does not export
or import `PreWriteValidator`, `ValidatorDecision`, `CorrectivePacket`,
validator tests, validator decisions, or tool blocking.

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
  `advance_stage_call`, `stage_packet_returned`, and post-evaluation
  `final_outcome`.
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

## Tests

Focused tests in `tests/test_streaming/test_stagegate.py` cover:

- baseline/unset environment leaves prompt and tool list unchanged;
- `TAU2_STAGEGATE_CONDITION=stage_only` adds exactly one `advance_stage` tool;
- `TAU2_STAGEGATE_CONDITION=stagegate` currently enables StageOnly behavior
  only and has no ledger or validator attributes;
- `advance_stage` returns a `stagegate.stage_packet.v1` packet;
- `advance_stage` does not call `Environment.get_response()` and does not
  mutate a fake domain toolkit counter;
- normal non-`advance_stage` domain tools still call
  `Environment.get_response()` unchanged;
- `TAU2_TRACE_JSONL` writes canonical `stagegate.trace.v1` rows for
  StageOnly packets, model function calls, domain tool calls/results, and
  posthoc final outcome rows.
- entity ledger domain-slot initialization and JSON serialization;
- model tool-argument updates with source/event/tick evidence;
- successful official tool-result verification;
- errored tool-result no-op behavior;
- contradiction handling and `ambiguous_facts` in StageGate packets;
- `ledger_update` trace event shape and metadata.
- regression coverage that retail product names and telecom plan names do not
  get misclassified as customer names.

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

Warnings observed in the passing focused and voice test commands:

- `audioop` deprecation warning from
  `src/tau2/voice/utils/audio_preprocessing.py`.
- unknown pytest config option `asyncio_default_fixture_loop_scope`.

## Decision Log

- `TAU2_STAGEGATE_CONDITION=stagegate` remains accepted for CLI and
  experiment compatibility, but it behaves identically to `stage_only` until a
  later V2 ExecPlan implements ledger and validation.
- `FullDuplexOrchestrator._execute_stagegate_tool_call()` is the right boundary
  for `advance_stage` interception because it can return a normal
  `ToolMessage` without touching domain state.
- `JsonlTraceWriter` remains independent from τ-bench checkpointing so traces
  can be enabled without changing `SimulationRun`, evaluator inputs, scoring,
  or result files.
- `final_outcome` trace emission is done only after `run_simulation()` attaches
  evaluator `reward_info`; it is marked `visible_to_agent=false` and
  `leakage_risk=posthoc_evaluator`.
- 2026-05-09: Batch trial context is attached to the orchestrator before
  `run_simulation()` so runtime events and posthoc `final_outcome` rows share
  the same trial identifier.
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

## Remaining Work

- Run a later one-task OpenAI audio-native smoke with credentials and runtime
  budget available.
- Implement pre-write validation under a separate explicit plan and validity
  review.

## Outcomes & Retrospective

StageOnly and the V2 entity ledger are implemented and narrowly tested.
Baseline remains disabled unless `TAU2_STAGEGATE_CONDITION` is set to
`stage_only` or `stagegate`; non-stage domain tools still execute through
`Environment.get_response()` unchanged, and StageGate ledger updates are
trace-only bookkeeping with no domain-state mutation or tool blocking.
