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
  It also defines the StageGate runtime evidence model:
  `AGENT_VISIBLE_TRANSCRIPT`, `MODEL_TOOL_ARGUMENT`, `DOMAIN_TOOL_OUTPUT`,
  `ASSISTANT_UTTERANCE`, and the runtime-forbidden
  `SIMULATOR_GOLD_TEXT` / `POSTHOC_ORACLE` sources.
- `src/tau2/voice/audio_native/openai/stagegate/trace.py`
  provides `JsonlTraceWriter.from_env()` controlled by `TAU2_TRACE_JSONL`.
- `src/tau2/voice/audio_native/openai/stagegate/orchestrator.py`
  provides `StagePacketOrchestrator`, which builds deterministic, stage-scoped
  packets from `advance_stage` arguments, public domain name, public tool
  names, and, for StageGate only, the typed entity ledger. Packets expose only
  stage-relevant missing facts rather than every missing domain slot.
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
  validator ownership when `condition="stagegate"`, explicit assistant
  utterance and agent-visible user transcript recording, `advance_stage` loop
  guard state, trace summary counters, and `ledger_update` / validator trace
  events.
- `src/tau2/agent/discrete_time_audio_native_agent.py` wires the controller
  into OpenAI audio-native session setup without changing baseline behavior.
- `src/tau2/orchestrator/full_duplex_orchestrator.py` records assistant
  utterances for the validator, does not pass audio-native user chunk
  `UserMessage.content` into StageGate runtime state, intercepts
  `advance_stage`, and validates non-`advance_stage` domain tool calls before
  domain execution only when StageOnly or StageGate is enabled. Allowed calls
  continue through `Environment.get_response()` unchanged; blocked calls return
  an error `ToolMessage` containing a corrective stage packet and do not touch
  domain state. Baseline plus passive JSONL tracing stays on the normal domain
  tool execution path.
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
- [x] Fix the audio-native clean-text leak: full-duplex user chunks are no
  longer recorded into StageGate validator or ledger state from
  `UserMessage.content`, because the OpenAI audio-native agent receives only
  `user_audio`.
- [x] Add explicit runtime evidence-source admission. User confirmation can be
  recorded only from `AGENT_VISIBLE_TRANSCRIPT`; model tool arguments remain
  unconfirmed model belief, official domain-tool outputs remain verified facts,
  and simulator gold text / posthoc oracle sources are rejected at runtime.
- [x] Keep baseline tracing passive: `TAU2_TRACE_JSONL` with
  `condition="baseline"` may emit run-level traces, but baseline domain tools
  do not route through `_execute_stagegate_tool_call()`.
- [x] Preserve trace-only per-tick visibility without enforcement: baseline
  trace-only runs now record assistant utterance, model function call, domain
  tool call, and domain tool result trace rows while executing domain tools
  through direct `Environment.get_response()` calls outside the validator
  wrapper.
- [x] Wire OpenAI input-audio transcription events into the explicit
  `AGENT_VISIBLE_TRANSCRIPT` evidence path. These provider transcripts can
  satisfy confirmation; clean audio-native `UserMessage.content` remains
  rejected and unused.
- [x] Avoid logging raw provider user transcript text. OpenAI adapter debug
  logs now include only input-transcription metadata, while the transcript
  remains available inside the runtime evidence object.
- [x] Move evaluator-derived final outcome rows out of StageGate runtime and
  into `scripts/stagegate_posthoc_outcomes.py`, which writes
  `oracle_analysis.jsonl` after result files exist.
- [x] Add `scripts/stagegate_prohibited_diff_guard.py` to prevent StageGate
  branches from modifying benchmark-controlled task, domain, evaluator, user,
  metrics, or scoring files.
- [x] Reserve `benchmark_task_id` for trace/posthoc metadata and rename the
  validator's mock-domain task identifier to `service_task_ref`.
- [x] Add `scripts/stagegate_final_run_hygiene.py` to reject final-run manifests
  with task filters, non-regular speech, missing condition/domain coverage, or
  inconsistent settings across conditions.
- [x] Add Milestone 7.5 StageGate hardening before paid voice smoke tests:
  per-run `advance_stage` loop guard defaults, stage-scoped packets, fallback
  packets, `stage_loop_guard_triggered` events, and trace summary payloads on
  `trace_summary` and `run_end`.
- [x] Replace transcript-pattern confirmation with a structured pending-write
  protocol. Side-effecting write attempts now create a pending write keyed by
  tool name and stable argument fingerprint; the model must call
  `record_pending_write_summary` and `record_pending_write_confirmation` before
  the same fingerprinted retry is allowed.
- [x] Make pending-write tools operate on the single active pending write so the
  model-facing protocol does not require copying an opaque pending-write ID.
- [x] Block `transfer_to_human_agents` while a resolvable active pending write is
  waiting for summary, structured confirmation, or direct retry.
- [x] Add `next_required_steps`, `allowed_internal_tools`, and
  `disallowed_tools` to corrective and active-pending packets so the model sees
  the pending-write protocol as a mechanical checklist instead of prose.
- [x] Add `next_tool_call` to pending-write packets so the model sees one
  immediate structured tool-call affordance in addition to the checklist.
- [x] Keep denied and unclear decisions structural: denied blocks the write and
  allows non-write resolution; unclear keeps the protocol active, blocks
  transfer, and requires a later user turn before recording another decision.
- [x] Emit pending-write trace events for creation, structured summary record,
  structured confirmation/denial/unclear decisions, mismatched retry, and
  consumption.
- [x] Gate `advance_stage` with pending-write state so unconsumed pending writes
  cannot drift to `verify_result_and_close`.

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
- clean audio-native user chunk `UserMessage.content` does not satisfy
  validator confirmation at the same tick or later ticks;
- clean `UserMessage.content` is rejected as StageGate runtime evidence and
  cannot create user-confirmed ledger state;
- explicit `record_agent_visible_user_transcript()` evidence can satisfy
  confirmation after an assistant action summary and required read-tool
  inspection;
- model tool argument text cannot satisfy user confirmation;
- forbidden simulator-gold-text and posthoc-oracle evidence sources are rejected
  at runtime;
- baseline with `TAU2_TRACE_JSONL` does not route domain tools through
  `_execute_stagegate_tool_call()`;
- StageGate runtime package code does not read evaluator-derived reward fields;
- StageGate runtime package code does not import or call the posthoc oracle
  outcome script;
- prohibited-path guard classification for benchmark-controlled files.
- static control-code coverage that StageGate validator/ledger/packet logic does
  not use `task_id` as a domain identifier;
- action summaries must state a consequence, not only an intended action;
- read-only tools are not overblocked;
- validator leakage guards show no task objective, expected final DB,
  user-simulator private state, evaluator result, or task-ID routing inputs.
- StageGate hardening tests cover max `advance_stage` call guard, repeated
  stage guard, repeated blocker guard, guard trace emission, fallback packet
  no-crash/no-mutation behavior, stage-scoped missing facts, identity-stage
  exclusion of later-stage facts in retail/airline/telecom, StageGate ledger
  enrichment, and trace summary counters.
- pending-write tests cover blocked exchange creation, structured summary
  recording, structured confirmation after a later user turn, same-fingerprint
  retry allow, changed-argument mismatch blocks, denied/unclear decisions
  blocking retry, corrective direct-retry packet text, stage guard behavior for
  blocked/confirmed/consumed writes, pending-write trace events,
  baseline/StageOnly absence of validator state, and absence of semantic
  transcript regex in the validator source.

Trace/query tests in `tests/test_stagegate_trace_viewer.py` cover:

- posthoc oracle analysis may read `reward_info` and source task IDs only in the
  separate `scripts/stagegate_posthoc_outcomes.py` path;
- `scripts/trace_queries.sql` uses `benchmark_task_id` consistently and does
  not query the runtime trace field as `task_id`.

Focused tests in `tests/test_stagegate_final_run_hygiene.py` cover:

- accepting the required baseline/stage_only/stagegate matrix across retail,
  airline, and telecom;
- rejecting `--num-tasks`, `--task-ids`, and manifest task filters;
- rejecting non-regular speech complexity;
- rejecting inconsistent model, timeout, seed, or concurrency across conditions;
- rejecting missing required domain/condition cells.

## Validation Evidence

- 2026-05-10 audio-text leak fix:
  `uv run pytest tests/test_streaming/test_stagegate.py -q`
  initially failed at collection in the fresh worktree because `tau2` was not
  importable from the unsynced `.venv`; `uv sync --extra voice --extra dev
  --extra experiments` completed successfully.
- 2026-05-10 audio-text leak fix:
  `uv run pytest tests/test_streaming/test_stagegate.py -q`
  result after sync: `42 passed, 2 warnings in 0.12s`.
- 2026-05-10 audio-text leak fix:
  `uv run pytest tests/test_stagegate_trace_viewer.py -q`
  result: `4 passed, 2 warnings in 0.01s`.
- 2026-05-10 audio-text leak fix:
  `uv run pytest tests/test_streaming/test_stagegate.py
  tests/test_stagegate_trace_viewer.py
  tests/test_stagegate_prohibited_diff_guard.py -q`
  result: `49 passed, 2 warnings in 0.08s`.
- 2026-05-10 audio-text leak fix:
  `uv run pytest tests/test_stagegate_final_run_hygiene.py
  tests/test_stagegate_modal_runner_config.py -q`
  result: `23 passed, 2 warnings in 0.11s`.
- 2026-05-10 audio-text leak fix:
  `python3 scripts/stagegate_prohibited_diff_guard.py --base origin/main --head HEAD`
  result: `StageGate prohibited-path guard passed.`
- 2026-05-10 audio-text leak fix:
  `make check-all` result after Ruff formatting pass: `All checks passed!`;
  final rerun result: `All checks passed!` and `329 files left unchanged`.
- 2026-05-10 audio-text leak fix:
  `make test-voice` first hit sandbox denial on `/Users/.../.cache/uv`;
  rerun with approved uv-cache access passed:
  `287 passed, 3 skipped, 83 deselected, 2 warnings in 0.80s`.
- 2026-05-10 audio-text leak fix:
  `make test` first hit the same uv-cache sandbox denial; rerun with approved
  uv-cache access reached the suite and failed because `OPENAI_API_KEY` is not
  set for credential-dependent LiteLLM/OpenAI tests:
  `17 failed, 192 passed, 1 xfailed, 14 warnings in 22.13s`.
- 2026-05-10 audio-text leak fix:
  `npm run format`, `npm run check`, and `npm run lint` each failed with
  `ENOENT` because this Python repository has no root `package.json`.
- 2026-05-10 review follow-up:
  `uv run pytest tests/test_streaming/test_stagegate.py::test_baseline_trace_jsonl_does_not_route_tools_through_stagegate
  tests/test_streaming/test_stagegate.py::test_baseline_trace_jsonl_records_assistant_utterance_without_stagegate
  tests/test_streaming/test_stagegate.py::test_openai_adapter_records_input_transcription_event
  tests/test_streaming/test_stagegate.py::test_agent_wires_provider_user_transcript_to_stagegate_confirmation
  tests/test_streaming/test_stagegate.py::test_next_tick_clean_user_content_does_not_satisfy_validator -q`
  result: `5 passed, 2 warnings in 0.02s`.
- 2026-05-10 review follow-up:
  `uv run pytest tests/test_streaming/test_stagegate.py
  tests/test_stagegate_trace_viewer.py
  tests/test_stagegate_prohibited_diff_guard.py -q`
  result: `52 passed, 2 warnings in 0.12s`.
- 2026-05-10 review follow-up:
  `uv run pytest tests/test_streaming/test_discrete_time_audio_native_agent.py -q`
  result: `33 passed, 2 warnings in 0.06s`.
- 2026-05-10 review follow-up:
  `make check-all` result: `All checks passed!` and
  `329 files left unchanged`.
- 2026-05-10 sensitive logging follow-up:
  `uv run pytest tests/test_streaming/test_stagegate.py::test_openai_adapter_records_input_transcription_event -q`
  result: `1 passed, 2 warnings in 0.04s`.
- 2026-05-10 sensitive logging follow-up:
  `uv run pytest tests/test_streaming/test_stagegate.py
  tests/test_stagegate_trace_viewer.py
  tests/test_stagegate_prohibited_diff_guard.py -q`
  result: `52 passed, 2 warnings in 0.10s`.
- 2026-05-10 sensitive logging follow-up:
  `make check-all` result: `All checks passed!` and
  `329 files left unchanged`.
- 2026-05-10 Milestone 7.5 hardening:
  `uv run --extra dev --extra voice python -m pytest tests/test_streaming/test_stagegate.py -q`
  result after stage packet and loop guard changes:
  `55 passed, 2 warnings in 0.12s`.
- 2026-05-10 Milestone 7.5 hardening:
  `uv run --extra dev --extra voice python -m pytest tests/test_stagegate_trace_viewer.py tests/test_stagegate_prohibited_diff_guard.py tests/test_stagegate_final_run_hygiene.py tests/test_stagegate_modal_runner_config.py -q`
  result: `30 passed, 2 warnings in 0.09s`.
- 2026-05-10 Milestone 7.5 hardening:
  `make format` result: `2 files reformatted, 327 files left unchanged`.
- 2026-05-10 Milestone 7.5 hardening:
  `make check-all` result: `All checks passed!` and
  `329 files left unchanged`.
- 2026-05-10 Milestone 7.5 review follow-up:
  `uv run --extra dev --extra voice python -m pytest tests/test_streaming/test_stagegate.py -q`
  result after fixing StageOnly observed-fact hint matching:
  `56 passed, 2 warnings in 0.14s`.
- 2026-05-10 Milestone 7.5 review follow-up:
  `uv run --extra dev --extra voice python -m pytest tests/test_streaming/test_stagegate.py -q`
  result after treating slash-separated hint options as alternatives:
  `57 passed, 2 warnings in 0.16s`.

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
- 2026-05-09 task-ID isolation and final-run hygiene fix:
  `uv run pytest tests/test_streaming/test_stagegate.py tests/test_stagegate_trace_viewer.py tests/test_stagegate_prohibited_diff_guard.py -q`
  result: `43 passed, 2 warnings in 0.11s`.
- 2026-05-09 task-ID isolation and final-run hygiene fix:
  `uv run pytest tests/test_stagegate_final_run_hygiene.py -q`
  result: `5 passed, 2 warnings in 0.01s`.
- 2026-05-09 task-ID isolation and final-run hygiene fix:
  `uv run ruff check .`
  result: `All checks passed!`.
- 2026-05-09 task-ID isolation and final-run hygiene fix:
  `make check-all`
  result: Ruff check passed and Ruff format left 327 files unchanged.
- 2026-05-09 task-ID isolation and final-run hygiene fix:
  `python3 scripts/stagegate_prohibited_diff_guard.py --base origin/main --head HEAD`
  result: passed.
- 2026-05-10 smoke_007 confirmation-validator triage:
  `uv run --extra voice --extra dev python -m pytest tests/test_streaming/test_stagegate.py -q`
  result after final formatting: `64 passed, 2 warnings in 0.17s`.
- 2026-05-10 smoke_007 confirmation-validator triage:
  `uv run --extra voice --extra dev python -m pytest tests/test_stagegate_trace_viewer.py tests/test_stagegate_prohibited_diff_guard.py tests/test_stagegate_final_run_hygiene.py tests/test_stagegate_modal_runner_config.py -q`
  result: `39 passed, 2 warnings in 0.11s`.
- 2026-05-10 smoke_007 confirmation-validator triage:
  `make format`
  result after final formatting: Ruff format left 329 files unchanged.
- 2026-05-10 smoke_007 confirmation-validator triage:
  `make check-all`
  result: Ruff check passed and Ruff format left 329 files unchanged.
- 2026-05-10 smoke_007 confirmation-validator triage:
  `git diff --check`
  result: passed with no whitespace errors.
- 2026-05-10 smoke_007 confirmation-validator triage:
  `uv run python scripts/stagegate_prohibited_diff_guard.py --base origin/stagegate --head HEAD`
  result: `StageGate prohibited-path guard passed.`
- 2026-05-10 smoke_008 post-patch smoke:
  `uv run --with modal modal run modal_tau3_voice_stagegate.py --batch-id smoke_008 --repo-url https://github.com/we-are-pap-Inc/tau2-bench.git --repo-ref e595172fad05c15dc95e35adb35431f54c328d48 --mode smoke`
  completed all three Modal jobs. Baseline result: reward `1.0`,
  termination `user_stop`. StageOnly result: reward `1.0`, termination
  `user_stop`. StageGate result: reward `0.0`, termination `max_steps`.
- 2026-05-10 smoke_008 post-patch smoke:
  `uv run --with modal modal run modal_tau3_voice_stagegate.py --batch-id smoke_008 --collect-completed`
  result: completed manifest written with 3 runs at
  `/runs/smoke_008/batch_manifest_completed.json`.
- 2026-05-10 smoke_008 post-patch smoke:
  StageGate trace replay against the updated local validator showed the live
  exchange summaries would now match the attempted write at ticks `1000` and
  `1368`, with user confirmations at ticks `1127` and `1383`.
- 2026-05-10 smoke_008 exchange-summary follow-up:
  `uv run --extra voice --extra dev python -m pytest tests/test_streaming/test_stagegate.py -q -k 'exchange_summary or confirmation_before_exchange_summary or missing_exchange_summary'`
  result: `5 passed, 60 deselected, 2 warnings in 0.07s`.
- 2026-05-10 smoke_008 exchange-summary follow-up:
  `uv run --extra voice --extra dev python -m pytest tests/test_streaming/test_stagegate.py -q`
  result: `65 passed, 2 warnings in 0.17s`.
- 2026-05-10 smoke_008 exchange-summary follow-up:
  `uv run --extra voice --extra dev python -m pytest tests/test_stagegate_trace_viewer.py tests/test_stagegate_prohibited_diff_guard.py tests/test_stagegate_final_run_hygiene.py tests/test_stagegate_modal_runner_config.py -q`
  result: `39 passed, 2 warnings in 0.12s`.
- 2026-05-10 smoke_008 exchange-summary follow-up:
  `make format` result: Ruff format left `329 files` unchanged.
- 2026-05-10 smoke_008 exchange-summary follow-up:
  `make check-all` result: Ruff check passed and Ruff format left
  `329 files` unchanged.
- 2026-05-10 smoke_008 exchange-summary follow-up:
  `git diff --check` result: passed with no whitespace errors.
- 2026-05-10 smoke_008 exchange-summary follow-up:
  `uv run python scripts/stagegate_prohibited_diff_guard.py --base origin/stagegate --head HEAD`
  result: `StageGate prohibited-path guard passed.`
- 2026-05-10 smoke_008 exchange-summary follow-up:
  `npm run format`, `npm run check`, and `npm run lint` all failed with npm
  `ENOENT` because this repository has no root `package.json`.
- 2026-05-10 structural pending-write confirmation:
  `uv run --extra voice --extra dev python -m pytest tests/test_streaming/test_stagegate.py -q`
  result: `70 passed, 2 warnings in 0.29s`.
- 2026-05-10 structural pending-write confirmation:
  `uv run --extra voice --extra dev python -m pytest tests/test_stagegate_trace_viewer.py tests/test_stagegate_prohibited_diff_guard.py tests/test_stagegate_final_run_hygiene.py tests/test_stagegate_modal_runner_config.py -q`
  result: `39 passed, 2 warnings in 0.12s`.
- 2026-05-10 structural pending-write confirmation:
  local replay of `/private/tmp/stagegate_smoke_008_stagegate_trace_events.jsonl`
  through the updated validator allowed both attempted
  `exchange_delivered_order_items` writes; each matched `action_type`,
  `old_items`, `new_items`, `consequence`, and `confirmation_request`.
- 2026-05-10 structural pending-write confirmation:
  `make format` result: Ruff format reformatted 1 file, then final
  `make check-all` left `329 files` unchanged with Ruff checks passing.
- 2026-05-10 structural pending-write confirmation:
  `uv run python scripts/stagegate_prohibited_diff_guard.py --base origin/main --head HEAD`
  result: `StageGate prohibited-path guard passed.`
- 2026-05-10 structural pending-write confirmation:
  `npm run format`, `npm run check`, and `npm run lint` all failed with npm
  `ENOENT` because this repository has no root `package.json`.
- 2026-05-10 structured pending-write protocol:
  `uv run --extra voice --extra dev python -m pytest tests/test_streaming/test_stagegate.py -q`
  result after formatting: `70 passed, 2 warnings in 0.25s`.
- 2026-05-10 structured pending-write protocol:
  `uv run --extra voice --extra dev python -m pytest tests/test_stagegate_trace_viewer.py tests/test_stagegate_prohibited_diff_guard.py tests/test_stagegate_final_run_hygiene.py tests/test_stagegate_modal_runner_config.py -q`
  result: `39 passed, 2 warnings in 0.12s`.
- 2026-05-10 structured pending-write protocol:
  `make format` result: `2 files reformatted, 327 files left unchanged`;
  `make check-all` result: `All checks passed!` and `329 files left unchanged`.
- 2026-05-10 structured pending-write protocol:
  `uv run python scripts/stagegate_prohibited_diff_guard.py --base origin/main --head HEAD`
  result: `StageGate prohibited-path guard passed.`
- 2026-05-10 structured pending-write protocol:
  local smoke_008 trace replay against
  `/private/tmp/stagegate_smoke_008_stagegate_trace_events.jsonl` shows direct
  replay blocks as `missing_action_summary` because the historical trace lacks
  the new internal protocol events. Injecting the structured
  `record_pending_write_summary` / later user turn /
  `record_pending_write_confirmation` sequence for the same fingerprint returns
  `allow` / `validated`.
- 2026-05-10 structured pending-write protocol:
  `npm run format`, `npm run check`, and `npm run lint` all failed with npm
  `ENOENT` because this repository has no root `package.json`.
- 2026-05-10 active pending-write protocol:
  `uv run --extra voice --extra dev python -m pytest tests/test_streaming/test_stagegate.py -q`
  result: `73 passed, 2 warnings in 0.35s`.
- 2026-05-10 active pending-write protocol:
  `uv run --extra voice --extra dev python -m pytest tests/test_stagegate_trace_viewer.py tests/test_stagegate_prohibited_diff_guard.py tests/test_stagegate_final_run_hygiene.py tests/test_stagegate_modal_runner_config.py -q`
  result: `39 passed, 2 warnings in 0.11s`.
- 2026-05-10 active pending-write protocol:
  `make check-all` result: `All checks passed!` and `329 files left
  unchanged`.
- 2026-05-10 active pending-write protocol:
  `uv run python scripts/stagegate_prohibited_diff_guard.py --base origin/main --head HEAD`
  result: `StageGate prohibited-path guard passed.`
- 2026-05-10 active pending-write affordance hardening:
  `uv run --extra voice --extra dev python -m pytest tests/test_streaming/test_stagegate.py -q`
  result: `78 passed, 2 warnings in 0.33s`.
- 2026-05-10 active pending-write affordance hardening:
  `uv run --extra voice --extra dev python -m pytest tests/test_stagegate_trace_viewer.py tests/test_stagegate_prohibited_diff_guard.py tests/test_stagegate_final_run_hygiene.py tests/test_stagegate_modal_runner_config.py -q`
  result: `39 passed, 2 warnings in 0.11s`.
- 2026-05-10 active pending-write affordance hardening:
  `make check-all` result: `All checks passed!` and `329 files left
  unchanged`.
- 2026-05-10 active pending-write affordance hardening:
  `uv run python scripts/stagegate_prohibited_diff_guard.py --base origin/main --head HEAD`
  result: `StageGate prohibited-path guard passed.`
- 2026-05-11 pending-write `next_tool_call` affordance:
  `uv run --extra voice --extra dev python -m pytest tests/test_streaming/test_stagegate.py -q`
  result: `78 passed, 2 warnings in 0.24s`.
- 2026-05-11 pending-write `next_tool_call` affordance:
  `uv run --extra voice --extra dev python -m pytest tests/test_stagegate_trace_viewer.py tests/test_stagegate_prohibited_diff_guard.py tests/test_stagegate_final_run_hygiene.py tests/test_stagegate_modal_runner_config.py -q`
  result: `39 passed, 2 warnings in 0.09s`.
- 2026-05-11 pending-write `next_tool_call` affordance:
  `make check-all` result: `All checks passed!` and `329 files left unchanged`.
- 2026-05-11 pending-write `next_tool_call` affordance:
  `uv run python scripts/stagegate_prohibited_diff_guard.py --base origin/main --head HEAD`
  result: `StageGate prohibited-path guard passed.`

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
- 2026-05-09 pre-write validator pass: Benchmark task IDs remain trace/posthoc
  metadata only and are represented as `benchmark_task_id` in StageGate traces.
  Domain-control identifiers must use domain-specific names; the mock service
  task identifier is represented internally as `service_task_ref`, not `task_id`.
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
- 2026-05-10 Milestone 7.5 hardening: StageOnly and StageGate now guard
  repeated `advance_stage` calls without returning tool errors or mutating
  domain state. The guard returns a final fallback stage packet and records
  `stage_loop_guard_triggered`.
- 2026-05-10 Milestone 7.5 hardening: Stage packets are scoped by stage and
  public domain name. Identity/authentication packets intentionally avoid
  asking for later-stage payment, fee, address, plan, device, or change facts
  unless the user already raised them.
- 2026-05-10 Milestone 7.5 hardening: Runtime trace summaries remain oracle-free
  and include stage-call counts, validator allow/block counts, ledger update
  count, final stage, stage sequence, last stage packet, and last validator or
  corrective packet.
- 2026-05-10 Milestone 7.5 review follow-up: StageOnly observed-fact matching
  now uses word-boundary tokenization, stopword filtering, and full-token
  alternative matching. This prevents short words such as `or`, `and`, and `if`
  from matching unrelated words and suppressing required stage hints.
- 2026-05-10 Milestone 7.5 review follow-up: StageOnly observed-fact matching
  treats slash-separated hint terms such as `payment/refund` as alternatives
  instead of requiring both terms to appear.
- 2026-05-10 smoke_007/smoke_008 transcript-pattern triage is superseded: the
  validator no longer aggregates transcript text or matches natural-language
  summaries/confirmations. Transcript events establish ordering only.
- 2026-05-10 smoke_007 triage: `advance_stage` is not allowed to advance from
  `execute_write_action` to `verify_result_and_close` in StageGate unless a
  side-effecting domain tool has actually returned successfully. Model-supplied
  `observed_facts` cannot override the validator's blocked-write state.
- 2026-05-10 smoke_007 triage: Retail item identifiers are split into semantic
  slots for order items, candidate replacements, selected old items, and
  selected new items. Product variant lists are not surfaced as user-facing
  `item_id` ambiguities in closeout packets.
- 2026-05-10 smoke_008 follow-up is superseded: exchange-specific transcript
  summary matching has been removed in favor of the structured pending-write
  protocol.
- 2026-05-10 structural pending-write confirmation: The validator now treats
  the attempted side-effecting write as the object being summarized and
  confirmed. Confirmation is valid only for the matching tool name and argument
  fingerprint; changed retries require a fresh summary and confirmation.
- 2026-05-10 structural pending-write confirmation: `advance_stage` cannot
  close a write-intent path while the pending write is `needs_summary`,
  `summarized`, `confirmed`, or `mismatched_retry`; only a consumed successful
  side-effecting domain-tool result permits closeout.
- 2026-05-10 structured pending-write protocol: The validator no longer parses
  assistant or user transcript text for summary or consent. The only write
  confirmation path is a StageGate-only internal protocol:
  `record_pending_write_summary`, a later user-turn event, then
  `record_pending_write_confirmation`.
- 2026-05-10 smoke_009 triage: the runtime correctly created
  `pending_write=needs_summary`, but the model-facing affordance failed because
  the model believed the pending-write record step required an unavailable ID
  and transferred to a human. Pending-write recorder tools now resolve the
  active pending write by default, and transfer is blocked while that active
  write remains structurally resolvable.

## Remaining Work

- Run a later one-task OpenAI audio-native smoke with credentials and runtime
  budget available after all Milestone 7.5 verification commands pass.

## Outcomes & Retrospective

StageOnly, the V2 entity ledger, and the pre-write validator are implemented
and narrowly tested. Baseline remains disabled unless
`TAU2_STAGEGATE_CONDITION` is set to `stage_only` or `stagegate`; `stage_only`
keeps the prior orchestration-only behavior, while `stagegate` now blocks
unsafe write/action calls before domain-state mutation and returns corrective
stage packets to the model.
