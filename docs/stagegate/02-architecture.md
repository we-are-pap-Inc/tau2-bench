# StageGate Architecture

## Summary

StageGate is a custom realtime voice-agent scaffold for τ-Voice.

It adds three pieces to the OpenAI audio-native path:

1. `advance_stage` orchestration tool.
2. Typed entity ledger.
3. Pre-write validator.

It must not change benchmark tasks, evaluator, user simulator, domain policies, domain tools, or scoring.

## Conditions

### V0 baseline

No StageGate code path active.

### V1 StageOnly

Adds a non-environment orchestration tool named `advance_stage`. The model calls it to receive a compact stage packet.

### V2 StageGate

Adds StageOnly plus typed ledger and validator.

## Persistent prompt

Use a compact session prompt:

    You are the customer-service voice agent for the current τ³ / τ-Voice domain.

    Global rules:
    - Follow the public domain policy exactly.
    - Use official domain tools for customer, order, reservation, account, flight, plan, or service actions.
    - Do not invent tool results.
    - Do not reveal internal stages, packets, benchmark internals, or orchestration.
    - Ask for clarification when identity, IDs, dates, numbers, plan names, or intent are ambiguous.
    - Before changing account, order, reservation, plan, or service state, make sure all policy prerequisites and confirmation requirements are satisfied.
    - When the next procedural step is unclear, call advance_stage.
    - If a write/action tool call is blocked by the server, follow the corrective instruction and ask only for the missing information.

## Orchestration tool schema

    {
      "type": "function",
      "name": "advance_stage",
      "description": "Request the next stage-specific operating instructions from the server-side orchestrator. This tool does not modify customer records or domain state.",
      "parameters": {
        "type": "object",
        "properties": {
          "current_stage": {"type": "string"},
          "observed_facts": {"type": "array", "items": {"type": "string"}},
          "last_action": {"type": "string"},
          "blocker": {"type": "string"}
        },
        "required": ["current_stage", "observed_facts", "last_action"]
      }
    }

## Stage machine

Use this simple cross-domain state machine:

1. `understand_intent`
2. `identify_or_authenticate`
3. `collect_required_exact_entities`
4. `inspect_state_with_read_tools`
5. `check_policy_eligibility`
6. `propose_action_and_confirm`
7. `execute_write_action`
8. `verify_result_and_close`

## Stage packet schema

A stage packet should be compact JSON:

    {
      "schema_version": "stagegate.stage_packet.v1",
      "stage": "identify_or_authenticate",
      "objective": "Establish the customer's identity before account-specific actions.",
      "known_facts": {
        "order_id": {"value": "O-12345", "status": "heard_not_confirmed"}
      },
      "missing_facts": ["customer email or phone number"],
      "ask_next": "Ask for the customer's email or phone number. Ask for one value only.",
      "allowed_read_tools": ["lookup_user", "lookup_order"],
      "allowed_write_tools": [],
      "do_not": ["Do not refund, cancel, modify, or change account state yet."],
      "exit_condition": "Identity is verified by an official lookup result, or the user cannot provide enough information.",
      "when_done": "Call advance_stage again with verified identity facts and the last tool result."
    }

## Entity ledger schema

The ledger is a typed dialogue-state tracker.

Each slot should include:

    {
      "field": "order_id",
      "value": "O-12345",
      "normalized_value": "O-12345",
      "status": "heard_not_confirmed",
      "confidence": 0.72,
      "source": "model_tool_argument",
      "evidence": [
        {"turn_id": 8, "event_id": "evt_abc", "quote_or_summary": "User appeared to say O-12345"}
      ],
      "last_updated_turn": 8
    }

Status enum:

- `missing`
- `hypothesized`
- `heard_not_confirmed`
- `repeated_back`
- `user_confirmed`
- `tool_verified`
- `contradicted`
- `stale`

Domain slots:

Retail:

- customer_name
- email
- phone
- order_id
- item_id
- order_item_ids
- candidate_replacement_item_ids
- selected_old_item_ids
- selected_new_item_ids
- return_reason
- refund_or_exchange_intent
- address
- payment_method
- confirmation

Airline:

- passenger_name
- reservation_id
- flight_number
- origin
- destination
- date
- requested_change
- fare_class
- payment_or_fee_acknowledgment
- confirmation

Telecom:

- account_id
- customer_name
- phone_line
- service_address
- plan_name
- device_id
- issue_type
- requested_change
- confirmation

## Pre-write validator

Wrap side-effecting domain tools. Do not mutate domain state inside the validator.

The validator admits only typed runtime evidence:

- `ASSISTANT_UTTERANCE` for model-spoken action summaries.
- `AGENT_VISIBLE_TRANSCRIPT` for user confirmation, only when the provider or
  adapter exposes the transcript to the same model path that received the user
  audio.
- `MODEL_TOOL_ARGUMENT` for model belief about entities and intended tool
  calls; this never satisfies user confirmation by itself.
- `DOMAIN_TOOL_OUTPUT` for verified facts returned by official domain tools.

`SIMULATOR_GOLD_TEXT` and `POSTHOC_ORACLE` are forbidden in runtime StageGate
state. In audio-native runs, clean `UserMessage.content` from the simulator is
not agent-visible when the model receives only `user_audio`, so it must not
update the ledger or satisfy confirmation. Agent-visible user transcript events
may establish only user-turn ordering; the validator does not parse transcript
text for consent or action summaries.

Validator checks:

1. Is this a write/action tool?
2. Are tool arguments complete and non-ambiguous?
3. Is identity/account/order/reservation verified?
4. Are exact identifiers confirmed or tool-verified?
5. Has relevant state been inspected with read tools?
6. Are policy preconditions satisfied?
7. Is there a matching pending write keyed by the attempted tool and stable
   argument fingerprint?
8. Has a later user turn occurred after the pending write was created?
9. Has the model called `commit_pending_write` with a structured decision for
   the active pending write?

Pending write confirmation:

- A side-effecting tool attempt with satisfied non-conversation checks creates
  a pending write record from the attempted tool name and arguments.
- The record stores the original tool name and arguments, a stable argument
  fingerprint, commit decision/tick, user-turn ordering state, and status.
- The validator does not regex-match assistant or user transcript text. The
  agent must explicitly call the StageGate internal tool `commit_pending_write`
  after summarizing the pending write to the user, asking for confirmation, and
  receiving a later user response.
- There is at most one active pending write. The model-facing commit tool
  applies to that active pending write and does not require a pending write ID.
- Corrective and active-pending packets include `next_required_steps` with exact
  instructions for telling the user the pending action/consequence, asking for
  confirmation, waiting for the user response, and calling
  `commit_pending_write`.
- Corrective and active-pending packets also include `next_tool_call`, a single
  immediate tool-call affordance for the next structured step.
- If `commit_pending_write(decision="confirmed")` is structurally valid,
  StageGate executes the stored original domain write once through the normal
  environment path.
- A successful side-effecting domain-tool result marks the pending write
  `consumed`; only consumed writes can satisfy write-intent closeout.
- `denied` keeps the write blocked and permits graceful close or alternative
  non-write assistance. `unclear` keeps the protocol active, asks one
  clarification, and allows another structured commit decision after a
  later user turn.
- `transfer_to_human_agents` is blocked while a resolvable active pending write
  is `needs_confirmation`, `unclear`, or `mismatched_retry`. Transfer is allowed
  with no active pending write, or after the active pending write is `denied`,
  `consumed`, or `expired`.

## Trajectory and Replay Boundary

StageGate separates model-visible scaffold behavior from replayable benchmark
environment actions:

- `model_requested_domain_tool`: the model attempted a domain tool call. This is
  trace-visible, but it is not automatically an environment execution.
- `stagegate_blocked_domain_tool`: StageGate intercepted the requested domain
  write before `Environment.get_response()`. The model still receives the
  corrective tool output, but the blocked call/result is stored only in
  internal tick fields and is excluded from replay.
- `environment_domain_tool_result`: the domain tool actually executed through
  the environment. Only these results are serialized into the canonical
  replayable tool-call/result fields.

The canonical `Tick.agent_tool_calls`, `Tick.user_tool_calls`, and matching
result fields contain only real environment executions. StageGate-internal
control tools (`advance_stage`, `commit_pending_write`) and StageGate-blocked
domain writes are stored in internal tick fields for auditability and
model-visible continuity, but are ignored by replay/evaluation conversion. When
`commit_pending_write` is confirmed, the stored original domain write is the
canonical replayable environment action.

Allow result:

    {"decision": "allow"}

Block result:

    {
      "decision": "block",
      "reason": "missing_confirmation",
      "corrective_packet": {
        "say_next": "Tell the user the pending action and consequence, ask for explicit confirmation, then after the user responds call commit_pending_write with a structured decision.",
        "next_required_steps": [
          {"step": "tell_user_pending_action"},
          {"step": "ask_user_to_confirm"},
          {"step": "wait_for_user_response"},
          {"step": "call_tool_if_user_confirms", "tool_name": "commit_pending_write", "arguments": {"decision": "confirmed", "basis": "latest_user_turn"}}
        ],
        "next_tool_call": {"name": "commit_pending_write", "arguments": {"decision": "confirmed", "basis": "latest_user_turn"}, "when": "after_user_confirms"},
        "allowed_internal_tools": ["commit_pending_write"],
        "disallowed_tools": ["advance_stage", "transfer_to_human_agents"],
        "allowed_next_tools": [],
        "do_not": ["Do not call advance_stage before commit_pending_write.", "Do not transfer while the active pending write is still resolvable."]
      }
    }

## Suggested file locations

Codex must inspect the actual repository before deciding exact paths. A likely layout is:

    src/tau2/voice/audio_native/openai/stagegate/
      __init__.py
      orchestrator.py
      stage_schema.py
      ledger.py
      validator.py
      trace.py
      packets.py

Patch the existing OpenAI audio-native adapter minimally to insert StageGate behavior when environment variable `TAU2_STAGEGATE_CONDITION` is set:

- `baseline`
- `stage_only`
- `stagegate`

## Trace events

All StageGate behavior should emit JSONL events with `schema_version`. See `docs/stagegate/05-tracing-and-visualization.md`.

Pending-write runtime events are:

- `pending_write_created`
- `pending_write_commit_requested`
- `pending_write_committed`
- `pending_write_denied`
- `pending_write_unclear`
- `pending_write_mismatch`
- `pending_write_consumed`
- `pending_write_commit_failed`
- `transfer_blocked_pending_write`
- `stagegate_blocked_domain_tool`
