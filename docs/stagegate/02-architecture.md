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
update the ledger or satisfy confirmation. If no agent-visible user transcript
exists, confirmation remains missing.

Validator checks:

1. Is this a write/action tool?
2. Is identity/account/order/reservation verified?
3. Are exact identifiers confirmed or tool-verified?
4. Has relevant state been inspected with read tools?
5. Are policy preconditions satisfied?
6. Did the assistant summarize intended action and consequence?
7. Did the user confirm after that summary?
8. Are tool arguments complete and non-ambiguous?

Allow result:

    {"decision": "allow"}

Block result:

    {
      "decision": "block",
      "reason": "missing_confirmation",
      "corrective_packet": {
        "say_next": "Before I make that change, please confirm that you want me to cancel reservation ABC123.",
        "allowed_next_tools": ["advance_stage"],
        "do_not": ["Do not call the cancellation tool until the user confirms."]
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
