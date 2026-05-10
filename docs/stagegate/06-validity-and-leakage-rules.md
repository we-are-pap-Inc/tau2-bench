# Validity and Leakage Rules

## Classification

Baseline may be a standard submission only if truly unmodified.

StageOnly and StageGate are custom submissions because they add orchestration tools, modify prompt architecture, and change scaffold/control flow.

## Allowed harness inputs

- domain name
- public domain policy
- public tool schemas
- conversation history visible to the agent
- model tool calls
- official domain tool outputs
- agent-visible transcript/audio events
- server state derived from the above
- `benchmark_task_id` as trace/posthoc metadata only

## Forbidden harness inputs

- hidden task objective
- task ID as a rule selector
- expected final database state
- evaluator output
- reward/failure signal
- user simulator private plan
- clean simulator text not available to the voice agent
- failed-task-only retries for reported final scores

## Forbidden implementation changes

Do not modify:

- task files
- domain policy files
- domain tools
- user simulator prompts
- evaluator code
- scoring code
- database final-state comparison logic

## Validator constraints

The validator may:

- block a side-effecting domain tool call;
- return a corrective packet to the model;
- request clarification or confirmation;
- log the event.

The validator must not:

- mutate the benchmark database;
- call domain tools itself unless the original model tool call has passed validation;
- infer the expected answer from hidden task data;
- use task ID to choose special-case logic.

## Identifier naming

Use `benchmark_task_id` only for trace and posthoc metadata that identifies the
benchmark task being run. StageGate validator, ledger, and stage-packet control
logic must never use `task_id` as a domain identifier or rule selector. Domain
identifiers must use unambiguous domain-specific names such as `account_id`,
`reservation_id`, `order_id`, or the mock-domain internal alias
`service_task_ref`.

## Required leakage tests

Add tests with names like:

- `test_stagegate_does_not_read_task_objective`
- `test_stagegate_does_not_read_expected_final_db`
- `test_stagegate_does_not_read_user_simulator_private_state`
- `test_stagegate_does_not_read_evaluator_result`
- `test_stagegate_does_not_route_by_task_id`
- `test_validator_never_mutates_domain_state_when_blocking`

## Final run hygiene

A final result is invalid if it used:

- `--num-tasks`
- `--task-ids`
- `control` speech complexity
- different timeouts by condition
- different seeds by condition
- different concurrency by condition
- different code commits by condition
- reruns of only failed tasks

Final reported runs must pass `scripts/stagegate_final_run_hygiene.py` against a
manifest covering baseline, stage_only, and stagegate for retail, airline, and
telecom. The manifest must prove regular speech complexity and matching model,
provider, reasoning effort, timeout, seed, and concurrency across conditions.

## Reporting language

Correct:

> StageGate is a custom scaffold evaluated on τ-Voice.

Incorrect:

> GPT-Realtime-2 achieved this score as a standard model-only result.
