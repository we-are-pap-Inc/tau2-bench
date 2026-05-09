# Experiment Protocol

## Goal

Measure whether StageGate improves GPT-Realtime-2 task completion on τ-Voice while holding model and benchmark conditions constant.

## Conditions

Run three final conditions:

1. `baseline`
2. `stage_only`
3. `stagegate`

## Domains

Run all three voice domains:

- retail
- airline
- telecom

## Final constants

    MODEL=gpt-realtime-2
    EFFORT=high
    SPEECH=regular
    TICK=0.2
    MAXSEC=1200
    MAX_CONCURRENCY=1
    SEED=300

## Final command shape

    uv run tau2 run \
      --domain "$DOMAIN" \
      --audio-native \
      --audio-native-provider openai \
      --audio-native-model "$MODEL" \
      --reasoning-effort "$EFFORT" \
      --speech-complexity "$SPEECH" \
      --tick-duration "$TICK" \
      --max-steps-seconds "$MAXSEC" \
      --max-concurrency "$MAX_CONCURRENCY" \
      --seed "$SEED" \
      --verbose-logs \
      --auto-resume \
      --save-to "$SAVE_NAME"

Do not use `--num-tasks` or `--task-ids` in final runs.

## Development-only smoke tests

Smoke tests may use:

- `--speech-complexity control`
- `--num-tasks 1`
- `--num-tasks 5`
- `--audio-taps`

Never report smoke-test results as final results.

## Final job matrix

    baseline   × retail
    baseline   × airline
    baseline   × telecom
    stage_only × retail
    stage_only × airline
    stage_only × telecom
    stagegate  × retail
    stagegate  × airline
    stagegate  × telecom

## Required outputs

- `summary_by_condition_domain.csv`
- `per_task_outcomes.csv`
- `paired_delta_by_domain.csv`
- `failure_taxonomy.csv`
- `trace_events.jsonl`
- `stagegate_events_summary.csv`
- Sierra `submission.json` packages

## Primary analysis table

    Condition        Retail   Airline   Telecom   Mean   Delta vs baseline
    baseline         X.X      X.X       X.X       X.X    —
    stage_only       Y.Y      Y.Y       Y.Y       Y.Y    Y-X
    stagegate        Z.Z      Z.Z       Z.Z       Z.Z    Z-X

## Paired delta table

    Domain    Baseline-only wins   StageGate-only wins   Both pass   Both fail   Net StageGate delta
    retail    ...                  ...                   ...         ...         ...
    airline   ...                  ...                   ...         ...         ...
    telecom   ...                  ...                   ...         ...         ...

## Manual failure analysis categories

- identity/authentication failure
- exact entity capture failure
- wrong policy branch
- missing confirmation
- premature write/action tool
- malformed tool arguments
- correction ignored
- barge-in/interruption recovery failure
- timeout/loop
- correct tools but wrong final response
- simulator/transcription issue
- unknown

## Evidence standard

For any claim of improvement, include at least:

- one baseline-failed / StageGate-passed trace example per improved domain;
- one baseline-passed / StageGate-failed trace example if any regressions occur;
- counts of validator blocks by reason;
- counts of ledger corrections and confirmations;
- total stage calls per successful and failed task.
