# StageGate Results Report Template

## Claim

StageGate is a custom realtime voice-agent scaffold evaluated on τ-Voice. The evaluator, tasks, speech complexity, timing, model, and domain tools were held constant across conditions.

## Run metadata

- Repo commit:
- Model:
- Reasoning effort:
- Speech complexity:
- Tick duration:
- Max steps seconds:
- Max concurrency:
- Seed:
- Domains:
- Task filters used in final runs: none

## Main result

| Condition | Retail | Airline | Telecom | Mean | Delta vs baseline |
|---|---:|---:|---:|---:|---:|
| Baseline |  |  |  |  | — |
| StageOnly |  |  |  |  |  |
| StageGate |  |  |  |  |  |

## Paired task delta

| Domain | Baseline-only wins | StageGate-only wins | Both pass | Both fail | Net StageGate delta |
|---|---:|---:|---:|---:|---:|
| Retail |  |  |  |  |  |
| Airline |  |  |  |  |  |  |
| Telecom |  |  |  |  |  |

## Trace-derived mechanism evidence

| Condition | Domain | Stage calls/task | Ledger updates/task | Validator blocks/task | Most common block reason |
|---|---|---:|---:|---:|---|
| StageOnly | Retail |  |  |  |  |
| StageGate | Retail |  |  |  |  |

## Failure taxonomy

| Failure category | Baseline count | StageGate count | Delta | Notes |
|---|---:|---:|---:|---|
| identity/authentication failure |  |  |  |  |
| exact entity capture failure |  |  |  |  |
| missing confirmation |  |  |  |  |
| premature write/action tool |  |  |  |  |

## Example trace: baseline failed, StageGate passed

Task:
Domain:
Baseline failure:
StageGate mechanism:
Trace excerpt:

## Example trace: baseline passed, StageGate failed

Task:
Domain:
Regression:
Trace excerpt:

## Limitations

- Custom scaffold, not standard model-only result.
- Local voice personas may differ from Sierra's official parity setup.
- Pass@1 variance remains possible without multiple trials.
- StageGate may trade latency/tool-call overhead for safety.

## Submission language

Use custom-submission language and disclose modified prompts/orchestration.
