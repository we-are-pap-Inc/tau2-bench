# Sierra Submission Notes

## Submission type

Baseline:

- `submission_type: standard` only if no prompt/scaffold/tool/orchestration changes are made.

StageOnly and StageGate:

- `submission_type: custom`
- disclose modified prompts and scaffold changes
- include code and methodology references

## Suggested custom submission metadata

    {
      "model_name": "gpt-realtime-2 + StageGate",
      "model_organization": "OpenAI",
      "submitting_organization": "YOUR_ORG",
      "submission_type": "custom",
      "modality": "voice",
      "reasoning_effort": "high",
      "methodology": {
        "notes": "Custom realtime voice-agent scaffold. Stable identity and global rules remain in session instructions. Procedural workflow is delivered through advance_stage tool outputs. A typed entity ledger tracks exact identifiers and confirmation state. A pre-write validator blocks side-effecting domain tool calls when identity, exact entities, policy preconditions, or user confirmation are missing. No benchmark tasks, evaluator logic, user simulator, domain policies, or domain tools were modified. The harness did not read hidden task objectives, expected final database state, user simulator private state, or evaluator outputs.",
        "verification": {
          "modified_prompts": true,
          "omitted_questions": false,
          "details": "Custom scaffold added staged orchestration, entity ledger, and pre-write validation. All final voice runs used regular speech complexity and all tasks in retail, airline, and telecom."
        }
      }
    }

## Prepare command

    uv run tau2 submit prepare \
      data/simulations/<retail_dir> \
      data/simulations/<airline_dir> \
      data/simulations/<telecom_dir> \
      --output external_upload/submission_stagegate \
      --voice

    uv run tau2 submit validate external_upload/submission_stagegate

## PR package

Commit only the expected public submission files to the leaderboard path. Provide external artifact links for trajectory/audio files if required by maintainers.

PR should state:

- custom voice scaffold
- model/provider
- domains
- speech complexity
- tick duration
- max steps seconds
- max concurrency
- seed
- task filters: none
- code commit
- artifacts link
- methodology note
- validity note

## Public writeup headline

Use:

> StageGate: staged instruction delivery for full-duplex voice agents on τ-Voice

Do not use:

> We beat τ-Voice with GPT-Realtime-2

unless the score genuinely beats the public top and the custom nature is still disclosed.
