# Project Brief — StageGate on τ³ / τ-Voice

## Project name

StageGate: staged instruction delivery with entity ledger and pre-write validation for realtime voice agents.

## Benchmark target

Use Sierra's current τ³-bench repository, currently named `sierra-research/tau2-bench`, and focus on the τ-Voice/full-duplex audio-native track.

Primary domains:

- retail
- airline
- telecom

Do not make `banking_knowledge` the first target; it is relevant to τ³ overall but not to this realtime voice harness.

## Core hypothesis

Realtime voice agents fail not only because the model is weak, but because long, noisy, interruptible conversations degrade procedural control, exact entity tracking, and safe tool timing.

StageGate tests whether task completion improves when:

1. persistent identity and global rules remain in session instructions,
2. procedural workflow is delivered through stage-local tool outputs,
3. a typed entity ledger tracks exact values and confirmation state, and
4. a pre-write validator blocks side-effecting tool calls until required facts, policy checks, and user confirmation are present.

## Experimental conditions

- V0 baseline: official OpenAI audio-native GPT-Realtime-2 path, unmodified.
- V1 StageOnly: add `advance_stage` and stage packets.
- V2 StageGate: StageOnly plus entity ledger and pre-write validator.

Optional later ablations:

- V3 StageGate without validator.
- V4 StageGate without ledger.
- V5 extreme staged prompt with almost all workflow in tool outputs.

Do not start with V5. The extreme variant is interesting but likely brittle.

## Primary metric

Pass@1 by domain and mean across retail, airline, and telecom.

The most important analysis is paired task-level delta:

- baseline-only wins
- StageOnly-only wins
- StageGate-only wins
- both pass
- both fail
- net delta by domain

## Final run constants

Use the same constants for all conditions:

- model: `gpt-realtime-2`
- reasoning effort: `high`
- speech complexity: `regular`
- tick duration: `0.2`
- max steps seconds: `1200`
- max concurrency: `1`
- seed: `300`
- task split: base/default
- task filters: none

## Final public claim

Do not claim a standard model score for StageGate.

Correct claim:

> We evaluated a custom realtime voice-agent scaffold on τ-Voice. Holding model, benchmark, tasks, speech complexity, timing, and evaluator fixed, StageGate changed only the scaffold and produced a paired task-level delta of X.

## What success looks like

Weak but useful:

- StageGate improves less than 2 percentage points but exposes a clear failure taxonomy.

Strong:

- StageGate improves at least 2-5 percentage points, especially by reducing exact-entity, missing-confirmation, and premature-write failures.

Very strong:

- StageGate improves at least 5 percentage points in at least two domains, with trace evidence and no benchmark leakage.
