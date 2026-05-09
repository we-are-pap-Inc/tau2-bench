# Research Thesis

## Thesis

StageGate is built on the claim that voice-agent reliability is an orchestration problem as much as a model problem.

In a long full-duplex support call, the model must simultaneously handle speech, turn-taking, ambiguous audio, user corrections, policy reasoning, exact identifiers, tool calls, and state-changing actions. A giant always-on prompt is a weak way to control this. StageGate instead keeps permanent identity/global rules persistent and delivers the current procedural step through tool outputs at the moment it matters.

## Empirical motivation

### τ-Voice identifies real agent-side failures

The τ-Voice paper reports 278 tasks across retail, airline, and telecom and finds that voice agents retain only a fraction of strong text-agent performance under realistic conditions. The paper also reports that qualitative analysis attributes 79-90% of failures to agent behavior rather than simulator artifacts.

Implication: there is room for a custom scaffold that targets agent-side failures such as policy-order mistakes, entity drift, malformed tool calls, and premature write actions.

### Realtime prompting guidance supports simple prompts plus targeted rules

OpenAI's realtime prompting guide says `gpt-realtime-2` is intended for stronger reasoning, tool selection, exact entity handling, and long-session state. The same guide says to start simple, avoid over-prompting, define responsibilities and tool-calling behavior precisely, and set confirmation boundaries before write actions.

Implication: StageGate should not be a massive prompt. It should be a compact prompt plus targeted stage packets and validator logic.

### Long-context research supports stage-local instruction salience

The Lost in the Middle result shows that models often use information best when relevant information is at the beginning or end of context and can degrade when important facts are buried in the middle.

Implication: a stage packet near the conversational frontier may be more reliable than burying every procedural rule in one large initial prompt.

### ReAct supports action-observation loops

ReAct showed that interleaving reasoning and actions improves interactive decision-making and makes trajectories more interpretable.

Implication: StageGate should make the action-observation-control loop explicit: stage packet, user interaction, tool call, tool observation, ledger update, next stage.

### Dialogue-state tracking supports the entity ledger

Task-oriented dialogue systems traditionally track user goals and slot values over time. StageGate's entity ledger is a typed, auditable dialogue-state tracker for exact fields in τ-Voice.

Implication: do not store entity state as an unstructured conversation summary. Store it as typed slots with value, status, confidence, source, evidence, and last-updated turn.

## The simplest empirical experiment

Run three conditions:

1. Baseline GPT-Realtime-2.
2. StageOnly: staged packets only.
3. StageGate: staged packets plus entity ledger plus pre-write validator.

Hold everything else constant.

Publish:

- domain-level Pass@1
- paired task-level deltas
- failure taxonomy
- trace examples showing baseline fail / StageGate pass and baseline pass / StageGate fail

## Why StageGate, not extreme staged prompting

The extreme approach moves almost all workflow instructions into tool outputs. It is clean as an ablation but brittle as a primary bet. The model may fail to request the next packet when it needs it, and the persistent prompt may become too thin to preserve global invariants.

The best first contribution is hybrid:

- persistent rules for identity, domain policy, exactness, and confirmation boundaries;
- staged tool outputs for the current procedural step;
- ledger and validator to enforce exactness and safety around tool use.

## Reference URLs

- τ³ / τ-Voice repo: https://github.com/sierra-research/tau2-bench
- τ-Voice paper: https://arxiv.org/abs/2603.13686
- Sierra τ-Voice blog: https://sierra.ai/blog/tau-voice-benchmarking-real-time-voice-agents-on-real-world-tasks
- OpenAI realtime prompting guide: https://developers.openai.com/api/docs/guides/realtime-models-prompting
- Realtime conversations/tool outputs: https://developers.openai.com/api/docs/guides/realtime-conversations
- Lost in the Middle: https://arxiv.org/abs/2307.03172
- ReAct: https://arxiv.org/abs/2210.03629
