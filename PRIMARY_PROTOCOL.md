# PI correction-loop pilot: specification before execution

This experiment replaces the previous gate-only interpretation. The tested loop
is: failed public assertion -> causal scope -> desired local postcondition ->
infer candidate skills from demonstrated before/after changes -> sandbox a local
patch on a small dataset -> audit the full task -> reinforce verified skills.

It uses symbolic controllers and a finite typed tool vocabulary, with no LLM,
neural training, or claim of natural-language understanding. Goal parsing and
data-lineage localization are programmed for this environment. The environment
returns structured, instrumented assertion explanations to every method. It
never returns the correct action ID or its hidden registry.

## Environment and transfer

- Three stages: population selection, identity/revision resolution, amount
  normalization; a final report contains exact totals and source lineage.
- Stage action names are opaque (F0/F1, D0/D1/D2, N0/N1/N2). Their implementations
  are independently shuffled for tool revisions A, B and C in each seed.
- The public request specifies source denomination, population policy and
  revision policy. It does not specify action IDs.
- 24 pre-existing successful demonstrations: two source tool revisions A/B,
  six atomic/default requests, two distinct source schemas. Whole compositions
  with two or three non-default goals are held out from the demonstrations and
  acquisition stream. All policies receive the same public corpus.
- Infer fragment effects from actual before/after rows, rather than attaching
  generator-provided correct skill tags to demonstrations.
- Small trials use two complete order histories, ordinarily 4 rows; larger
  schema-shift tasks use 6 rows. Every controller uses the same micro-dataset.
- Auditing independently checks the final aggregate and supporting row IDs.
  Local explanations check each stage against the public contract on its
  actual incoming rows, avoiding unlocalized downstream cascades.

## Fixed run protocol

- Seeds: 11, 29, 47, 61, 83, 101, 127, 149, 173, 197.
- 96 acquisition tasks per seed/method: versions A/B, six atomic/default goal
  combinations, eight instances each. Acquisition small-trial budget: 3.
- 240 final tasks: 60 atomic instances, 60 held-out compositions, 60 larger
  schemas with changes of payment status across revisions, 60 unseen revision C
  compositions. Same tasks for every method and cold/trained modes.
- Evaluate cold and trained controllers at budget 3. After training, also
  evaluate budgets 1 and 6. Checkpoints 0/24/48/96 use 24 separate frozen probes.
- At most two full submissions and K small trials. All trials, submissions,
  visited row-stage pairs, rejected candidates and collateral regressions are
  recorded. Controller computation / corpus parsing are not counted as tokens.
- Frozen evaluation state hashes must match before/after each batch. Episodes
  have no persistent updates during evaluation; acquisition updates are explicit.

## Controls

1. Whole-trajectory replay: retrieves complete demonstrations / successful
   workflows; can trial complete proposals using the same sandbox and feedback.
2. PI full: localize, infer a postcondition, retrieve matching fragments, locally
   trial, audit, and reinforce successful conditions/actions.
3. PI no localization: uses the same goal-based retrieval, but changes complete
   programs rather than preserving already-correct stages.
4. PI no goal abstraction: projects raw trajectory steps onto the diagnosed
   stage; does not classify before/after effects to retrieve by postcondition.
5. PI no small trial: adopts the best goal-based local patch without sandboxing,
   then makes the second full submission.
6. PI no incentive: performs the same episode-level correction, but does not
   retain reinforced action preferences or promoted verified skills across tasks.
7. Compositional repair: equally expressive local goal/fragment/trial controller,
   with simple verified mapping reuse instead of numerical incentive scores.
   This is intentionally strong: a tie prevents attributing common repair
   capabilities uniquely to the PI name or a motivational interpretation.
8. Self trial only: localization and goal inference, opaque-action enumeration
   and verified learning, without using demonstrations. The common corpus is
   available but unused by this policy; outside imitation is not necessary.

The incentive is a computational learning signal: positive reduction in failed
assertions after a sandbox patch, followed by full-task acceptance. A conditioned
action is eligible for proactive reuse after two distinct accepted tasks. PI
maintains success/failure evidence and a beta-style confidence score; the strong
repair control uses a simple accepted-action lookup. This is not neural reward
training or a demonstration of psychological motivation.

No result-dependent tuning of seeds, tasks, thresholds, budgets or stage
implementations is permitted. Report both matches to the strong repair control
and failures on new tool revisions. This pilot is not a literature-method
reproduction, fixed-token-budget experiment, or evidence of AGI.

Run `verify_correction.py`, `run_correction.py`, then `build_report.py`.
All experiment computations use the Python standard library. Plotting alone
needs matplotlib. Results live in
`output/experiments/passive_imitation/correction_pilot/`.
