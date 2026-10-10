# PI: Passive Imitation

Reproducible CPU-only experiments for **Imitation Holds a Key to Evolution: Localized Repair and Verified Skill Acquisition for Adaptive Agents**, by Kai Hu, Beijing International Studies University.

This repository studies an architectural hypothesis: turn an observed discrepancy into a localized goal, select a candidate skill from observed effects or self-generated trials, test a small patch, audit the complete task, and retain conditioned evidence for future reuse.

## Run

Python 3.10 or newer; the experiments require only the standard library. Tested with Python 3.12. No GPU, model download, API key, or paid service is required.

```sh
python verify_correction.py
python run_correction.py
python run_online_adaptation.py
```

Run the primary experiment before the online follow-up. Output is written into `results/` beside the scripts. The primary run generates approximately 95,000 episodes, including acquisition, frozen evaluations, checkpoints, and trace examples. Raw JSONL logs are regenerated locally; compact result tables are included in the release archive.

## What Is Actually Implemented

The environment is a finite, typed, three-stage reporting workflow: population selection, revision resolution, and amount normalization. Each simulated tool revision independently permutes opaque action identifiers. Requests, schema-role bindings, pipeline lineage, and instrumented constraint feedback are supplied. Goal extraction and effect inference are symbolic rules; no language model or neural training is used.

The PI loop applies single-stage patches, tests them on two complete order histories, and performs a final full-task audit. During acquisition, it updates action evidence conditioned on tool revision, stage, and goal. Proactive reuse requires two distinct accepted tasks and a heuristic score of at least 0.7. The score is not a calibrated probability.

## Main Findings

Results below are means over ten fixed simulated worlds, with 60 final tasks per group per world and a budget of three small trials.

| Setting | PI | Whole-trajectory control | Strong compositional repair |
|---|---:|---:|---:|
| Cold held-out compositions: final success | 100.0% | 6.17% | 100.0% |
| Learned compositions: first-submission success | 100.0% | 6.17% | 100.0% |
| Unseen tool revision: final success | 70.0% | 22.67% | 70.0% |

On the unseen tool revision, self-trial enumeration reaches 79.0%, exceeding PI. Removing small trials yields 24.0%. After an exploratory follow-up with 24 separate acquisition tasks from that revision, PI and the strong control both reach 100% first-submission success. That last result measures retention in a learned revision, not zero-shot transfer.

**The evidence supports this correction-and-retention mechanism in the constructed environment, not a unique advantage of the incentive formula.** PI ties the strong compositional control. The benchmark intentionally withholds complete multi-goal demonstration combinations, making a whole-trajectory-only control weak. These are not reproductions of ReAct, Reflexion, ExpeL, or Agent Workflow Memory. Row-stage visits are an execution-work proxy, not token usage. No claim of AGI, real-world reliability, fixed-memory optimality, or LLM improvement is established.

## Files

- `run_correction.py`: environment, controllers, primary experiment, and logging.
- `verify_correction.py`: exhaustive finite-vocabulary integrity checks, separation of data splits, and frozen-state checks.
- `run_online_adaptation.py`: explicitly exploratory follow-up on a new tool revision.
- `PRIMARY_PROTOCOL.md`: the original local protocol; it was not publicly preregistered.
- `INSTRUMENTATION.md`: correction of a logging-only collateral-regression counter.
- `results-release.zip`: compact tables, execution metadata, and one complete correction example.

The public scripts preserve the original controller logic. Only the output directory was made portable. A release check compares deterministic tables against the original experiment outputs. Complete generated logs can be recreated with the commands above.

Contact: Kai Hu, 2025110321@stu.bisu.edu.cn.

Publication status: research preprint in preparation; no ICML acceptance is claimed.
