"""Exploratory online adaptation, declared after the primary frozen pilot.

Frozen unseen-revision evaluation cannot measure learning across new-revision
tasks. Here we allow 24 separate C acquisition tasks, then freeze again at
0/6/12/24 exposures. Same original seeds, K=3, operators and final C tasks.
No score-dependent change to the primary experiment.
"""
import itertools
import json
import random
import time
import run_correction as p


def adaptation_tasks(seed):
    requests = [dict(zip(p.STAGES, choices))
                for choices in itertools.product(*(p.EFFECTS[s] for s in p.STAGES))
                if sum(choice != p.DEFAULT_GOALS[s] for s, choice in zip(p.STAGES, choices)) >= 2]
    assert len(requests) == 12
    requests = requests * 2
    random.Random(seed + 5000).shuffle(requests)
    return [p.make_task(seed, f'adapt-C-{i:03}', 'online_C_acquisition', 'C', request)
            for i, request in enumerate(requests)]


def experiment():
    started = time.perf_counter()
    results, acquisition, archived_tasks = [], [], []
    methods = ['pi_full', 'pi_no_incentive', 'self_trial_only', 'compositional_repair']
    warmup_episodes = 0
    for seed in p.SEEDS:
        env = p.Environment(seed)
        corpus = p.demonstrations(env, seed)
        stream = adaptation_tasks(seed)
        tests = [task for task in p.evaluation_tasks(seed) if task.group == 'new_tool_revision']
        assert not {t.task_id for t in stream}.intersection(t.task_id for t in tests)
        archived_tasks.extend({'seed': seed, **p.asdict(task)} for task in stream)
        for method in methods:
            controller = p.Controller(method, corpus)
            for task in p.acquisition_tasks(seed):
                p.episode(controller, task, env.submit, budget=3, learn=True)
                warmup_episodes += 1
            previous = 0
            for checkpoint in (0, 6, 12, 24):
                for task in stream[previous:checkpoint]:
                    row = p.episode(controller, task, env.submit, budget=3, learn=True)
                    row.update(seed=seed, method=method, mode='online_C_acquisition', trial_budget=3)
                    acquisition.append(row)
                rows = p.frozen_batch(env, controller, tests, 3, seed, 'after_online_C')
                for row in rows: row['checkpoint'] = checkpoint
                results.extend(rows)
                previous = checkpoint
    p.write_csv(p.OUT/'online_adaptation_summary.csv', p.summarize(results, ['checkpoint', 'method', 'group']))
    p.write_csv(p.OUT/'online_adaptation_per_seed.csv', p.summarize(results, ['seed', 'checkpoint', 'method', 'group']))
    p.write_csv(p.OUT/'online_adaptation_acquisition.csv', p.summarize(acquisition, ['seed', 'method']))
    for filename, records in [('online_adaptation_episodes.jsonl', results),
                              ('online_adaptation_tasks.jsonl', archived_tasks),
                              ('online_adaptation_learning.jsonl', acquisition)]:
        with (p.OUT/filename).open('w', encoding='utf-8') as f:
            for row in records: f.write(json.dumps(row) + '\n')
    metrics = {'status': 'executed_exploratory_online_adaptation',
               'declared_after_primary_scores': True,
               'primary_protocol_unchanged': True,
               'new_C_acquisition_tasks_per_seed': 24, 'trial_budget': 3,
               'checkpoints': [0, 6, 12, 24], 'methods': methods,
               'warmup_episodes': warmup_episodes, 'adaptation_episodes': len(acquisition),
               'frozen_evaluation_episodes': len(results),
               'total_episodes': warmup_episodes + len(acquisition) + len(results),
               'elapsed_seconds': time.perf_counter() - started,
               'llm_calls': 0,
               'limitation': 'After C acquisition, this is retention in a learned C environment, not zero-shot transfer to an unseen tool revision.'}
    (p.OUT/'online_adaptation_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    experiment()
