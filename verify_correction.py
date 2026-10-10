"""Integrity checks performed before inspecting controller scores."""
import copy
import itertools
import json
from collections import Counter
from dataclasses import fields
import run_correction as p


def verify():
    task_count = program_count = 0
    for seed in p.SEEDS:
        env = p.Environment(seed)
        train, tests, probes = p.acquisition_tasks(seed), p.evaluation_tasks(seed), p.evaluation_tasks(seed, 'probe', 24)
        corpus = p.demonstrations(env, seed)
        assert len(train) == 96 and len(corpus) == 24
        assert set(Counter((t.tool_revision, tuple(t.request[s] for s in p.STAGES)) for t in train).values()) == {8}
        assert set(Counter(t.group for t in tests).values()) == {60}
        assert all(sum(t.request[s] != p.DEFAULT_GOALS[s] for s in p.STAGES) <= 1 for t in train)
        assert all(sum(d['task']['request'][s] != p.DEFAULT_GOALS[s] for s in p.STAGES) <= 1 for d in corpus)
        sets = [{t.task_id for t in train}, {t.task_id for t in tests}, {t.task_id for t in probes},
                {d['demo_id'] for d in corpus}]
        assert sum(map(len, sets)) == len(set.union(*sets))
        for demo in corpus:
            for entry in demo['trace']:
                inferred = p.infer_effect(entry)
                actual = env.registry[demo['task']['tool_revision']][entry['stage']][entry['action']]
                assert inferred == actual
        for task in train + tests + probes:
            expert = env.expert_program(task)
            golden = env.submit(task, expert)
            assert golden['feedback']['accepted']
            bad = dict(golden['artifact'], total_yuan=golden['artifact']['total_yuan'] + 1)
            assert not env.feedback(task, bad, golden['trace'])['accepted']
            bad = dict(golden['artifact'], source_rows=golden['artifact']['source_rows'][:-1])
            assert not env.feedback(task, bad, golden['trace'])['accepted']
            accepted_programs = []
            for actions in itertools.product(*(p.ACTION_IDS[s] for s in p.STAGES)):
                program = dict(zip(p.STAGES, actions))
                full = env.submit(task, program)
                micro = env.submit(p.micro_task(task), program)
                # In this deliberately finite vocabulary, the shared fixture
                # must separate actions exactly. This is NOT a claim that small
                # tests certify arbitrary programs.
                assert full['feedback']['passed'] == micro['feedback']['passed'], (seed, task.task_id, program)
                if full['feedback']['accepted']: accepted_programs.append(program)
                assert set(full['feedback']) == {'accepted', 'passed', 'diagnostics'}
                assert all('action' not in d and 'stage' not in d for d in full['feedback']['diagnostics'])
                program_count += 1
            assert accepted_programs == [expert]
            task_count += 1
        for method in p.METHODS:
            controller = p.Controller(method, corpus)
            assert len(controller.fragments) == 16
            original = controller.state_hash()
            results = p.frozen_batch(env, controller, tests[:8] + tests[-8:], 3, seed, 'integrity')
            assert controller.state_hash() == original
            assert all(r['small_trials'] <= 3 and r['full_submissions'] <= 2 for r in results)
            for result in results:
                if method not in ('pi_no_localization', 'trajectory_replay'):
                    assert all(len(e['scope']) == 1 for e in result['events'])
                if method == 'pi_no_trial': assert result['small_trials'] == 0
            for task in train[:8]:
                p.episode(controller, task, env.submit, 3, learn=True)
            if method == 'pi_no_incentive':
                assert controller.evidence == {} and controller.workflows == {}
        # Frozen evaluation does not reinforce skills; acquisition does.
        learner = p.Controller('pi_full', corpus)
        before = learner.state_hash()
        for task in train:
            p.episode(learner, task, env.submit, 3, learn=True)
        assert learner.state_hash() != before
        assert learner.evidence
    assert {f.name for f in fields(p.Task)} == {
        'task_id', 'group', 'tool_revision', 'request', 'roles', 'rows'}
    result = {'status': 'passed', 'generated_tasks_checked': task_count,
              'full_micro_program_comparisons': program_count,
              'seeds': p.SEEDS,
              'checks': ['independent evaluator and executor agreement',
                         'wrong aggregate/provenance rejected',
                         'unique identifying program for each task',
                         'micro fixture agrees with full assertions for every anonymous program',
                         'demo fragment effects inferred from actual state changes',
                         'held-out multi-goal compositions',
                         'disjoint task/example identifiers',
                         'same budgets and public feedback',
                         'feedback omits correct action and hidden fault stage',
                         'one-stage patch scope except explicit no-scope and trajectory controls',
                         'frozen evaluation hashes',
                         'no-incentive control retains no verified action evidence']}
    p.OUT.mkdir(parents=True, exist_ok=True)
    (p.OUT/'integrity_checks.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    verify()
