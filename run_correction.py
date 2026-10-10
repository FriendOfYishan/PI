"""CPU-only PI correction-loop benchmark. All scores arise from execution.

Policy inputs contain public requests, anonymous tool action IDs, observed
traces and assertion explanations. Hidden registries are used only by tools,
demonstrators and the independent evaluator, never read by policy code.
"""
from __future__ import annotations
import copy
import csv
import hashlib
import itertools
import json
import random
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
STAGES = ('population', 'revision', 'amount')
ACTION_IDS = {'population': ('F0', 'F1'), 'revision': ('D0', 'D1', 'D2'),
              'amount': ('N0', 'N1', 'N2')}
EFFECTS = {'population': ('all', 'settled'), 'revision': ('all', 'oldest', 'latest'),
           'amount': ('yuan', 'cent', 'hundred_yuan')}
DEFAULT_GOALS = {'population': 'all', 'revision': 'all', 'amount': 'yuan'}
DEFAULT_PROGRAM = {stage: ACTION_IDS[stage][0] for stage in STAGES}
SEEDS = [11, 29, 47, 61, 83, 101, 127, 149, 173, 197]
METHODS = ['trajectory_replay', 'pi_full', 'pi_no_localization', 'pi_no_goal',
           'pi_no_trial', 'pi_no_incentive', 'compositional_repair', 'self_trial_only']
ASSERTIONS = {
    'population_constraint': {'path': 'selected_record_ids',
        'explanation': "Included records violate the report's requested population."},
    'revision_constraint': {'path': 'resolved_record_ids',
        'explanation': 'Chosen revisions violate the requested identity/revision rule.'},
    'value_constraint': {'path': 'normalized_amounts',
        'explanation': 'Displayed amounts do not preserve value under the declared source denomination.'}}
# Public typed pipeline data lineage, not generator-provided fault locations.
LINEAGE = {'selected_record_ids': 'population', 'resolved_record_ids': 'revision',
           'normalized_amounts': 'amount'}


def stable_int(value):
    return int(hashlib.sha256(value.encode()).hexdigest()[:16], 16)


@dataclass
class Task:
    task_id: str
    group: str
    tool_revision: str
    request: dict
    roles: dict
    rows: list


def make_task(seed, task_id, group, revision, request, shifted=False):
    r = random.Random(stable_int(f'{seed}|{task_id}|rows'))
    roles = {name: f'{name}_{r.randrange(10**9):09d}' for name in
             ('amount', 'order', 'settled', 'updated', 'row_id')}
    rows = []
    for order in range(8 if shifted else 6):
        for version in range(3 if shifted else 2):
            rows.append({roles['amount']: r.randrange(100, 20000) + version,
                         roles['order']: f'order{order}',
                         roles['settled']: (order + version) % 2 == 0 if shifted else order % 2 == 0,
                         roles['updated']: version,
                         roles['row_id']: f'{task_id}:order{order}:v{version}'})
    r.shuffle(rows)
    return Task(task_id, group, revision, dict(request), roles, rows)


def atomic_requests():
    requests = [dict(DEFAULT_GOALS)]
    for stage in STAGES:
        for goal in EFFECTS[stage]:
            if goal != DEFAULT_GOALS[stage]:
                request = dict(DEFAULT_GOALS); request[stage] = goal; requests.append(request)
    assert len(requests) == 6
    return requests


def acquisition_tasks(seed):
    specs = [(revision, request) for revision in ('A', 'B') for request in atomic_requests()] * 8
    random.Random(seed + 1000).shuffle(specs)
    return [make_task(seed, f'learn-{i:03}', 'atomic_acquisition', rev, req)
            for i, (rev, req) in enumerate(specs)]


def evaluation_tasks(seed, prefix='test', size=240):
    r = random.Random(seed + (3000 if prefix == 'test' else 4000))
    compositions = [dict(zip(STAGES, values)) for values in itertools.product(*(EFFECTS[s] for s in STAGES))
                    if sum(a != DEFAULT_GOALS[s] for s, a in zip(STAGES, values)) >= 2]
    tasks = []
    for i in range(size):
        group = ('atomic_new', 'heldout_composition', 'schema_shift', 'new_tool_revision')[i // (size // 4)]
        rev = 'C' if group == 'new_tool_revision' else r.choice(('A', 'B'))
        request = r.choice(atomic_requests() if group == 'atomic_new' else compositions)
        tasks.append(make_task(seed, f'{prefix}-{i:03}', group, rev, request, shifted=group == 'schema_shift'))
    return tasks


def micro_task(task):
    """Same public microcase for all methods; two complete order histories."""
    order = task.roles['order']
    chosen = sorted({row[order] for row in task.rows})[:2]
    return Task(task.task_id + ':micro', task.group, task.tool_revision,
                dict(task.request), dict(task.roles),
                [dict(row) for row in task.rows if row[order] in chosen])


def observed_rows(task, rows):
    """Public adapter: bind varying field names to typed logical roles."""
    return [{role: row[column] for role, column in task.roles.items()} for row in rows]


def transform(rows, stage, effect):
    """Implementation of the environment's typed anonymous tool actions."""
    if stage == 'population':
        return [dict(row) for row in rows if effect == 'all' or row['settled']]
    if stage == 'revision':
        if effect == 'all': return [dict(row) for row in rows]
        ordered = sorted(rows, key=lambda row: row['updated'], reverse=effect == 'latest')
        seen, result = set(), []
        for row in ordered:
            if row['order'] not in seen: result.append(dict(row)); seen.add(row['order'])
        return result
    divisor = {'yuan': 1, 'cent': 100, 'hundred_yuan': .01}[effect]
    return [dict(row, amount=round(row['amount'] / divisor, 2)) for row in rows]


def canonical(rows):
    return sorted(rows, key=lambda row: row['row_id'])


class Environment:
    def __init__(self, seed):
        rng = random.Random(seed)
        self.registry = {}
        for revision in ('A', 'B', 'C'):
            self.registry[revision] = {}
            for stage in STAGES:
                values = list(EFFECTS[stage]); rng.shuffle(values)
                self.registry[revision][stage] = dict(zip(ACTION_IDS[stage], values))

    def expert_program(self, task):
        # Only demonstration generation and integrity tests call this method.
        return {s: next(action for action, effect in self.registry[task.tool_revision][s].items()
                        if effect == task.request[s]) for s in STAGES}

    def execute(self, task, program):
        rows = observed_rows(task, task.rows)
        trace, work = [], 0
        for stage in STAGES:
            before = rows
            effect = self.registry[task.tool_revision][stage][program[stage]]
            rows = transform(before, stage, effect)
            work += len(before)
            trace.append({'stage': stage, 'action': program[stage], 'before': before, 'after': rows})
        work += len(rows)  # aggregate / emit
        artifact = {'total_yuan': round(sum(row['amount'] for row in rows), 2),
                    'source_rows': sorted(row['row_id'] for row in rows)}
        return artifact, trace, work

    def independent_goal_rows(self, rows, stage, goal):
        """Separate reference implementation, not transform()."""
        if stage == 'population':
            return [dict(row) for row in rows if goal != 'settled' or bool(row['settled'])]
        if stage == 'revision':
            if goal == 'all': return [dict(row) for row in rows]
            buckets = defaultdict(list)
            for row in rows: buckets[row['order']].append(row)
            chooser = min if goal == 'oldest' else max
            return [dict(chooser(bucket, key=lambda x: x['updated'])) for bucket in buckets.values()]
        factor = {'yuan': 100, 'cent': 1, 'hundred_yuan': 10000}[goal]
        return [dict(row, amount=round(row['amount'] * factor / 100, 2)) for row in rows]

    def feedback(self, task, artifact, trace):
        passed = {}
        diagnostics = []
        for assertion, entry in zip(ASSERTIONS, trace):
            stage = entry['stage']
            wanted = self.independent_goal_rows(entry['before'], stage, task.request[stage])
            passed[assertion] = canonical(wanted) == canonical(entry['after'])
            if not passed[assertion]: diagnostics.append({'assertion': assertion, **ASSERTIONS[assertion]})
        reference = observed_rows(task, task.rows)
        for stage in STAGES:
            reference = self.independent_goal_rows(reference, stage, task.request[stage])
        expected = {'total_yuan': round(sum(row['amount'] for row in reference), 2),
                    'source_rows': sorted(row['row_id'] for row in reference)}
        accepted = (abs(artifact['total_yuan'] - expected['total_yuan']) < 1e-6
                    and artifact['source_rows'] == expected['source_rows'] and all(passed.values()))
        # Numeric answer / reference rows / registry are never in returned feedback.
        return {'accepted': accepted, 'passed': passed, 'diagnostics': diagnostics}

    def submit(self, task, program):
        artifact, trace, work = self.execute(task, program)
        return {'artifact': artifact, 'trace': trace, 'work': work,
                'feedback': self.feedback(task, artifact, trace)}


def demonstrations(environment, seed):
    corpus = []
    for revision in ('A', 'B'):
        for index, request in enumerate(atomic_requests()):
            for repeat in range(2):
                task = make_task(seed, f'source-{revision}-{index}-{repeat}',
                                 'external_atomic', revision, request)
                program = environment.expert_program(task)
                result = environment.submit(task, program)
                assert result['feedback']['accepted']
                corpus.append({'demo_id': task.task_id, 'task': asdict(task),
                               'program': program, 'trace': result['trace'],
                               'accepted': True})
    random.Random(seed + 2000).shuffle(corpus)
    return corpus


def infer_effect(entry):
    """Classify observed state changes. No registry or supplied correct-effect tag."""
    before, after = entry['before'], entry['after']
    before_ids = {row['row_id'] for row in before}; after_ids = {row['row_id'] for row in after}
    if entry['stage'] == 'population':
        if before_ids == after_ids: return 'all'
        if after_ids == {row['row_id'] for row in before if row['settled']}: return 'settled'
    elif entry['stage'] == 'revision':
        if before_ids == after_ids: return 'all'
        groups = defaultdict(list)
        for row in before: groups[row['order']].append(row)
        for label, choose in (('oldest', min), ('latest', max)):
            chosen = {choose(group, key=lambda x: x['updated'])['row_id'] for group in groups.values()}
            if chosen == after_ids: return label
    else:
        originals = {row['row_id']: row['amount'] for row in before}
        ratios = {round(row['amount'] / originals[row['row_id']], 8) for row in after}
        if len(ratios) == 1:
            return {1: 'yuan', .01: 'cent', 100: 'hundred_yuan'}.get(next(iter(ratios)))
    return None


def diagnosed_goals(task, feedback):
    """Symbolic explanation -> public producer scope -> requested postcondition."""
    result = []
    for diagnostic in feedback['diagnostics']:
        stage = LINEAGE[diagnostic['path']]
        result.append({'assertion': diagnostic['assertion'], 'stage': stage,
                       'goal': task.request[stage], 'explanation': diagnostic['explanation']})
    return result


def failed_count(feedback):
    return sum(not value for value in feedback['passed'].values())


class Controller:
    def __init__(self, method, corpus):
        self.method = method
        self.corpus = copy.deepcopy(corpus)
        self.templates = []
        self.fragments = []
        for demo in corpus:
            template = {'revision': demo['task']['tool_revision'], 'request': demo['task']['request'],
                        'program': demo['program'], 'demo_id': demo['demo_id']}
            if not any(t['revision'] == template['revision'] and t['request'] == template['request'] for t in self.templates):
                self.templates.append(template)
            for entry in demo['trace']:
                effect = infer_effect(entry)
                if effect is None: continue
                fragment = {'revision': demo['task']['tool_revision'], 'stage': entry['stage'],
                            'effect': effect, 'action': entry['action'], 'demo_id': demo['demo_id']}
                if not any(all(f[k] == fragment[k] for k in ('revision', 'stage', 'effect', 'action')) for f in self.fragments):
                    self.fragments.append(fragment)
        self.evidence = {}
        self.workflows = {}

    def state_hash(self):
        return hashlib.sha256(json.dumps(self.__dict__, sort_keys=True).encode()).hexdigest()

    def key(self, task, stage):
        return '|'.join((task.tool_revision, stage, task.request[stage]))

    def whole_key(self, task):
        return '|'.join([task.tool_revision] + [task.request[s] for s in STAGES])

    def candidate_evidence(self, task, stage, action):
        return self.evidence.get(self.key(task, stage), {}).get(action, {'success': 0, 'failure': 0,
                                                                        'accepted_tasks': [], 'reward': 0})

    def incentive_score(self, record):
        return (record['success'] + 1) / (record['success'] + record['failure'] + 2)

    def initial_program(self, task):
        program = dict(DEFAULT_PROGRAM)
        if self.method == 'pi_no_incentive': return program
        if self.method == 'trajectory_replay':
            record = self.workflows.get(self.whole_key(task))
            if record and len(record['tasks']) >= 2: return dict(record['program'])
            return program
        for stage in STAGES:
            options = self.evidence.get(self.key(task, stage), {})
            eligible = [(action, record) for action, record in options.items()
                        if len(record['accepted_tasks']) >= 2
                        and (self.incentive_score(record) >= .7 if self.method != 'compositional_repair' else True)]
            if eligible:
                program[stage] = max(eligible, key=lambda ar: (self.incentive_score(ar[1]), ar[1]['reward']))[0]
        return program

    def ranked_templates(self, task, stage=None, goal=None):
        def score(template):
            similarity = sum(template['request'][s] == task.request[s] for s in STAGES)
            compatible = template['revision'] == task.tool_revision
            goal_match = stage is not None and template['request'][stage] == goal
            return (4 * goal_match + 2 * compatible + similarity, template['demo_id'])
        return sorted(self.templates, key=score, reverse=True)

    def local_candidates(self, task, stage, goal):
        if self.method == 'self_trial_only': return list(ACTION_IDS[stage])
        if self.method == 'pi_no_goal':
            actions = [template['program'][stage] for template in self.ranked_templates(task)]
        else:
            candidates = [f for f in self.fragments if f['stage'] == stage and f['effect'] == goal]
            candidates.sort(key=lambda f: (f['revision'] == task.tool_revision,
                                           self.incentive_score(self.candidate_evidence(task, stage, f['action']))
                                           if self.method != 'compositional_repair' else 0,
                                           f['demo_id']), reverse=True)
            actions = [f['action'] for f in candidates]
        if self.method != 'pi_no_incentive':
            known = sorted(self.evidence.get(self.key(task, stage), {}).items(),
                           key=lambda ar: self.incentive_score(ar[1]), reverse=True)
            actions = [action for action, record in known if record['success'] > 0] + actions
        return list(dict.fromkeys(actions + list(ACTION_IDS[stage])))

    def reinforce(self, task, updates, final_program, final_feedback):
        if self.method == 'pi_no_incentive': return
        if self.method == 'trajectory_replay':
            if final_feedback['accepted']:
                key = self.whole_key(task)
                record = self.workflows.setdefault(key, {'program': dict(final_program), 'tasks': []})
                if task.task_id not in record['tasks']: record['tasks'].append(task.task_id)
            return
        for update in updates:
            stage, action = update['stage'], update['action']
            record = self.evidence.setdefault(self.key(task, stage), {}).setdefault(action,
                      {'success': 0, 'failure': 0, 'accepted_tasks': [], 'reward': 0})
            record['success' if update['passed'] else 'failure'] += 1
            record['reward'] += update['reward']
        if final_feedback['accepted']:
            for stage, action in final_program.items():
                record = self.evidence.setdefault(self.key(task, stage), {}).setdefault(action,
                          {'success': 0, 'failure': 0, 'accepted_tasks': [], 'reward': 0})
                record['success'] += 1
                record['reward'] += 1
                if task.task_id not in record['accepted_tasks']: record['accepted_tasks'].append(task.task_id)


def episode(controller, task, submit, budget=3, learn=False, detailed=False):
    program = controller.initial_program(task)
    first_program = dict(program)
    first = submit(task, program)
    current_feedback = first['feedback']
    initial_feedback = copy.deepcopy(current_feedback)
    full_work, probe_work, trials, events, updates = first['work'], 0, 0, [], []
    full_submissions = 1
    regression_commits = 0
    goals_before = diagnosed_goals(task, current_feedback)
    micro = micro_task(task)
    seen = set()
    if not current_feedback['accepted'] and controller.method == 'pi_no_trial':
        for diagnosis in goals_before:
            stage, goal = diagnosis['stage'], diagnosis['goal']
            program[stage] = controller.local_candidates(task, stage, goal)[0]
            events.append({'scope': [stage], 'goal': goal, 'action': program[stage],
                           'trial': False, 'reason': diagnosis['explanation']})
    elif not current_feedback['accepted']:
        if controller.method == 'trajectory_replay':
            best_failures = failed_count(current_feedback)
            for template in controller.ranked_templates(task):
                if trials >= budget: break
                proposal = dict(template['program'])
                signature = json.dumps(proposal, sort_keys=True)
                if signature in seen or proposal == program: continue
                seen.add(signature)
                result = submit(micro, proposal); trials += 1; probe_work += result['work']
                feedback = result['feedback']
                committed = failed_count(feedback) < best_failures
                regressions = sum(value and not feedback['passed'][assertion]
                                  for assertion, value in current_feedback['passed'].items())
                if committed:
                    program = proposal; current_feedback = feedback; best_failures = failed_count(feedback)
                    regression_commits += regressions
                events.append({'scope': list(STAGES), 'goal': 'complete_request',
                               'candidate_source': template['demo_id'], 'program': proposal,
                               'probe_passed': feedback['accepted'], 'committed': committed,
                               'regressions': regressions if committed else 0,
                               'failed_constraints': failed_count(feedback)})
                if feedback['accepted']: break
        else:
            while trials < budget:
                diagnoses = diagnosed_goals(task, current_feedback)
                if not diagnoses: break
                diagnosis = diagnoses[0]
                stage, goal = diagnosis['stage'], diagnosis['goal']
                proposals = []
                if controller.method == 'pi_no_localization':
                    for template in controller.ranked_templates(task, stage, goal):
                        proposals.append((dict(template['program']), template['demo_id']))
                else:
                    for action in controller.local_candidates(task, stage, goal):
                        proposal = dict(program); proposal[stage] = action
                        proposals.append((proposal, 'effect_fragment_or_own_trial'))
                proposed = next(((proposal, source) for proposal, source in proposals
                                 if proposal != program and
                                 (stage, json.dumps(proposal, sort_keys=True)) not in seen), None)
                if proposed is None: break
                proposal, source = proposed
                seen.add((stage, json.dumps(proposal, sort_keys=True)))
                result = submit(micro, proposal); trials += 1; probe_work += result['work']
                feedback = result['feedback']
                passed = feedback['passed'][diagnosis['assertion']]
                reward = failed_count(current_feedback) - failed_count(feedback) - .01 * result['work'] / max(1, len(micro.rows))
                # A localized pass is tentative. Full-task acceptance is required
                # for proactive future reuse of this conditioned action.
                updates.append({'stage': stage, 'action': proposal[stage], 'passed': passed,
                                'reward': reward if passed else min(0, reward)})
                regressions = sum(value and not feedback['passed'][assertion]
                                  for assertion, value in current_feedback['passed'].items())
                if passed:
                    program = proposal; current_feedback = feedback
                    regression_commits += regressions
                events.append({'scope': list(STAGES) if controller.method == 'pi_no_localization' else [stage],
                               'goal': goal, 'assertion': diagnosis['assertion'], 'reason': diagnosis['explanation'],
                               'candidate_source': source, 'program': proposal, 'probe_passed': passed,
                               'committed': passed, 'regressions': regressions if passed else 0,
                               'incentive': reward if passed else min(0, reward),
                               'failed_constraints': failed_count(feedback)})
    if not first['feedback']['accepted']:
        final = submit(task, program); full_submissions += 1; full_work += final['work']
    else:
        final = first
    final_feedback = final['feedback']
    if learn: controller.reinforce(task, updates, program, final_feedback)
    regression = any(value and not final_feedback['passed'][assertion]
                     for assertion, value in initial_feedback['passed'].items())
    assert full_submissions <= 2 and trials <= budget
    result = {'task_id': task.task_id, 'group': task.group, 'tool_revision': task.tool_revision,
              'success': int(final_feedback['accepted']), 'first_success': int(first['feedback']['accepted']),
              'initial_failed_constraints': failed_count(initial_feedback),
              'final_failed_constraints': failed_count(final_feedback),
              'small_trials': trials, 'full_submissions': full_submissions,
              'row_stage_visits': full_work + probe_work, 'probe_row_stage_visits': probe_work,
              'final_collateral_regression': int(regression),
              'collateral_regression_commits': regression_commits,
              'probe_failures': sum(e.get('probe_passed') is False for e in events),
              'accepted_incentive': sum(e.get('incentive', 0) for e in events if e.get('committed')),
              'initial_program': first_program, 'final_program': dict(program), 'events': events}
    if detailed:
        result.update(task=asdict(task), initial_response=first, final_response=final)
    return result


def frozen_batch(environment, controller, tasks, budget, seed, mode):
    before = controller.state_hash()
    rows = []
    for task in tasks:
        row = episode(controller, task, environment.submit, budget=budget)
        row.update(seed=seed, method=controller.method, mode=mode, trial_budget=budget)
        rows.append(row)
    assert controller.state_hash() == before, 'Persistent state changed during frozen evaluation'
    return rows


def summarize(rows, keys):
    grouped = defaultdict(list)
    for row in rows: grouped[tuple(row[k] for k in keys)].append(row)
    metrics = ['success', 'first_success', 'initial_failed_constraints', 'final_failed_constraints',
               'small_trials', 'full_submissions', 'row_stage_visits', 'probe_row_stage_visits',
               'final_collateral_regression', 'collateral_regression_commits', 'probe_failures', 'accepted_incentive']
    result = []
    for group, entries in sorted(grouped.items()):
        item = dict(zip(keys, group)); item['episodes'] = len(entries)
        for metric in metrics: item[metric] = statistics.mean(r[metric] for r in entries)
        result.append(item)
    return result


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, list(rows[0])); writer.writeheader(); writer.writerows(rows)


def experiment():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    final, acquisition, curves, examples, corpora, task_archive, snapshots, checks = [], [], [], [], [], [], [], []
    for seed in SEEDS:
        environment = Environment(seed)
        corpus = demonstrations(environment, seed)
        corpora.extend(dict(seed=seed, **entry) for entry in corpus)
        train = acquisition_tasks(seed); tests = evaluation_tasks(seed); probes = evaluation_tasks(seed, 'probe', 24)
        task_archive.extend({'seed': seed, **asdict(task)} for task in train + tests + probes)
        for method in METHODS:
            controller = Controller(method, corpus)
            final.extend(frozen_batch(environment, controller, tests, 3, seed, 'cold'))
            for checkpoint in (0, 24, 48, 96):
                if checkpoint:
                    previous = {24: 0, 48: 24, 96: 48}[checkpoint]
                    for task in train[previous:checkpoint]:
                        row = episode(controller, task, environment.submit, budget=3, learn=True)
                        row.update(seed=seed, method=method, mode='acquisition', trial_budget=3)
                        acquisition.append(row)
                rows = frozen_batch(environment, controller, probes, 3, seed, f'checkpoint_{checkpoint}')
                for row in rows: row['checkpoint'] = checkpoint
                curves.extend(rows)
            for budget in (1, 3, 6):
                final.extend(frozen_batch(environment, controller, tests, budget, seed, 'trained'))
            checks.append({'seed': seed, 'method': method, 'frozen_evaluation': True,
                           'condition_keys': len(controller.evidence), 'whole_workflows': len(controller.workflows)})
            if method == 'pi_full':
                snapshots.append({'seed': seed, 'evidence': controller.evidence,
                                  'fragments': controller.fragments, 'state_hash': controller.state_hash()})
                reset = Controller(method, corpus)
                reset_rows = frozen_batch(environment, reset, tests, 3, seed, 'reset_verified_state')
                final.extend(reset_rows)
            if seed == SEEDS[0] and method in ('pi_full', 'trajectory_replay', 'pi_no_localization',
                                              'pi_no_trial', 'pi_no_goal'):
                fresh = Controller(method, corpus)
                for index in (60, 61, 62, 120, 180, 181, 182):
                    row = episode(fresh, tests[index], environment.submit, budget=3, detailed=True)
                    row.update(seed=seed, method=method, mode='cold_trace_example', trial_budget=3)
                    examples.append(row)
        print(f'seed {seed}: completed', flush=True)
    write_csv(OUT/'summary.csv', summarize(final, ['mode', 'trial_budget', 'method', 'group']))
    write_csv(OUT/'per_seed_summary.csv', summarize(final, ['seed', 'mode', 'trial_budget', 'method', 'group']))
    write_csv(OUT/'learning_curves.csv', summarize(curves, ['checkpoint', 'method', 'group']))
    write_csv(OUT/'acquisition_summary.csv', summarize(acquisition, ['seed', 'method']))
    for filename, records in [('episodes.jsonl', final), ('acquisition_episodes.jsonl', acquisition),
                              ('trace_examples.jsonl', examples), ('demonstrations.jsonl', corpora),
                              ('tasks.jsonl', task_archive)]:
        with (OUT/filename).open('w', encoding='utf-8') as f:
            for record in records: f.write(json.dumps(record, ensure_ascii=False) + '\n')
    (OUT/'learned_incentives.json').write_text(json.dumps(snapshots, indent=2), encoding='utf-8')
    metrics = {'status': 'executed_cpu_correction_loop', 'seeds': SEEDS, 'methods': METHODS,
               'trial_budgets': [1, 3, 6], 'full_submission_limit': 2,
               'external_demonstrations_per_seed': 24, 'acquisition_tasks_per_seed': 96,
               'final_tasks_per_seed': 240, 'main_final_episodes': len(final),
               'acquisition_episodes': len(acquisition), 'checkpoint_episodes': len(curves),
               'detailed_example_executions': len(examples),
               'total_episodes': len(final) + len(acquisition) + len(curves) + len(examples),
               'elapsed_seconds': time.perf_counter()-start,
               'llm_calls': 0, 'trained_neural_parameters': 0,
               'state_checks': checks,
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'limitations': ['finite typed tools and symbolic goal parsing',
                 'structured instrumented assertion explanations available equally to all policies',
                 'schema adapter and executable composition grammar supplied',
                 'small fixtures identify this finite operator vocabulary; not arbitrary-program safety',
                 'reinforcement updates counts and preferences, not neural parameters',
                 'not LLM baseline reproductions or token-budget measurements']}
    (OUT/'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in metrics.items() if k not in ('state_checks','limitations')}, indent=2))


if __name__ == '__main__':
    experiment()
