"""Describe completed, independently replayed coding cases without regenerating text."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from coding.data import DATA,read_rows
from evaluation.benchmark import file_sha256
from evaluation.checkpoint import write_json_atomic
from training.experiment_state import rows_sha256

def failure_category(record):
    if record['passed']: return 'passed'
    if not record['syntax_valid']: return 'syntax'
    if record.get('error_type')=='Timeout': return 'timeout'
    if record.get('error_type')=='AssertionError': return 'test_assertion'
    return 'runtime'

def review():
    folder=ROOT/'outputs/coding-mbpp-r8-fp32/evaluation'
    audit=json.loads((folder/'independent-audit.json').read_text())
    if not audit['coverage_and_scores_verified']:
        raise ValueError('Complete the independent score audit first.')
    rows=read_rows('test')
    if rows_sha256(rows)!=json.loads((DATA/'manifest.json').read_text())['row_sha256']['test']:
        raise ValueError('Frozen HumanEval rows changed.')
    records={}; summaries={}
    for variant in ('base','tool','coding'):
        path=folder/f'{variant}-test-164-tokens512.jsonl'
        if file_sha256(path)!=audit['results'][variant+'/test']['prediction_sha256']:
            raise ValueError('Predictions changed after independent replay.')
        values=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        if len(values)!=164 or {r['test_index'] for r in values}!=set(range(164)):
            raise ValueError('Incomplete or duplicate coding coverage.')
        for record in values:
            if record['task_sha256']!=rows_sha256([rows[record['test_index']]]):
                raise ValueError('Saved coding references changed.')
        records[variant]={r['test_index']:r for r in values}
        summaries[variant]=json.loads(path.with_name(path.stem+'-summary.json').read_text())
        if summaries[variant]['passed']!=sum(r['passed'] for r in values):
            raise ValueError('Summary scores differ from audited predictions.')
    cases=[]; transitions=Counter(); categories=Counter()
    for index,row in enumerate(rows):
        base,coding=records['base'][index],records['coding'][index]
        transition=('both_passed' if base['passed'] and coding['passed'] else
                    'coding_only_passed' if coding['passed'] else
                    'base_only_passed' if base['passed'] else 'neither_passed')
        category=failure_category(coding)
        transitions[transition]+=1; categories[category]+=1
        cases.append({'test_index':index,'task_id':row['task_id'],'prompt':row['prompt'],
            'reference_completion':row['completion'],'entry_point':row['entry_point'],
            'base_passed':base['passed'],'coding_passed':coding['passed'],
            'tool_passed':records['tool'][index]['passed'],'transition':transition,'category':category,
            'base_completion':base['completion'],'coding_completion':coding['completion'],
            'base_output':base['output_text'],'coding_output':coding['output_text'],
            'base_error':base.get('error',''),'coding_error':coding.get('error',''),
            'coding_error_type':coding.get('error_type'),'coding_hit_token_limit':coding['hit_token_limit']})
    result={'coverage_and_replay_verified':True,'examples':164,'summaries':summaries,
            'paired_transitions':dict(transitions),'coding_categories':dict(categories),
            'coding_failures':164-summaries['coding']['passed'],'cases':cases,
            'conclusion':'This pilot did not beat Base on held-out HumanEval. Keep the Coding adapter experimental; do not tune or train on these official cases.'}
    write_json_atomic(folder/'test-case-review.json',result)
    lines=['# Coding pilot: paired HumanEval case review','',
           'All164 cases per mode match the saved independent Docker replay audit. No model text was regenerated.','',
           '| Mode | Passed | Syntax valid | Token-limit hits |','|---|---:|---:|---:|']
    for variant,summary in summaries.items():
        values=list(records[variant].values())
        lines.append(f"| {variant} | {summary['passed']}/164 ({summary['pass_at_1']:.2%}) | {sum(r['syntax_valid'] for r in values)}/164 | {sum(r['hit_token_limit'] for r in values)}/164 |")
    lines += ['',f"Coding gained {transitions['coding_only_passed']} cases and lost {transitions['base_only_passed']} versus Base; net {transitions['coding_only_passed']-transitions['base_only_passed']:+d}.",
              result['conclusion'],'A one-case difference on one seed is not evidence of a general ordering.',
              'Validation19/32 versus Base17/32 did not translate into a held-out pass-rate gain.',
              'Syntax, execution, and task-test passing are different properties; passing finite tests is not proof for every input.','',
              '| Coding result category | Cases |','|---|---:|']
    for name,count in categories.items(): lines.append(f'| {name} | {count} |')
    lines += ['', 'Full paired programs, reference completions and errors: test-case-review.json.',
              'Manual adapter switching can be verified independently of whether this Coding pilot improves accuracy.',
              'Further training uses training/validation data and a separate output; keep this held-out report frozen.','']
    (folder/'test-case-review.md').write_text('\n'.join(lines),encoding='utf-8')
    return result

if __name__=='__main__':
    value=review()
    print(json.dumps({k:value[k] for k in ('examples','paired_transitions','coding_categories','coding_failures','conclusion')},indent=2))
