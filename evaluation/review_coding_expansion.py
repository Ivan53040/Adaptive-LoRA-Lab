"""Review audited MBPP predictions using CPU-only file reads; never load a model."""
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from coding.expansion_data import read_rows,verify_splits
from evaluation.audit_coding_expansion import CONDITIONS,STUDY
from evaluation.benchmark import file_sha256
from evaluation.checkpoint import write_json_atomic
from evaluation.coding_expansion import completion_text
from evaluation.review_coding import failure_category
from training.experiment_state import rows_sha256

def paired_counts(left,right):
    if set(left)!=set(right) or not left:
        raise ValueError('Paired comparisons require the same nonempty task coverage.')
    counts={'both_passed':0,'right_only_passed':0,'left_only_passed':0,'neither_passed':0}
    for index in left:
        a,b=left[index]['passed'],right[index]['passed']
        key='both_passed' if a and b else 'right_only_passed' if b else 'left_only_passed' if a else 'neither_passed'
        counts[key]+=1
    counts['net_gain']=counts['right_only_passed']-counts['left_only_passed']
    counts['percentage_point_change']=100*counts['net_gain']/len(left)
    return counts

def load_verified_records(path,rows,audited):
    if (audited['prediction_sha256']!=file_sha256(path) or audited['rows_sha256']!=rows_sha256(rows)
        or audited['examples']!=len(rows)):
        raise ValueError('Predictions or references changed after independent replay.')
    values=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    if (len(values)!=len(rows) or any(type(r.get('test_index')) is not int for r in values)
        or {r['test_index'] for r in values}!=set(range(len(rows)))):
        raise ValueError('Paired review requires complete unique coverage.')
    for r in values:
        row=rows[r['test_index']]
        if (r['task_id']!=row['task_id'] or r['task_sha256']!=rows_sha256([row])
            or type(r['passed']) is not bool or type(r['syntax_valid']) is not bool
            or completion_text(r['output_text'])!=r['completion']):
            raise ValueError('Prediction identity or raw completion differs from the audited task.')
    return {r['test_index']:r for r in values}

def review():
    manifest=verify_splits()
    audit_path=STUDY/'independent-audit.json'
    if not audit_path.exists():raise ValueError('Complete the independent Docker audit first.')
    audit=json.loads(audit_path.read_text(encoding='utf-8'))
    expected={f'{name}/{split}' for name in CONDITIONS for split in ('validation','test')}
    if audit.get('coverage_and_scores_verified') is not True or set(audit['results'])!=expected:
        raise ValueError('Independent audit must cover all six comparison logs.')
    protocol=json.loads((STUDY/'study_protocol.json').read_text(encoding='utf-8'))
    if any(file_sha256(ROOT/name)!=digest for name,digest in protocol['source'].items()):
        raise ValueError('Frozen study source changed.')
    records={};summaries={};categories={};validation={}
    for name,(variant,folder) in CONDITIONS.items():
        for split in ('validation','test'):
            rows=read_rows(split);stem=f'{variant}-{split}-{len(rows)}-tokens512'
            values=load_verified_records(folder/f'{stem}.jsonl',rows,audit['results'][f'{name}/{split}'])
            replay=[json.loads(line) for line in (STUDY/'audit'/f'{name}-{split}.jsonl').read_text(encoding='utf-8').splitlines()]
            if len(replay)!=len(rows) or {r['test_index'] for r in replay}!=set(range(len(rows))):
                raise ValueError('Independent replay log lacks complete coverage.')
            for r in replay:
                original=values[r['test_index']]
                if r.get('independent_replay_verified') is not True or any(r.get(key)!=value for key,value in original.items()):
                    raise ValueError('Replay record differs from the original prediction.')
            summary=json.loads((folder/f'{stem}-summary.json').read_text(encoding='utf-8'))
            passed=sum(r['passed'] for r in values.values())
            if (summary['examples']!=len(rows) or summary['passed']!=passed
                or summary['rows_sha256']!=rows_sha256(rows) or summary['pass_at_1']!=passed/len(rows)
                or summary['syntax_valid_rate']!=sum(r['syntax_valid'] for r in values.values())/len(rows)
                or summary['token_limit_hit_rate']!=sum(r['hit_token_limit'] for r in values.values())/len(rows)
                or summary['median_latency_seconds']!=statistics.median(r['latency_seconds'] for r in values.values())):
                raise ValueError('Summary does not match independently audited records.')
            if split=='test':
                records[name]=values;summaries[name]=summary
                categories[name]=dict(Counter(failure_category(r) for r in values.values()))
            else:validation[name]=summary
    pairs={f'{right}_vs_{left}':paired_counts(records[left],records[right])
           for left,right in (('base','pilot'),('base','expanded'),('pilot','expanded'))}
    rows=read_rows('test');cases=[]
    for index,row in enumerate(rows):
        outcomes={name:record[index]['passed'] for name,record in records.items()}
        cases.append({'test_index':index,'task_id':row['task_id'],'prompt':row['prompt'],
            'reference_completion':row['completion'],'passed':outcomes,
            'expanded_vs_base':'both_passed' if outcomes['base'] and outcomes['expanded'] else
                'expanded_only_passed' if outcomes['expanded'] else 'base_only_passed' if outcomes['base'] else 'neither_passed',
            'outputs':{name:{key:record[index].get(key) for key in
                ('completion','output_text','syntax_valid','error_type','error','hit_token_limit','latency_seconds')}
                for name,record in records.items()}})
    result={'coverage_and_replay_verified':True,'examples':len(rows),'audited_predictions':sum(r['examples'] for r in audit['results'].values()),
            'audit_sha256':file_sha256(audit_path),'review_code_sha256':file_sha256(Path(__file__)),
            'data_rows_sha256':manifest['row_sha256'],'summaries':summaries,'validation':validation,
            'paired_comparisons':pairs,'result_categories':categories,'cases':cases,
            'scope':'Eligible MBPP465 subset; original HumanEval164 and its pilot conclusions remain unchanged.',
            'decision':'Keep the runtime default and original Coding adapter unchanged. No GPU inference, automatic promotion or test-based recipe tuning.'}
    write_json_atomic(STUDY/'test-case-review.json',result)
    lines=['# Coding expansion: audited results and paired case review','',
        f'All {result["audited_predictions"]} saved validation/test predictions independently reproduced in pinned CPU Docker containers. No model text regenerated for this review.','',
        '| Condition | Validation passed | New MBPP test passed | Syntax valid | Token-limit hits | Median generation latency |',
        '|---|---:|---:|---:|---:|---:|']
    for name,s in summaries.items():
        values=list(records[name].values())
        lines.append(f"| {name} | {validation[name]['passed']}/32 | {s['passed']}/465 ({s['pass_at_1']:.2%}) | {sum(r['syntax_valid'] for r in values)}/465 | {sum(r['hit_token_limit'] for r in values)}/465 | {s['median_latency_seconds']:.3f}s |")
    lines+=['','Latency measures original model generation only; excludes Docker checks and HTTP. It is not a fresh performance benchmark.','',
            '| Paired comparison (right versus left) | Both passed | Gained | Lost | Neither passed | Net | Change |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for name,p in pairs.items():
        lines.append(f"| {name} | {p['both_passed']} | {p['right_only_passed']} | {p['left_only_passed']} | {p['neither_passed']} | {p['net_gain']:+d} | {p['percentage_point_change']:+.2f} pp |")
    lines+=['','| Result category | Base | Original Coding | Expanded Coding |','|---|---:|---:|---:|']
    for category in ('passed','syntax','test_assertion','runtime','timeout'):
        lines.append('| '+category+' | '+' | '.join(str(categories[name].get(category,0)) for name in CONDITIONS)+' |')
    lines+=['','The precommitted 3k adapter improved this eligible MBPP subset by 25 passing tasks versus Base and 8 versus the original Coding pilot.',
            'It gained66 and lost41 versus Base; improvements are not universal. Validation remained17/32, below the original pilot19/32.',
            'Expanded Coding has fewer syntax failures and token-limit hits, but its recorded median generation latency is longer than Base.',
            'This is one seed on an eligible subset, not the full official MBPP or a general coding superiority claim. No new HumanEval score exists for the expanded adapter.',
            'Training quantity and source composition both changed:339 tested MBPP plus2661 MIT Magicoder synthetic references, which are not execution-certified.',
            'Exact body filtering does not establish semantic or pretraining decontamination. Passing finite task tests is not proof for every input.',
            result['decision'],'','## Representative paired cases','']
    for title,transition in (('Expanded gained over Base','expanded_only_passed'),('Expanded lost to Base','base_only_passed'),('Both failed','neither_passed')):
        lines.append('### '+title);lines.append('')
        for case in [c for c in cases if c['expanded_vs_base']==transition][:3]:
            lines+=['#### '+case['task_id'],'','```python',case['prompt'].rstrip(),'```','']
            for name in ('base','expanded'):
                output=case['outputs'][name]
                lines+=[f"{name}: passed={case['passed'][name]}, error={output['error_type'] or 'none'}.",'',
                        '```python',output['completion'].rstrip(),'```','']
    lines+=['All465 paired programs, references and error messages are available in test-case-review.json. These held-out cases are for inspection, not new training examples.','']
    (STUDY/'test-case-review.md').write_text('\n'.join(lines),encoding='utf-8')
    return result

if __name__=='__main__':
    result=review();print(json.dumps({key:result[key] for key in
        ('examples','audited_predictions','paired_comparisons','result_categories','decision')},indent=2))
