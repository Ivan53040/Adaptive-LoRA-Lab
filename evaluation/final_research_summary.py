"""Validate and summarize completed local research artifacts using CPU-only reads."""
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.benchmark import file_sha256,parse_calls,score_calls
from runtime.mock_tools import MockWorld
from evaluation.checkpoint import write_json_atomic
from evaluation.review_coding_expansion import load_verified_records
from coding import data as pilot
from coding import expansion_data as expanded
from training.experiment_state import rows_sha256

STUDY=ROOT/'outputs/gpu-remaining-20261007'
def read(path):return json.loads(path.read_text(encoding='utf-8'))

def range_summary(values):
    if not values:raise ValueError('A summary needs at least one value.')
    return {'mean':statistics.mean(values),'minimum':min(values),'maximum':max(values),'conditions':len(values)}

def verified_coding(folder,split,family):
    rows=family.read_rows(split);path=folder/f'coding-{split}-{len(rows)}-tokens512.jsonl'
    audit_path=path.with_name(path.stem+'-audit.json');audit=read(audit_path)
    if audit.get('coverage_and_scores_verified') is not True:raise ValueError('Coding replay is not verified.')
    values=load_verified_records(path,rows,audit);summary=read(path.with_name(path.stem+'-summary.json'))
    passed=sum(r['passed'] for r in values.values())
    if summary['passed']!=passed or audit['passed']!=passed or summary['examples']!=len(rows):
        raise ValueError('Coding scores differ from audited records.')
    return summary,audit

def summarize():
    if read(STUDY/'study_status.json')['status']!='completed':raise ValueError('Complete the GPU queue first.')
    protocol=read(STUDY/'study_protocol.json')
    for group in ('sources','configs','data'):
        if any(file_sha256(ROOT/name)!=digest for name,digest in protocol[group].items()):
            raise ValueError('Frozen remaining-study source/config/data changed.')
    human,human_audit=verified_coding(STUDY/'humaneval','test',pilot)
    baseline_folder=ROOT/'outputs/coding-mbpp-r8-fp32/evaluation'
    base_meta=read(baseline_folder/'base-test-164-tokens512.metadata.json')
    new_meta=read(STUDY/'humaneval/coding-test-164-tokens512.metadata.json')
    if any(base_meta.get(key)!=new_meta.get(key) for key in set(base_meta)|set(new_meta) if key not in ('variant','adapter')):
        raise ValueError('HumanEval baseline protocol differs.')
    base_human=read(baseline_folder/'base-test-164-tokens512-summary.json')
    original_human=read(baseline_folder/'coding-test-164-tokens512-summary.json')
    human_table=[{'condition':'base',**base_human},{'condition':'pilot',**original_human},{'condition':'expanded',**human}]
    coding=[];coding_audits=164;control_manifest=None
    for seed in (42,43,44):
        folder=ROOT/'outputs/coding-expansion-3k/r8' if seed==42 else STUDY/f'coding-seed{seed}'
        manifest=read(folder/'training_manifest.json');train=read(folder/'run_metrics.json')
        if train['optimizer_steps']!=3000:raise ValueError('Incomplete3k training.')
        common={key:manifest[key] for key in ('train_sha256','validation_sha256','test_sha256','base_files','versions','label_policy','training_code_sha256','helper_code_sha256')}
        if control_manifest is not None and common!=control_manifest:raise ValueError('3k replica data or implementation differs.')
        control_manifest=common
        if seed==42:
            report=read(ROOT/'outputs/coding-expansion-3k/test-case-review.json')
            validation,test=report['validation']['expanded'],report['summaries']['expanded']
        else:
            validation,_=verified_coding(folder/'evaluation','validation',expanded)
            test,_=verified_coding(folder/'evaluation','test',expanded);coding_audits+=32+465
        coding.append({'seed':seed,'updates':train['optimizer_steps'],'validation_passed':validation['passed'],
                       'test_passed':test['passed'],'test_rate':test['pass_at_1']})
    tools=[];tool_identity=None;initializations=set()
    for seed in (42,43,44):
        for rank in (8,16,32):
            folder=ROOT/f'outputs/fair-rank-completion-1k/r{rank}' if seed==42 else STUDY/f'tool-seed{seed}-r{rank}'
            manifest=read(folder/'training_manifest.json');training=read(folder/'run_metrics.json');summary=read(folder/'evaluation/validation-summary.json')
            common={key:manifest[key] for key in ('train_sha256','validation_sha256','test_sha256','base_files','versions','label_policy','training_code_sha256','helper_code_sha256')}
            if tool_identity is not None and common!=tool_identity:raise ValueError('Tool replica data or implementation differs.')
            tool_identity=common;initializations.add(training['initial_base_logits_sha256'])
            rows=[json.loads(line) for line in (folder/'validation.jsonl').read_text(encoding='utf-8').splitlines()]
            values=[json.loads(line) for line in (folder/'evaluation/validation-100-tokens256-predictions.jsonl').read_text(encoding='utf-8').splitlines()]
            if (len(values)!=100 or {r['test_index'] for r in values}!=set(range(100))
                or rows_sha256(rows)!=manifest['validation_sha256'] or training['optimizer_steps']!=1000):
                raise ValueError('Incomplete Tool coverage.')
            exact=sum(r['full_call_exact'] for r in values);mock=sum(r['mock_execution_ok'] for r in values)
            for record in values:
                row=rows[record['test_index']]
                expected=[{'name':c['function']['name'],'arguments':c['function']['arguments']} for c in row['messages'][-1]['tool_calls']]
                predicted,error=parse_calls(record['output_text'])
                rescored=score_calls(predicted,expected,error)
                execution=MockWorld(row['tools']).execute(predicted) if error is None else {'ok':False}
                if (record['expected_calls']!=expected or record['prompt']!=row['messages'][-2]['content']
                    or record['full_call_exact']!=rescored['full_call_exact'] or record['mock_execution_ok']!=execution['ok']):
                    raise ValueError('Saved Tool case did not independently reparse/rescore/mock-execute.')
            if summary['metrics']['full_call_exact_match_rate']!=exact/100 or summary['metrics']['mock_execution_success_rate']!=mock/100:
                raise ValueError('Tool summary differs from saved records.')
            tools.append({'seed':seed,'rank':rank,'updates':1000,'exact':exact,'mock':mock,
                          'peak_training_gib':training['peak_allocated_vram_gib']})
    if len(initializations)!=1:raise ValueError('Tool conditions did not initialize from identical Base logits.')
    rank_summary={str(rank):range_summary([r['exact'] for r in tools if r['rank']==rank]) for rank in (8,16,32)}
    result={'completed':True,'sources_and_configs_verified':True,'known_humaneval':human_table,
        'humaneval_net_tasks_vs_base':human['passed']-base_human['passed'],
        'coding_seeds':coding,'coding_test_rate_summary':range_summary([r['test_rate'] for r in coding]),
        'tool_conditions':tools,'tool_rank_summary':rank_summary,'remaining_study_coding_replay_records':coding_audits,
        'tool_validation_cases_independently_rescored':900,
        'limits':'Known HumanEval descriptive comparison; eligible MBPP465 subset; Tool rank evidence is validation-only on100cases. Same cases repeated across seeds are not independent new datasets. No test-based tuning or automatic promotion.'}
    write_json_atomic(STUDY/'final-summary.json',result)
    lines=['# Final research summary','',
        'Completed studies; generated from validated saved artifacts without model loading or new GPU work.','',
        '## Known HumanEval164 comparison','',
        '| Condition | Passed | Pass@1 |','|---|---:|---:|']
    for row in human_table:lines.append(f"| {row['condition']} | {row['passed']}/164 | {row['pass_at_1']:.2%} |")
    lines+=['',f"Expanded seed42 gained a net {result['humaneval_net_tasks_vs_base']} passing tasks versus the protocol-matched cached Base. Its164 saved predictions independently reproduced in CPU Docker.",
            'This known benchmark comparison is descriptive; it was not a new unseen test and did not select a model or training recipe.','',
            '## 3k Coding seed replication','',
            '| Seed | Updates | Validation | Eligible MBPP test |','|---|---:|---:|---:|']
    for row in coding:lines.append(f"| {row['seed']} | {row['updates']} | {row['validation_passed']}/32 | {row['test_passed']}/465 ({row['test_rate']:.2%}) |")
    cs=result['coding_test_rate_summary'];lines+=['',f"Mean test pass@1: {cs['mean']:.2%}; observed range: {cs['minimum']:.2%}–{cs['maximum']:.2%}.",
        'All three conditions retain the same3000 training rows and decoder. The mean/range describe training-seed variation on the same465 tasks; they are not a confidence interval or evidence from three independent benchmarks.','',
        '## Tool rank replication','',
        '| Rank | Seed42 exact | Seed43 exact | Seed44 exact | Mean exact | Observed range |','|---|---:|---:|---:|---:|---:|']
    for rank in (8,16,32):
        rows=[r for r in tools if r['rank']==rank];s=rank_summary[str(rank)]
        lines.append(f"| {rank} | "+' | '.join(f"{r['exact']:.2f}%" for r in rows)+f" | {s['mean']:.2f}% | {s['minimum']:.0f}–{s['maximum']:.0f}% |")
    lines+=['','All nine conditions use1000 frozen training rows,100 validation cases and1000 updates. r8 has the highest observed mean with fewer adapter parameters. Three seeds and a shared100-case validation set do not establish universal rank superiority.',
            'All900 saved Tool validation cases independently reparsed, rescored and mock-executed on CPU. New Tool replicas did not rerun the official961 test. Existing selected Tool3k/r8 benchmark and runtime defaults remain unchanged.',
            f"This remaining study independently replayed {coding_audits} Coding predictions (HumanEval164 plus two497-case replica logs). Earlier seed42 replay proof remains separate.",
            result['limits'],'','No new model generation, training, or adapter promotion was performed to create this report.','']
    (STUDY/'final-summary.md').write_text('\n'.join(lines),encoding='utf-8')
    return result

if __name__=='__main__':
    value=summarize();print(json.dumps({key:value[key] for key in ('completed','humaneval_net_tasks_vs_base','coding_test_rate_summary','tool_rank_summary')},indent=2))
