"""Sequential, resumable remaining research: fixed HumanEval, 3k seeds and Tool rank seeds."""
import argparse
import json
from pathlib import Path
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.benchmark import file_sha256
from evaluation.checkpoint import exclusive_prediction_file,write_json_atomic
from experiments.rank_ablation import read_json,run_stage,set_status,process_alive

STUDY=ROOT/'outputs/gpu-remaining-20261007'
CODING_NAMES=('coding-seed43','coding-seed44')
TOOLS=tuple(f'tool-seed{seed}-r{rank}' for seed in (43,44) for rank in (8,16,32))
SOURCES=('experiments/gpu_remaining.py','training/train_coding_expansion.py','training/train_fresh_protocol.py',
    'evaluation/coding_benchmark.py','evaluation/coding_expansion.py','evaluation/rank_validation.py',
    'evaluation/audit_remaining_coding.py','coding/expansion_data.py','coding/data.py','coding/sandbox.py')

def status():
    state=read_json(STUDY/'study_status.json') or {'status':'not_started'}
    if state['status']=='running':
        state['runner_alive']=process_alive(state.get('runner_pid'))
        if state['runner_alive'] is False:state['status']='interrupted'
        elif state['runner_alive'] is None:state['status']='unknown'
    return {'study':state,'humaneval':read_json(STUDY/'humaneval/coding-test-164-tokens512.progress.json'),
        'coding':{name:{'training':read_json(STUDY/name/'training_progress.json'),
            'validation':read_json(STUDY/name/'evaluation/coding-validation-32-tokens512.progress.json'),
            'test':read_json(STUDY/name/'evaluation/coding-test-465-tokens512.progress.json')} for name in CODING_NAMES},
        'tool_rank':{name:{'training':read_json(STUDY/name/'training_progress.json'),
            'validation':read_json(STUDY/name/'evaluation/validation-100-tokens256-progress.json')} for name in TOOLS}}

def report(state):
    coding={name:{'training':read_json(STUDY/name/'run_metrics.json'),
        'validation':read_json(STUDY/name/'evaluation/coding-validation-32-tokens512-summary.json'),
        'test':read_json(STUDY/name/'evaluation/coding-test-465-tokens512-summary.json')} for name in CODING_NAMES}
    tools={name:{'training':read_json(STUDY/name/'run_metrics.json'),
        'validation':read_json(STUDY/name/'evaluation/validation-summary.json')} for name in TOOLS}
    human=read_json(STUDY/'humaneval/coding-test-164-tokens512-summary.json')
    audit=read_json(STUDY/'humaneval/coding-test-164-tokens512-audit.json')
    # Baseline reuse is allowed only after checking every non-adapter fingerprint field.
    if human:
        expected=read_json(ROOT/'outputs/coding-mbpp-r8-fp32/evaluation/base-test-164-tokens512.metadata.json')
        actual=read_json(STUDY/'humaneval/coding-test-164-tokens512.metadata.json')
        for key in set(expected)|set(actual):
            if key not in ('variant','adapter') and expected.get(key)!=actual.get(key):
                raise ValueError(f'HumanEval baseline protocol mismatch: {key}.')
    value={'status':state,'humaneval':human,'humaneval_audit':audit,'coding_replicas':coding,'tool_rank_replicas':tools,
        'interpretation':'Precommitted descriptive HumanEval comparison and seed variability studies. No test-based tuning or automatic adapter promotion; Tool replicas use validation only.'}
    write_json_atomic(STUDY/'comparison.json',value)
    lines=['# Remaining GPU research','',f'Status: {state}.','',
        'Sequence: fixed expanded-adapter HumanEval comparison and CPU replay;3k Coding seeds43/44 with validation/test and CPU replay; Tool ranks8/16/32 at seeds43/44 with validation only.',
        'Original models, datasets, trainers, evaluators and prior benchmark reports stay separate. All seeds use the same frozen rows.',
        'This known HumanEval comparison is descriptive; do not tune on official cases. No adapter is promoted from test scores.',
        'Checkpoint every20 optimizer updates; coding predictions save each case. GPU stages are serialized.','',
        '## Expanded Coding on the known HumanEval164 benchmark','',
        f"New3k result: {human['passed']}/164 ({human['pass_at_1']:.2%}). Original cached Base98/164 and Coding97/164." if human else 'pending',
        'Independent replay: '+('verified' if audit else 'pending')+'.','',
        '## 3k Coding seed replication','',
        'Seed42 control: validation17/32, eligible MBPP249/465. Data composition and decoder are fixed across replicas.',
        '| Condition | Updates | Validation | MBPP465 test |','|---|---:|---:|---:|']
    for name,r in coding.items():
        cells=[str(r['training']['optimizer_steps']) if r['training'] else 'pending']
        cells+=[str(r[split]['passed'])+'/'+str(r[split]['examples']) if r[split] else 'pending' for split in ('validation','test')]
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    lines+=['','## Tool-use rank replication','',
        'Fixed1000 train /100 validation;2epochs/1000 updates; head/bodyLR1e-3/1e-4; rank/alpha8/16,16/32,32/64. New replicas do not repeat the known official961 test.',
        '| Condition | Updates | Validation exact | Validation mock |','|---|---:|---:|---:|']
    for name,r in tools.items():
        m=r['validation']['metrics'] if r['validation'] else None
        lines.append('| '+name+' | '+(str(r['training']['optimizer_steps']) if r['training'] else 'pending')+' | '+
            (f"{m['full_call_exact_match_rate']:.2%} | {m['mock_execution_success_rate']:.2%}" if m else 'pending | pending')+' |')
    lines+=['','Compare all precommitted seeds, including negative results; do not select the best seed from test scores.','']
    (STUDY/'comparison.md').write_text('\n'.join(lines),encoding='utf-8')

def run(stage,arguments):
    run_stage(stage,arguments,folder=STUDY);report('running')

def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--status',action='store_true');cli.add_argument('--prepare-only',action='store_true');cli.add_argument('--runner-log',type=Path)
    args=cli.parse_args()
    if args.status:print(json.dumps(status(),indent=2));return
    STUDY.mkdir(exist_ok=True);(STUDY/'logs').mkdir(exist_ok=True)
    if args.runner_log:
        path=args.runner_log.resolve()
        if not path.is_relative_to(ROOT):raise ValueError('Log must remain in this project.')
        sys.stdout=sys.stderr=path.open('a',encoding='utf-8',buffering=1)
    with exclusive_prediction_file(STUDY/'.runner.lock'):
        try:
            configs=sorted((ROOT/'configs/research_remaining').glob('*.json'))
            protocol={'sources':{name:file_sha256(ROOT/name) for name in SOURCES},
                'configs':{str(path.relative_to(ROOT)):file_sha256(path) for path in configs},
                'data':{name:file_sha256(ROOT/name) for name in ('data/processed/coding-expansion-3k/manifest.json','data/processed/coding-mbpp-pilot/manifest.json')},
                'policy':'Fixed seed42 expanded model for descriptive HumanEval; two3k training seed replicas; sixTool validation-only rank/seed replicas. No test-based selection.'}
            old=read_json(STUDY/'study_protocol.json')
            if old is not None and old!=protocol:raise ValueError('Frozen remaining-study protocol changed.')
            write_json_atomic(STUDY/'study_protocol.json',protocol)
            for name in CODING_NAMES:
                seed=name[-2:]
                run('prepare-'+name,['training/train_coding_expansion.py','--config',f'configs/research_remaining/coding_seed{seed}.json','--prepare-only'])
            for name in TOOLS:
                seed,rank=name.split('-')[1][4:],name.split('-')[2][1:]
                run('prepare-'+name,['training/train_fresh_protocol.py','--config',f'configs/research_remaining/tool_seed{seed}_r{rank}.json','--prepare-only'])
            if args.prepare_only:
                report('prepared');set_status('prepared','data-verified',folder=STUDY);return
            with urllib.request.urlopen('http://127.0.0.1:18765/api/status',timeout=10) as response:demo=json.load(response)
            if demo['status']!='paused' or demo['shared_base_load_count']!=0:
                raise RuntimeError('Pause the owned demo model before running this serialized GPU queue.')
            run('humaneval-expanded',['evaluation/coding_benchmark.py','--variant','coding','--split','test',
                '--adapter','outputs/coding-expansion-3k/r8','--output-dir',str(STUDY/'humaneval')])
            run('audit-humaneval',['evaluation/audit_remaining_coding.py','--family','pilot','--split','test',
                '--config','configs/coding/mbpp_r8.json','--predictions',str(STUDY/'humaneval/coding-test-164-tokens512.jsonl')])
            for name in CODING_NAMES:
                cfg=f'configs/research_remaining/coding_seed{name[-2:]}.json'
                run('train-'+name,['training/train_coding_expansion.py','--config',cfg])
                for split,n in (('validation',32),('test',465)):
                    run(split+'-'+name,['evaluation/coding_expansion.py','--config',cfg,'--variant','coding','--split',split])
                    run('audit-'+split+'-'+name,['evaluation/audit_remaining_coding.py','--family','expanded','--split',split,
                        '--config',cfg,'--predictions',str(STUDY/name/f'evaluation/coding-{split}-{n}-tokens512.jsonl')])
            for name in TOOLS:
                seed,rank=name.split('-')[1][4:],name.split('-')[2][1:]
                cfg=f'configs/research_remaining/tool_seed{seed}_r{rank}.json'
                run('train-'+name,['training/train_fresh_protocol.py','--config',cfg])
                run('validate-'+name,['evaluation/rank_validation.py','--adapter',str(STUDY/name)])
            report('completed');set_status('completed','report-written',folder=STUDY)
        except KeyboardInterrupt:
            current=read_json(STUDY/'study_status.json') or {};report('interrupted')
            set_status('interrupted',current.get('stage','setup'),folder=STUDY);raise
        except Exception as error:
            current=read_json(STUDY/'study_status.json') or {};report('failed')
            set_status('failed',current.get('stage','setup'),folder=STUDY,error=str(error));raise

if __name__=='__main__':main()
