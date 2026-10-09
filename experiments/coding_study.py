"""Resume the coding pilot, compare Base/Tool/Coding, and report measured outcomes."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from coding.data import DATA,read_rows
from coding.sandbox import evaluate_program
from evaluation.checkpoint import exclusive_prediction_file,write_json_atomic
from experiments.rank_ablation import read_json,run_stage,set_status,process_alive
from evaluation.benchmark import file_sha256

CONFIG=ROOT/'configs/coding/mbpp_r8.json'
STUDY=ROOT/json.loads(CONFIG.read_text())['output_dir']

def evaluation_summary(variant,split):
    return read_json(STUDY/'evaluation'/f'{variant}-{split}-{32 if split=="validation" else 164}-tokens512-summary.json')

def report(status):
    data=read_json(DATA/'manifest.json')
    results={v:{s:evaluation_summary(v,s) for s in ('validation','test')} for v in ('base','tool','coding')}
    value={'status':status,'data_counts':data.get('counts') if data else None,'training':read_json(STUDY/'run_metrics.json'),
           'results':results,'scope':'Single-seed raw Python completion pilot. MBPP339 training /32 eligible validation; HumanEval164 held-out.',
           'selection':'One predetermined r8 recipe. Official test outcomes are reported without selecting or tuning from test scores.'}
    if all(results[v]['validation'] for v in ('base','coding')):
        value['coding_validation_gain']=results['coding']['validation']['pass_at_1']-results['base']['validation']['pass_at_1']
    write_json_atomic(STUDY/'comparison.json',value)
    lines=['# Coding LoRA pilot: shared Qwen3-8B-Base','',f'Status: {status}.','',value['scope'],
           'All modes receive the identical raw function prefix, one greedy candidate,512-token cap and completion-boundary stops.',
           'Only prompt tokens are model inputs; reference solutions and tests remain outside generation.',
           'Generated programs run in pinned Docker containers with no network or host mounts, read-only root, unprivileged user and resource limits.',
           'Training is a fresh r8 adapter: NF4/BF16 Base,FP32 LoRA,alpha16,dropout.05,constant head/body LR1e-4,effective batch2,two epochs.','',
           '| Mode | MBPP validation pass@1 | HumanEval pass@1 | HumanEval syntax | Median generation seconds |','|---|---:|---:|---:|---:|']
    for variant,metrics in results.items():
        val,test=metrics['validation'],metrics['test']
        cells=[f"{val['passed']}/{val['examples']} ({val['pass_at_1']:.2%})" if val else 'pending',
               f"{test['passed']}/{test['examples']} ({test['pass_at_1']:.2%})" if test else 'pending',
               f"{test['syntax_valid_rate']:.2%}" if test else 'pending',
               f"{test['median_latency_seconds']:.2f}" if test else 'pending']
        lines.append('| '+variant+' | '+' | '.join(cells)+' |')
    lines += ['', 'An adapter is not a proven specialist merely because training completes; compare measured pass rates with Base.',
              'MBPP here is an eligible single-function validation subset, not the full official MBPP benchmark.',
              'One deterministic candidate per task is reported as greedy pass@1, not sampled pass@k.',
              'Project training excludes exact reference-body overlap; Base pretraining contamination is not certified absent.',
              'Do not add official HumanEval errors to training or use its scores to tune this reported experiment.',
              'Recorded timing depends on desktop workloads, output length and thermals. Docker test time is excluded from generation latency.',
              'The existing Tool-use adapter and all its benchmark artifacts remain unchanged.','']
    (STUDY/'comparison.md').write_text('\n'.join(lines),encoding='utf-8')
    return value

def snapshot():
    status=read_json(STUDY/'study_status.json') or {'status':'not_started','stage':None}
    if status.get('status')=='running':
        status['runner_alive']=process_alive(status.get('runner_pid'))
        if status['runner_alive'] is False: status['status']='interrupted'
        elif status['runner_alive'] is None: status['status']='unknown'
    evals={}
    folder=STUDY/'evaluation'
    for variant in ('base','tool','coding'):
        evals[variant]={split:read_json(folder/f'{variant}-{split}-{32 if split=="validation" else 164}-tokens512.progress.json') for split in ('validation','test')}
    return {'study':status,'data_counts':(read_json(DATA/'manifest.json') or {}).get('counts'),
            'training':read_json(STUDY/'training_progress.json'),'evaluation':evals}

def verify_references(image):
    path=STUDY/'humaneval-reference-verification.json'
    fingerprint={'test_sha256':file_sha256(DATA/'test.jsonl'),'sandbox_sha256':file_sha256(ROOT/'coding/sandbox.py'),'image':image}
    old=read_json(path)
    if old:
        if old['fingerprint']!=fingerprint or old['passed']!=164:
            raise ValueError('HumanEval reference verification differs.')
        return
    for index,row in enumerate(read_rows('test')):
        result=evaluate_program(row['canonical_code'],row['test_code'],image)
        if not result['passed']:
            raise ValueError(f'HumanEval reference failed in the pinned runtime: {row["task_id"]}: {result}')
        print(f'HumanEval reference {index+1}/164 passed.',flush=True)
    write_json_atomic(path,{'fingerprint':fingerprint,'passed':164})

def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--status',action='store_true')
    cli.add_argument('--runner-log',type=Path)
    cli.add_argument('--verify-references-only',action='store_true')
    args=cli.parse_args()
    if args.status:
        print(json.dumps(snapshot(),indent=2)); return 0
    cfg=json.loads(CONFIG.read_text())
    STUDY.mkdir(exist_ok=True); (STUDY/'logs').mkdir(exist_ok=True)
    if args.runner_log:
        log=args.runner_log.resolve()
        if not log.is_relative_to(ROOT): raise ValueError('Queue log must stay inside the project.')
        sys.stdout=sys.stderr=log.open('a',encoding='utf-8',buffering=1)
    if args.verify_references_only:
        import signal
        if hasattr(signal,'SIGBREAK'):
            def interrupt(*args): raise KeyboardInterrupt
            signal.signal(signal.SIGBREAK,interrupt)
        verify_references(cfg['docker_image']); return 0
    with exclusive_prediction_file(STUDY/'.runner.lock'):
        try:
            run_stage('prepare',['-m','coding.data','--image',cfg['docker_image']],folder=STUDY)
            run_stage('prepare-training',['training/train_coding_lora.py','--prepare-only'],folder=STUDY)
            report('running')
            for variant in ('base','tool'):
                run_stage(f'{variant}-validation',['evaluation/coding_benchmark.py','--variant',variant],folder=STUDY)
                report('running')
            run_stage('train-coding',['training/train_coding_lora.py'],folder=STUDY)
            run_stage('coding-validation',['evaluation/coding_benchmark.py','--variant','coding'],folder=STUDY)
            report('validation-completed')
            run_stage('verify-human-references',['experiments/coding_study.py','--verify-references-only'],folder=STUDY)
            for variant in ('base','tool','coding'):
                run_stage(f'{variant}-test',['evaluation/coding_benchmark.py','--variant',variant,'--split','test'],folder=STUDY)
                report('running')
            run_stage('audit-coding',['evaluation/audit_coding.py'],folder=STUDY)
            report('completed'); set_status('completed','report-written',folder=STUDY); return 0
        except KeyboardInterrupt:
            report('interrupted'); print('Stopped. Repeat the study command to resume checkpoints and saved cases.'); return 130
        except Exception as error:
            current=read_json(STUDY/'study_status.json') or {}
            set_status('failed',current.get('stage','setup'),folder=STUDY,error=str(error))
            report('failed'); raise

if __name__=='__main__':
    raise SystemExit(main())
