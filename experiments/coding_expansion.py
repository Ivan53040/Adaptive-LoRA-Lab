"""Queue wider Coding training after the controlled follow-ups; preserve held-out results."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from coding.expansion_data import DATA,verify_splits
from evaluation.audit_coding_expansion import CONDITIONS
from evaluation.benchmark import file_sha256
from evaluation.checkpoint import exclusive_prediction_file,write_json_atomic
from experiments.rank_ablation import read_json,run_stage,set_status,process_alive

STUDY=ROOT/'outputs/coding-expansion-3k'
CONFIG='configs/coding/expansion_r8.json'
FOLLOWUP=ROOT/'outputs/coding-followup/study_status.json'
PILOT='outputs/coding-mbpp-r8-fp32'

def snapshot():
    state=read_json(STUDY/'study_status.json') or {'status':'not_started'}
    if state['status']=='running':
        state['runner_alive']=process_alive(state.get('runner_pid'))
        if state['runner_alive'] is False:state['status']='interrupted'
        elif state['runner_alive'] is None:state['status']='unknown'
    return {'study':state,'followup':read_json(FOLLOWUP),
            'training':read_json(STUDY/'r8/training_progress.json'),
            'evaluation':{name:{split:read_json(folder/f'{variant}-{split}-{32 if split=="validation" else 465}-tokens512.progress.json')
                for split in ('validation','test')} for name,(variant,folder) in CONDITIONS.items()},
            'independent_audit':read_json(STUDY/'independent-audit.json')}

def report(status):
    values={name:{split:read_json(folder/f'{variant}-{split}-{32 if split=="validation" else 465}-tokens512-summary.json')
            for split in ('validation','test')} for name,(variant,folder) in CONDITIONS.items()}
    paired={}
    base_path=CONDITIONS['base'][1]/'base-test-465-tokens512.jsonl'
    if values['base']['test']:
        base={r['test_index']:r for r in map(json.loads,base_path.read_text(encoding='utf-8').splitlines())}
        for name in ('pilot','expanded'):
            if not values[name]['test']:continue
            rows=list(map(json.loads,(CONDITIONS[name][1]/'coding-test-465-tokens512.jsonl').read_text(encoding='utf-8').splitlines()))
            paired[name]={'gained':sum(r['passed'] and not base[r['test_index']]['passed'] for r in rows),
                          'lost':sum(not r['passed'] and base[r['test_index']]['passed'] for r in rows)}
    write_json_atomic(STUDY/'comparison.json',{'status':status,'results':values,'paired_against_base':paired,
                      'training':read_json(STUDY/'r8/run_metrics.json'),'audit':read_json(STUDY/'independent-audit.json')})
    lines=['# Coding expansion: fixed 3k recipe and new eligible MBPP test','',f'Status: {status}.','',
        'Fresh original Base; seed42/r8/alpha16/all projections+head/FP32 LoRA, constant head/body LR1e-4, two epochs/3000 updates.',
        'Training: original339 verified MBPP plus2661 MIT Magicoder synthetic references. Synthetic references are not execution-certified.',
        'Validation: unchanged32 tasks. New test:465 eligible MBPP test tasks with Docker-verified references. Never train on test programs or use test scores to choose a condition.',
        'Compare Base, the preserved339-row Coding pilot, and the precommitted3k adapter. Data quantity and source composition both change, so this is not a pure size ablation.',
        'HumanEval results stay frozen. Exact-body filtering does not certify semantic or pretraining decontamination.','',
        '| Condition | Validation pass@1 | New test pass@1 |','|---|---:|---:|']
    for name,entry in values.items():
        cells=[f"{entry[split]['passed']}/{entry[split]['examples']} ({entry[split]['pass_at_1']:.2%})" if entry[split] else 'pending'
               for split in ('validation','test')]
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    for name,result in paired.items():lines.append(f"\n{name} versus Base: {result['gained']} gained, {result['lost']} lost.")
    lines+=['','Independent Docker replay: '+('verified' if read_json(STUDY/'independent-audit.json') else 'pending')+'.',
            'Adapters are not promoted automatically. These are subset scores under one fixed generation and sandbox protocol.','']
    (STUDY/'comparison.md').write_text('\n'.join(lines),encoding='utf-8')

def wait_for_followup():
    set_status('running','waiting-for-coding-followup',folder=STUDY)
    print('Waiting for the controlled follow-up queue; GPU work stays sequential.',flush=True)
    while True:
        state=read_json(FOLLOWUP)
        if state and state['status']=='completed':return
        if not state or state['status']!='running' or process_alive(state.get('runner_pid')) is not True:
            raise RuntimeError('Follow-up is not verified running/completed; resume it before this queue.')
        time.sleep(5)

def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--status',action='store_true');cli.add_argument('--runner-log',type=Path);args=cli.parse_args()
    if args.status:print(json.dumps(snapshot(),indent=2));return
    STUDY.mkdir(exist_ok=True);(STUDY/'logs').mkdir(exist_ok=True)
    if args.runner_log:
        path=args.runner_log.resolve()
        if not path.is_relative_to(ROOT):raise ValueError('Queue log must stay in this project.')
        sys.stdout=sys.stderr=path.open('a',encoding='utf-8',buffering=1)
    with exclusive_prediction_file(STUDY/'.runner.lock'):
        try:
            verify_splits()
            protocol={'config':read_json(ROOT/CONFIG),'data_manifest':file_sha256(DATA/'manifest.json'),
                      'source':{name:file_sha256(ROOT/name) for name in ('experiments/coding_expansion.py',
                          'training/train_coding_expansion.py','evaluation/coding_expansion.py',
                          'evaluation/audit_coding_expansion.py','coding/expansion_data.py')},
                      'rule':'Fixed seed42 full-projection3k model; no held-out-based tuning or automatic promotion.'}
            old=read_json(STUDY/'study_protocol.json')
            if old is not None and old!=protocol:raise ValueError('Expansion protocol changed; use a separate study.')
            write_json_atomic(STUDY/'study_protocol.json',protocol);report('queued')
            wait_for_followup()
            run_stage('smoke3k',['training/train_coding_expansion.py','--smoke'],folder=STUDY)
            run_stage('train3k',['training/train_coding_expansion.py'],folder=STUDY);report('running')
            for split in ('validation','test'):
                for name,(variant,folder) in CONDITIONS.items():
                    command=['evaluation/coding_expansion.py','--variant',variant,'--split',split,'--output-dir',str(folder)]
                    if name=='pilot':command+=['--adapter',PILOT]
                    run_stage(f'{split}-{name}',command,folder=STUDY);report('running')
            run_stage('independent-audit',['evaluation/audit_coding_expansion.py'],folder=STUDY)
            report('completed');set_status('completed','report-written',folder=STUDY)
        except KeyboardInterrupt:
            current=read_json(STUDY/'study_status.json') or {}
            set_status('interrupted',current.get('stage','setup'),folder=STUDY);report('interrupted');raise
        except Exception as error:
            current=read_json(STUDY/'study_status.json') or {}
            set_status('failed',current.get('stage','setup'),folder=STUDY,error=str(error));report('failed');raise

if __name__=='__main__':main()
