"""Repeat seeds and compare attachment sites without revisiting the known official test."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.checkpoint import exclusive_prediction_file,write_json_atomic
from experiments.rank_ablation import run_stage,read_json,set_status,process_alive

STUDY=ROOT/'outputs/coding-followup'
NAMES=('seed43','seed44','attention-head-seed42')

def summary():
    baseline=ROOT/'outputs/coding-mbpp-r8-fp32'
    result={}
    for name,folder in [('all-head-seed42',baseline)]+[(n,STUDY/n) for n in NAMES]:
        result[name]={'training':read_json(folder/'run_metrics.json'),
                     'validation':read_json(folder/'evaluation/coding-validation-32-tokens512-summary.json')}
    return result

def report(status):
    values=summary();write_json_atomic(STUDY/'comparison.json',{'status':status,'results':values})
    lines=['# Coding follow-up: repeat seeds and attachment sites','',f'Status: {status}.','',
           'Same frozen339 train /32 validation tasks, original Base, r8/alpha16, two epochs/340 updates, FP32 adapters and constant LR1e-4.',
           'Compare all transformer projections+head across seeds42/43/44. Compare attention+head versus all+head at seed42; retain the head in both.',
           'Known HumanEval outcomes are not used to select these conditions; only validation is repeated. Larger-data work and a new held-out evaluation remain separate.','',
           '| Condition | Training updates | Validation pass@1 |','|---|---:|---:|']
    for name,value in values.items():
        train,val=value['training'],value['validation']
        lines.append(f"| {name} | {train['optimizer_steps'] if train else 'pending'} | "+(f"{val['passed']}/{val['examples']} ({val['pass_at_1']:.2%})" if val else 'pending')+' |')
    lines+=['','This small validation study assesses variability and attachment effects; it does not prove held-out generalization or a router-quality gain.','']
    (STUDY/'comparison.md').write_text('\n'.join(lines),encoding='utf-8')

def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--status',action='store_true');cli.add_argument('--runner-log',type=Path);args=cli.parse_args()
    if args.status:
        current=read_json(STUDY/'study_status.json') or {'status':'not_started'}
        if current.get('status')=='running':
            current['runner_alive']=process_alive(current.get('runner_pid'))
            if current['runner_alive'] is False:current['status']='interrupted'
            elif current['runner_alive'] is None:current['status']='unknown'
        print(json.dumps({'study':current,'conditions':{n:read_json(STUDY/n/'training_progress.json') for n in NAMES},
            'validation':{n:read_json(STUDY/n/'evaluation/coding-validation-32-tokens512.progress.json') for n in NAMES}},indent=2));return
    STUDY.mkdir(exist_ok=True);(STUDY/'logs').mkdir(exist_ok=True)
    if args.runner_log:
        path=args.runner_log.resolve()
        if not path.is_relative_to(ROOT):raise ValueError('Log must stay in this project.')
        sys.stdout=sys.stderr=path.open('a',encoding='utf-8',buffering=1)
    with exclusive_prediction_file(STUDY/'.runner.lock'):
        try:
            report('running')
            for name in NAMES:
                config=f'configs/coding_followup/{name}.json'
                run_stage(f'train-{name}',['training/train_coding_lora.py','--config',config],folder=STUDY)
                run_stage(f'validate-{name}',['evaluation/coding_benchmark.py','--config',config,'--variant','coding'],folder=STUDY)
                report('running')
            report('completed');set_status('completed','report-written',folder=STUDY)
        except KeyboardInterrupt:
            report('interrupted');print('Stopped; repeat to resume saved checkpoints and cases.');raise
        except Exception as error:
            current=read_json(STUDY/'study_status.json') or {}
            set_status('failed',current.get('stage','setup'),folder=STUDY,error=str(error));report('failed');raise

if __name__=='__main__':main()
