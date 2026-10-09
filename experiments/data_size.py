"""Controlled 1k/3k/full data study on fresh r8; two epochs, fixed validation."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.benchmark import file_sha256,parse_calls,score_calls,summarize
from evaluation.checkpoint import exclusive_prediction_file,write_json_atomic
from experiments.rank_ablation import read_json,run_stage,set_status,process_alive,BASE_CACHE
from experiments.fair_rank import finalize_finished_checkpoint
from training.experiment_state import rows_sha256,recipe_signature
from training.train_tool_lora import load_splits

STUDY=ROOT/'outputs/data-size-r8'
BASELINE=ROOT/'outputs/fair-rank-completion-1k/r8'
NAMES=('1k','3k','full')
COUNTS=(1000,3000,8258)
RULE='Highest 100-case validation exact match; then strict mock success; then fewer training rows.'
COMMON_KEYS=('model_id','base_model_id','dataset_id','max_length','epochs','gradient_accumulation_steps',
             'head_lr','transformer_lr','lora_r','lora_alpha','lora_dropout','target_modules','seed')


def configurations():
    configs=[read_json(ROOT/f'configs/data_size/{name}.json') for name in NAMES]
    original=read_json(BASELINE/'training_manifest.json')['config']
    for name,count,cfg in zip(NAMES,COUNTS,configs):
        if any(cfg[key]!=original[key] for key in COMMON_KEYS):
            raise ValueError('Study settings differ from the completed 1k baseline.')
        if cfg['training_rows']!=count or cfg['max_steps']!=count or cfg['epochs']!=2 or cfg['gradient_accumulation_steps']!=2:
            raise ValueError('This study requires two complete epochs at effective batch two.')
        if cfg['output_dir']!=f'outputs/data-size-r8/{name}' or cfg['data_source']!=str(BASELINE.relative_to(ROOT)).replace('\\','/'):
            raise ValueError('Unexpected study output or baseline path.')
    return configs


def verify_baseline_and_splits(configs):
    manifest=read_json(BASELINE/'training_manifest.json')
    metrics=read_json(BASELINE/'run_metrics.json')
    if not manifest['fresh_base'] or not metrics['fresh_base_initialization_verified'] or metrics['optimizer_steps']!=1000 or metrics['recipe_signature']!=recipe_signature(manifest):
        raise ValueError('The reference adapter is not a completed matching fresh-Base run.')
    if file_sha256(ROOT/'training/train_fresh_protocol.py')!=manifest['training_code_sha256']:
        raise ValueError('The original baseline training engine changed.')
    from importlib.metadata import version
    for name,expected in manifest['versions'].items():
        if version(name)!=expected:
            raise ValueError(f'Baseline software version differs: {name}.')
    for name,expected in manifest['helper_code_sha256'].items():
        if file_sha256(ROOT/name)!=expected:
            raise ValueError(f'Baseline helper changed: {name}.')
    old=(ROOT/'training/train_fresh_protocol.py').read_text(encoding='utf-8')
    new=(ROOT/'training/train_data_size.py').read_text(encoding='utf-8')
    marker='        import hashlib'
    if old[old.index(marker):]!=new[new.index(marker):]:
        raise ValueError('The data-size training kernel differs from the verified baseline kernel.')
    train,validation,test=load_splits(configs[0])
    if len(train)!=8258 or rows_sha256(train[:1000])!=manifest['train_sha256'] or rows_sha256(validation[:100])!=manifest['validation_sha256'] or rows_sha256(test)!=manifest['test_sha256']:
        raise ValueError('Raw data does not reproduce the completed 1k baseline.')
    def fingerprints(rows):
        return {json.dumps({key:value for key,value in row.items() if key!='metadata'},sort_keys=True) for row in rows}
    splits=[fingerprints(rows) for rows in (train,validation[:100],test)]
    if any(splits[i]&splits[j] for i,j in ((0,1),(0,2),(1,2))):
        raise ValueError('Training, validation and official test contain identical examples.')
    summary=read_json(BASELINE/'evaluation/validation-summary.json')
    if summary['examples']!=100 or summary['data_sha256']!=manifest['validation_sha256']:
        raise ValueError('The cached baseline validation belongs to different cases.')
    records=[json.loads(line) for line in (BASELINE/'evaluation/validation-100-tokens256-predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(records)!=100 or {r['test_index'] for r in records}!=set(range(100)):
        raise ValueError('Baseline validation cache is incomplete or duplicated.')
    from runtime.mock_tools import MockWorld
    for record in records:
        row=validation[record['test_index']]
        expected=[{'name':c['function']['name'],'arguments':c['function']['arguments']} for c in row['messages'][-1]['tool_calls']]
        calls,error=parse_calls(record['output_text'])
        scores=score_calls(calls,expected,error)
        mock_ok=error is None and MockWorld(row['tools']).execute(calls)['ok']
        if record['prompt']!=row['messages'][-2]['content'] or record['expected_calls']!=expected or calls!=record['predicted_calls'] or any(record[k]!=v for k,v in scores.items()) or mock_ok!=record['mock_execution_ok']:
            raise ValueError('Cached baseline validation differs from its frozen questions or current scoring.')
    rescored=summarize(records)
    if any(summary['metrics'][key]!=value for key,value in rescored.items()) or summary['metrics']['mock_execution_success_rate']!=sum(r['mock_execution_ok'] for r in records)/len(records):
        raise ValueError('Cached baseline validation summary differs from its saved predictions.')
    return {'training_rows':dict(zip(NAMES,COUNTS)),'validation_rows':100,'test_rows':961,
            'train_sha256':{name:rows_sha256(train[:count]) for name,count in zip(NAMES,COUNTS)},
            'validation_sha256':rows_sha256(validation[:100]),'test_sha256':rows_sha256(test),
            'source_sha256':file_sha256(ROOT/configs[0]['dataset_path']),
            'baseline_kernel_sha256':manifest['training_code_sha256'],'kernel_preserved':True,
            'baseline_adapter_sha256':file_sha256(BASELINE/'adapter_model.safetensors'),
            'baseline_manifest_sha256':file_sha256(BASELINE/'training_manifest.json'),
            'baseline_validation_cache_sha256':file_sha256(BASELINE/'evaluation/validation-100-tokens256-predictions.jsonl'),
            'base_files':manifest['base_files'],'versions':manifest['versions'],'label_policy':manifest['label_policy'],
            'selection_rule':RULE,'comparison_budget':'fixed epochs; total optimizer updates increase with data size'}


def folder_for(name):
    return BASELINE if name=='1k' else STUDY/name


def select_size(results):
    if len(results)!=3 or any(not r['validation'] for r in results):
        raise ValueError('Every data size needs complete validation before selection.')
    return max(results,key=lambda r:(r['validation']['metrics']['full_call_exact_match_rate'],
                                    r['validation']['metrics']['mock_execution_success_rate'],-r['training_rows']))['name']


def results():
    return [{'name':name,'training_rows':count,'adapter':str(folder_for(name)),
             'reused_completed_baseline':name=='1k','training':read_json(folder_for(name)/'run_metrics.json'),
             'validation':read_json(folder_for(name)/'evaluation/validation-summary.json')}
            for name,count in zip(NAMES,COUNTS)]


def report(status,data):
    current=results()
    trained=[r['training'] for r in current if r['training']]
    if trained and len({m['initial_base_logits_sha256'] for m in trained})!=1:
        raise ValueError('Fresh initial Base logits differ across data sizes.')
    selected=select_size(current) if all(r['validation'] for r in current) else None
    test=read_json(folder_for(selected)/'evaluation/test-summary.json') if selected else None
    value={'status':status,'data':data,'results':current,'selected_size':selected,'selection_rule':RULE,
           'test':test,'baseline_reused':True,'fixed_epochs':2,'seed':42}
    write_json_atomic(STUDY/'comparison.json',value)
    lines=['# Controlled training-data comparison: fresh r8','',f'Status: {status}.','',
        'Same Qwen3-8B-Base, new zero-B r8 LoRA, FP32 adapter weights, NF4/BF16 computation, seed42 and completion-through-EOS loss.',
        'Rank8 / alpha16 / dropout0.05; seven transformer projections plus lm_head; constant head/body LR1e-3/1e-4; effective batch2.',
        'Nested seeded training subsets: 1,000 ⊂ 3,000 ⊂ all8,258 rows. The same100 internal-validation and961 official-test cases stay excluded from training.',
        'Two epochs per condition. This is a fixed-epoch comparison, not a fixed-compute experiment: 1,000 /3,000 /8,258 optimizer updates.',
        'The completed 1k fresh r8 is reused only after verifying settings, frozen data, software, helpers and the identical training kernel.','',
        '| Training rows | Updates | Training min | Peak allocated GiB | Validation exact | Format | Mock |','|---|---:|---:|---:|---:|---:|---:|']
    for row in current:
        m,v=row['training'],row['validation']
        cost=[str(m['optimizer_steps']),f"{m['training_seconds']/60:.1f}",f"{m['peak_allocated_vram_gib']:.2f}"] if m else ['pending']*3
        quality=[f"{v['metrics'][key]:.1%}" for key in ('full_call_exact_match_rate','valid_tool_call_rate','mock_execution_success_rate')] if v else ['pending']*3
        lines.append(f"| {row['training_rows']:,}{' (reused)' if row['name']=='1k' else ''} | "+' | '.join(cost+quality)+' |')
    lines+=['',RULE,'',f"Selected size: {selected or 'pending'}.",
        'Selection uses validation only. The 1k official result is already known from the rank study; it is not used to select a size.',
        'Only the selected condition gets a new official test. If 1k wins, its verified saved test is reused.']
    if test:
        lines+=['',f"Selected official exact match: {test['metrics']['full_call_exact_match_rate']:.2%}.",
                f"Format: {test['metrics']['valid_tool_call_rate']:.2%}; mock: {test['metrics']['mock_execution_success_rate']:.2%}."]
    lines+=['','Single seed /100-case validation: repeat seeds before treating small differences as a general data-size effect.',
            'More data also changes the total number of updates. Desktop workload and thermals make wall times unsuitable as a controlled speed comparison.',
            'Recorded time includes checkpoints and early generation checks. The reused 1k run also had pilot pauses/checks.',
            'Mock success measures executability, not semantic correctness. Original test rows must never be added to training.','']
    (STUDY/'comparison.md').write_text('\n'.join(lines),encoding='utf-8')
    return value


def snapshot():
    status=read_json(STUDY/'study_status.json')
    if status and status['status']=='running':
        status['runner_alive']=process_alive(status.get('runner_pid'))
        if status['runner_alive'] is not True:
            status['status']='interrupted' if status['runner_alive'] is False else 'unknown'
    return {'study':status,'conditions':[{'name':name,'training_rows':count,'reused':name=='1k',
        'training':read_json(folder_for(name)/'training_progress.json'),
        'validation':read_json(folder_for(name)/'evaluation/validation-100-tokens256-progress.json'),
        'test':read_json(folder_for(name)/'evaluation/test-961-tokens256-progress.json')}
        for name,count in zip(NAMES,COUNTS)],'selection':read_json(STUDY/'selection.json')}


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--prepare-only',action='store_true')
    cli.add_argument('--status',action='store_true')
    cli.add_argument('--runner-log',type=Path)
    args=cli.parse_args()
    if args.status:
        print(json.dumps(snapshot(),indent=2))
        return 0
    configs=configurations()
    STUDY.mkdir(parents=True,exist_ok=True)
    (STUDY/'logs').mkdir(exist_ok=True)
    if args.runner_log:
        path=args.runner_log.resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Queue log must stay within this project.')
        path.parent.mkdir(parents=True,exist_ok=True)
        sys.stdout=sys.stderr=path.open('a',encoding='utf-8',buffering=1)
    data=None
    with exclusive_prediction_file(STUDY/'.runner.lock'):
        try:
            data=verify_baseline_and_splits(configs)
            prior=read_json(STUDY/'study_plan.json')
            if prior and prior!=data:
                raise ValueError('Frozen study protocol changed; use a new study directory.')
            write_json_atomic(STUDY/'study_plan.json',data)
            for name in NAMES[1:]:
                run_stage(f'prepare-{name}',['training/train_data_size.py','--config',f'configs/data_size/{name}.json','--prepare-only'],folder=STUDY)
                manifest=read_json(STUDY/name/'training_manifest.json')
                for key in ('validation_sha256','test_sha256','source_sha256','base_files','versions','label_policy'):
                    if manifest[key]!=data[key]:
                        raise ValueError(f'Prepared condition differs: {name}/{key}.')
                if manifest['train_sha256']!=data['train_sha256'][name]:
                    raise ValueError('Prepared training subset differs from the protocol.')
            report('prepared' if args.prepare_only else 'running',data)
            if args.prepare_only:
                set_status('prepared','data-verified',folder=STUDY)
                print('Baseline reuse, nested subsets, prompt/EOS masks and all three split boundaries verified.')
                return 0
            for name,cfg in zip(NAMES[1:],configs[1:]):
                finalize_finished_checkpoint(STUDY/name,cfg)
                run_stage(f'train-{name}',['training/train_data_size.py','--config',f'configs/data_size/{name}.json'],folder=STUDY)
                report('running',data)
                run_stage(f'validate-{name}',['evaluation/rank_validation.py','--adapter',str(STUDY/name)],folder=STUDY)
                report('running',data)
            value=report('validation-completed',data)
            selected=value['selected_size']
            selection={'size':selected,'training_rows':COUNTS[NAMES.index(selected)],'rule':RULE,'validation_sha256':data['validation_sha256']}
            prior=read_json(STUDY/'selection.json')
            if prior and prior!=selection:
                raise ValueError('Validation selection changed after freezing it.')
            write_json_atomic(STUDY/'selection.json',selection)
            run_stage(f'test-{selected}',['evaluation/rank_validation.py','--adapter',str(folder_for(selected)),'--split','test','--reuse-base',str(BASE_CACHE)],folder=STUDY)
            report('completed',data)
            set_status('completed','report-written',folder=STUDY,selected_size=selected)
            return 0
        except KeyboardInterrupt:
            report('interrupted',data)
            print('Stopped. Repeat this command to resume saved checkpoints and evaluation cases.')
            return 130
        except Exception as error:
            current=read_json(STUDY/'study_status.json') or {}
            set_status('failed',current.get('stage','setup'),folder=STUDY,error=str(error))
            report('failed',data)
            raise


if __name__=='__main__':
    raise SystemExit(main())
