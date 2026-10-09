"""Corrected fresh-Base rank comparison with a small generation gate before long runs."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.checkpoint import exclusive_prediction_file,write_json_atomic
from experiments.rank_ablation import RANKS,BASE_CACHE,read_json,run_stage,set_status,process_alive,select_rank,SELECTION_RULE
from training.experiment_state import recipe_signature,last_complete_checkpoint,rows_sha256

STUDY=ROOT/'outputs/fair-rank-completion-1k'


def configurations():
    configs=[read_json(ROOT/f'configs/fair_rank/r{rank}.json') for rank in RANKS]
    common=[{key:value for key,value in cfg.items() if key not in {'lora_r','lora_alpha','output_dir'}} for cfg in configs]
    if any(cfg != common[0] for cfg in common[1:]):
        raise ValueError('Conditions differ beyond rank, alpha and output path.')
    for rank,cfg in zip(RANKS,configs):
        if cfg['lora_r'] != rank or cfg['lora_alpha'] != rank*2 or cfg['max_steps'] != 1000 or cfg['gradient_accumulation_steps'] != 2 or cfg['output_dir'] != f'outputs/fair-rank-completion-1k/r{rank}':
            raise ValueError('Unexpected corrected rank-study condition.')
    return configs


def verify_data():
    manifests=[read_json(STUDY/f'r{rank}/training_manifest.json') for rank in RANKS]
    keys=('train_rows','validation_rows','test_rows','train_sha256','validation_sha256','test_sha256','base_files','label_policy','training_code_sha256','helper_code_sha256','versions')
    if not all(m['fresh_base'] is True for m in manifests) or any(any(m[key] != manifests[0][key] for key in keys) for m in manifests[1:]):
        raise ValueError('Frozen data, Base identity, label policy or software differs across ranks.')
    return {key:manifests[0][key] for key in keys}


def pilot_indices(rows):
    selected=list(range(min(8,len(rows))))
    multi=[i for i,row in enumerate(rows) if len(row['messages'][-1]['tool_calls'])>1 and i not in selected]
    selected.extend(multi[:8])
    if len(selected)<16:
        raise ValueError('The fixed pilot needs eight additional multi-call validation cases.')
    return selected


def pilot_passes(summary):
    m=summary['metrics']
    return m['valid_tool_call_rate']>=.9 and m['tool_selection_accuracy']>=.9 and m['mock_execution_success_rate']>=.9 and m['token_limit_hit_rate']==0


def export_pilot(folder,checkpoint,step=40):
    target=folder/f'pilot-{step}'
    if (target/'training_manifest.json').exists():
        return target
    target.mkdir(exist_ok=True)
    for name in ('adapter_config.json','adapter_model.safetensors','tokenizer.json','tokenizer_config.json','chat_template.jinja'):
        shutil.copyfile(checkpoint/name,target/name)
    manifest=read_json(folder/'training_manifest.json')
    rows=[json.loads(line) for line in (folder/'validation.jsonl').read_text(encoding='utf-8').splitlines()]
    indices=pilot_indices(rows)
    subset=[rows[index] for index in indices]
    manifest={**manifest,'validation_rows':len(subset),'validation_sha256':rows_sha256(subset),
              'pilot_optimizer_steps':step,'original_validation_sha256':manifest['validation_sha256'],'original_validation_indices':indices}
    (target/'validation.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in subset),encoding='utf-8')
    write_json_atomic(target/'training_manifest.json',manifest)
    return target


def finalize_finished_checkpoint(folder,cfg):
    if (folder/'run_metrics.json').exists():
        return
    manifest=read_json(folder/'training_manifest.json')
    checkpoint=last_complete_checkpoint(folder,recipe_signature(manifest))
    if checkpoint is None or read_json(checkpoint/'trainer_state.json')['global_step'] != cfg['max_steps']:
        return
    # An interruption during final export must not perform update 1001.
    with exclusive_prediction_file(folder/'.training.lock'):
        for name in ('adapter_config.json','adapter_model.safetensors','tokenizer.json','tokenizer_config.json','chat_template.jinja'):
            shutil.copyfile(checkpoint/name,folder/name)
        progress=read_json(folder/'training_progress.json')
        initial=read_json(folder/'initialization.json')
        from safetensors import safe_open
        with safe_open(str(folder/'adapter_model.safetensors'),framework='pt',device='cpu') as state:
            trainable=sum(__import__('math').prod(state.get_slice(key).get_shape()) for key in state.keys())
        write_json_atomic(folder/'run_metrics.json',{'recipe_signature':recipe_signature(manifest),'optimizer_steps':cfg['max_steps'],
            'training_seconds':progress['training_seconds'],'peak_allocated_vram_gib':progress['peak_allocated_vram_gib'],
            'trainable_parameters':trainable,'adapter_bytes':(folder/'adapter_model.safetensors').stat().st_size,
            **initial,'adapter_reload_max_logit_difference':None,'finalized_from_checkpoint':str(checkpoint)})
        write_json_atomic(folder/'training_progress.json',{**progress,'status':'completed'})


def report(status,data):
    results=[{'rank':rank,'training':read_json(STUDY/f'r{rank}/run_metrics.json'),
              'validation':read_json(STUDY/f'r{rank}/evaluation/validation-summary.json')} for rank in RANKS]
    trained=[r['training'] for r in results if r['training']]
    if trained and len({m['initial_base_logits_sha256'] for m in trained}) != 1:
        raise ValueError('Initial Base logits differ across ranks.')
    selected=select_rank(results) if all(r['validation'] for r in results) else None
    test=read_json(STUDY/f'r{selected}/evaluation/test-summary.json') if selected else None
    pilots={str(step):read_json(STUDY/f'r8/pilot-{step}/evaluation/validation-summary.json') for step in (40,100,200)}
    value={'status':status,'data':data,'results':results,'selected_rank':selected,'selection_rule':SELECTION_RULE,'test':test,
           'pilots':pilots}
    write_json_atomic(STUDY/'comparison.json',value)
    lines=['# Corrected fresh-Base rank comparison','',f'Status: {status}.','',
           'Every rank starts from the same frozen Qwen3-8B-Base with zero-initialized LoRA B; no earlier adapter is loaded.',
           'Fixed 1,000 train / 100 internal validation / 961 official test cases; seed 42; two epochs (1,000 optimizer updates).',
           'Batch 1 × accumulation 2; completion-only-through-EOS loss; FP32 trainable LoRA; NF4/BF16 Base computation.',
           'Constant LR 1e-3 for the output head and 1e-4 for transformer LoRA; dropout .05; alpha/rank = 2.',
           'Targets: all seven transformer projections plus lm_head. Only rank, alpha and output path vary.','',
           '| Rank | Updates | Trainable parameters | Adapter MiB | Training min | Peak GiB | Validation exact | Format | Mock |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        m,v=r['training'],r['validation']
        costs=[str(m['optimizer_steps']),f"{m['trainable_parameters']:,}",f"{m['adapter_bytes']/2**20:.1f}",f"{m['training_seconds']/60:.1f}",f"{m['peak_allocated_vram_gib']:.2f}"] if m else ['pending']*5
        quality=[f"{v['metrics'][key]:.1%}" for key in ('full_call_exact_match_rate','valid_tool_call_rate','mock_execution_success_rate')] if v else ['pending']*3
        lines.append(f"| {r['rank']} | "+' | '.join(costs+quality)+' |')
    lines += ['',SELECTION_RULE,'',f"Selected rank: {selected or 'pending'}.",
              'Selection uses all 100 validation cases, not the small generation pilot. Only the winner gets the official test.',
              'The original failed rank study and repaired warm-start r16 remain separate historical experiments.']
    if test:
        lines += ['',f"Selected adapter official exact match: {test['metrics']['full_call_exact_match_rate']:.2%}.",
                  f"Format: {test['metrics']['valid_tool_call_rate']:.2%}; mock: {test['metrics']['mock_execution_success_rate']:.2%}."]
    lines += ['', 'Single-seed pilot: small score differences need repeat seeds before a general rank conclusion.',
              'Training time includes checkpoints and early generation checks; excludes model loading and final reload verification.',
              'Desktop workload and thermals affect speed; peak memory is PyTorch allocated memory.','']
    (STUDY/'comparison.md').write_text('\n'.join(lines),encoding='utf-8')
    return value


def snapshot():
    status=read_json(STUDY/'study_status.json')
    if status and status['status']=='running':
        status['runner_alive']=process_alive(status.get('runner_pid'))
        if status['runner_alive'] is not True:
            status['status']='interrupted' if status['runner_alive'] is False else 'unknown'
    return {'study':status,'ranks':[{'rank':r,'training':read_json(STUDY/f'r{r}/training_progress.json'),
        'validation':read_json(STUDY/f'r{r}/evaluation/validation-100-tokens256-progress.json'),
        'test':read_json(STUDY/f'r{r}/evaluation/test-961-tokens256-progress.json')} for r in RANKS],
        'pilots':{str(step):read_json(STUDY/f'r8/pilot-{step}/evaluation/validation-summary.json') for step in (40,100,200)}}


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--status',action='store_true')
    cli.add_argument('--prepare-only',action='store_true')
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
            raise ValueError('Queue log must stay within the project.')
        path.parent.mkdir(parents=True,exist_ok=True)
        sys.stdout=sys.stderr=path.open('a',encoding='utf-8',buffering=1)
    data=None
    with exclusive_prediction_file(STUDY/'.runner.lock'):
        try:
            for rank in RANKS:
                run_stage(f'prepare-r{rank}',['training/train_fresh_protocol.py','--config',f'configs/fair_rank/r{rank}.json','--prepare-only'],folder=STUDY)
            data=verify_data()
            write_json_atomic(STUDY/'study_data.json',data)
            report('prepared' if args.prepare_only else 'running',data)
            if args.prepare_only:
                set_status('prepared','data-verified',folder=STUDY)
                return 0
            first=STUDY/'r8'
            pilot_ok=False
            for target_step in (40,100,200):
                summary_path=first/f'pilot-{target_step}/evaluation/validation-summary.json'
                if not summary_path.exists():
                    signature=recipe_signature(read_json(first/'training_manifest.json'))
                    checkpoint=last_complete_checkpoint(first,signature)
                    step=read_json(checkpoint/'trainer_state.json')['global_step'] if checkpoint else 0
                    if step<target_step:
                        run_stage(f'pilot-train-r8-{target_step}',['training/train_fresh_protocol.py','--config','configs/fair_rank/r8.json','--stop-after-steps',str(target_step-step)],folder=STUDY,accepted_exit_codes=(0,130))
                    checkpoint=first/f'checkpoint-{target_step}'
                    if not (checkpoint/'checkpoint_complete.json').exists():
                        raise ValueError(f'The completed {target_step}-step pilot checkpoint is missing.')
                    pilot=export_pilot(first,checkpoint,target_step)
                    run_stage(f'pilot-validate-r8-{target_step}',['evaluation/rank_validation.py','--adapter',str(pilot)],folder=STUDY)
                pilot_summary=read_json(summary_path)
                if pilot_passes(pilot_summary):
                    pilot_ok=True
                    break
            if not pilot_ok:
                report('pilot-failed',data)
                set_status('pilot_failed','pilot-validate-r8',folder=STUDY)
                print('The fixed pilot failed through 200 updates; the full three-rank runs were not started.')
                return 2
            for rank,cfg in zip(RANKS,configs):
                finalize_finished_checkpoint(STUDY/f'r{rank}',cfg)
                run_stage(f'train-r{rank}',['training/train_fresh_protocol.py','--config',f'configs/fair_rank/r{rank}.json'],folder=STUDY)
                report('running',data)
                run_stage(f'validate-r{rank}',['evaluation/rank_validation.py','--adapter',str(STUDY/f'r{rank}')],folder=STUDY)
                report('running',data)
            value=report('validation-completed',data)
            selected=value['selected_rank']
            selection={'rank':selected,'rule':SELECTION_RULE,'validation_sha256':data['validation_sha256']}
            prior=read_json(STUDY/'selection.json')
            if prior and prior != selection:
                raise ValueError('Frozen validation selection changed.')
            write_json_atomic(STUDY/'selection.json',selection)
            run_stage(f'test-r{selected}',['evaluation/rank_validation.py','--adapter',str(STUDY/f'r{selected}'),'--split','test','--reuse-base',str(BASE_CACHE)],folder=STUDY)
            report('completed',data)
            set_status('completed','report-written',folder=STUDY,selected_rank=selected)
            return 0
        except KeyboardInterrupt:
            report('interrupted',data)
            print('Stopped. Repeat this queue command to resume checkpoints and completed evaluation cases.')
            return 130
        except Exception as error:
            current=read_json(STUDY/'study_status.json') or {}
            set_status('failed',current.get('stage','setup'),folder=STUDY,error=str(error))
            report('failed',data)
            raise


if __name__=='__main__':
    raise SystemExit(main())
