"""Resume protocol repair, validate generation, then test only if validation passes."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.checkpoint import exclusive_prediction_file, write_json_atomic
from experiments.rank_ablation import read_json, run_stage, set_status, process_alive, BASE_CACHE
from evaluation.benchmark import file_sha256
from training.experiment_state import recipe_signature, last_complete_checkpoint


def recover_finished_checkpoint(folder,cfg):
    """Finalize a max-step checkpoint after interruption during post-training work."""
    if (folder/'run_metrics.json').exists() or not (folder/'training_manifest.json').exists():
        return False
    with exclusive_prediction_file(folder/'.training.lock'):
        manifest = read_json(folder/'training_manifest.json')
        if manifest['training_code_sha256'] != file_sha256(ROOT/'training/repair_protocol.py'):
            raise ValueError('Repair code changed; use a new output directory.')
        for key,value in cfg.items():
            if key != 'output_dir' and manifest['config'].get(key) != value:
                raise ValueError('Repair settings changed; refusing checkpoint finalization.')
        source = ROOT/cfg['source_adapter']
        if manifest['source_adapter_sha256'] != file_sha256(source/'adapter_model.safetensors'):
            raise ValueError('Source adapter changed.')
        signature = recipe_signature(manifest)
        checkpoint = last_complete_checkpoint(folder,signature)
        if checkpoint is None:
            return False
        state = read_json(checkpoint/'trainer_state.json')
        if state['global_step'] != cfg['max_steps']:
            return False
        names = ('adapter_config.json','adapter_model.safetensors','tokenizer.json','tokenizer_config.json','chat_template.jinja')
        if not all((checkpoint/name).is_file() for name in names):
            raise ValueError('Completed checkpoint lacks tokenizer/model files.')
        for name in names:
            shutil.copyfile(checkpoint/name,folder/name)
        progress = read_json(folder/'training_progress.json') or {}
        write_json_atomic(folder/'run_metrics.json',{
            'recipe_signature':signature,'optimizer_steps':state['global_step'],
            'source_adapter':str(source.resolve()),'continuation':True,
            'training_seconds':progress.get('training_seconds'),
            'peak_allocated_vram_gib':progress.get('peak_allocated_vram_gib'),
            'adapter_bytes':(folder/'adapter_model.safetensors').stat().st_size,
            'trainable_dtype':'float32','label_policy':manifest['label_policy'],
            'finalized_from_checkpoint':str(checkpoint),'validation':None,
        })
        write_json_atomic(folder/'training_progress.json',{**progress,'status':'completed'})
        print('Recovered the completed training checkpoint without any additional optimizer update.')
        return True


def validation_passes(summary):
    metrics = summary['metrics']
    return (metrics['valid_tool_call_rate'] >= .95 and metrics['tool_selection_accuracy'] >= .95
            and metrics['mock_execution_success_rate'] >= .95 and metrics['token_limit_hit_rate'] <= .05)


def report(folder,source):
    training = read_json(folder/'run_metrics.json')
    before = read_json(source/'evaluation/validation-summary.json')
    validation = read_json(folder/'evaluation/validation-summary.json')
    test = read_json(folder/'evaluation/test-summary.json')
    manifest = read_json(folder/'training_manifest.json')
    cfg = manifest['config'] if manifest else {}
    summary = {'training':training,'before_validation':before,'validation':validation,'test':test,
               'continuation':True,'source_adapter':str(source),
               'validation_gate_passed':validation_passes(validation) if validation else None}
    write_json_atomic(folder/'repair-report.json',summary)
    lines = ['# Protocol repair of the failed fresh r16 adapter','',
        'This continues the original 1k-data r16 adapter; it is excluded from the fresh rank comparison.',
        f"Joint transformer/head LoRA, FP32 trainable weights, constant head LR {cfg.get('head_lr')} and transformer LR {cfg.get('transformer_lr')}.",
        f"{cfg.get('max_steps')} updates, batch 1 × accumulation {cfg.get('gradient_accumulation_steps')}, seeded {cfg.get('train_rows')}-row subset of the original training data.",
        'Loss masks the exact generation prompt (including prefilled assistant/think tokens) and supervises the completion through EOS.',
        'All generated validation and test scores use the unchanged inference loader, greedy decoding and 256-token limit.','',
        '| Internal validation metric | Original r16 | Repaired r16 |','|---|---:|---:|']
    for key,label in (('valid_tool_call_rate','Valid format'),('tool_selection_accuracy','Correct tools/counts'),
                      ('full_call_exact_match_rate','Exact full calls'),('mock_execution_success_rate','Strict mock execution'),
                      ('token_limit_hit_rate','Token limit reached')):
        a = f"{before['metrics'][key]:.1%}" if before else 'pending'
        b = f"{validation['metrics'][key]:.1%}" if validation else 'pending'
        lines.append(f'| {label} | {a} | {b} |')
    if test:
        lines += ['', '## All 961 official cases','', '| Metric | Cached Base | Repaired r16 |','|---|---:|---:|']
        for key,label in (('valid_tool_call_rate','Valid format'),('tool_selection_accuracy','Correct tools/counts'),('full_call_exact_match_rate','Exact full calls')):
            lines.append(f"| {label} | {test['base_metrics'][key]:.2%} | {test['metrics'][key]:.2%} |")
        lines += ['',f"Strict mock execution: {test['metrics']['mock_execution_success_rate']:.2%}."]
    else:
        lines += ['', 'The new 961-case result is pending.']
    lines += ['', 'Several repair settings changed together; this validates the combined repair and does not isolate a single cause.',
              'The old rank-study outputs and earlier repaired full-data r16 demo adapter are preserved.',
              'Strict exact match can penalize equivalent wording. Mock execution does not establish semantic correctness.','']
    (folder/'repair-report.md').write_text('\n'.join(lines),encoding='utf-8')


def snapshot(folder):
    status = read_json(folder/'study_status.json')
    if status and status['status'] == 'running':
        status['runner_alive'] = process_alive(status.get('runner_pid'))
        if status['runner_alive'] is not True:
            status['status'] = 'interrupted' if status['runner_alive'] is False else 'unknown'
    return {'pipeline':status,'training':read_json(folder/'training_progress.json'),
            'validation':read_json(folder/'evaluation/validation-100-tokens256-progress.json'),
            'test':read_json(folder/'evaluation/test-961-tokens256-progress.json'),
            'validation_summary':read_json(folder/'evaluation/validation-summary.json'),
            'test_summary':read_json(folder/'evaluation/test-summary.json')}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--config',type=Path,default=ROOT/'configs/protocol_repair/r16_pilot.json')
    cli.add_argument('--status',action='store_true')
    cli.add_argument('--runner-log',type=Path)
    args = cli.parse_args()
    cfg = read_json(args.config)
    folder,source = (ROOT/cfg['output_dir']).resolve(),(ROOT/cfg['source_adapter']).resolve()
    if not folder.is_relative_to(ROOT) or not source.is_relative_to(ROOT) or folder == source:
        raise ValueError('Source and separate output must stay within this project.')
    if args.status:
        print(json.dumps(snapshot(folder),indent=2))
        return 0
    if args.runner_log:
        path = args.runner_log.resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Queue log must stay within this project.')
        path.parent.mkdir(parents=True,exist_ok=True)
        sys.stdout = sys.stderr = path.open('a',encoding='utf-8',buffering=1)
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'logs').mkdir(exist_ok=True)
    with exclusive_prediction_file(folder/'.pipeline.lock'):
        try:
            recover_finished_checkpoint(folder,cfg)
            run_stage('repair-train',['training/repair_protocol.py','--config',str(args.config.resolve())],folder=folder)
            report(folder,source)
            run_stage('repair-validation',['evaluation/rank_validation.py','--adapter',str(folder)],folder=folder)
            validation = read_json(folder/'evaluation/validation-summary.json')
            report(folder,source)
            if not validation_passes(validation):
                set_status('validation_failed','repair-validation',folder=folder)
                print('Generated validation did not meet the gate; official test was not started.')
                return 2
            run_stage('repair-test',['evaluation/rank_validation.py','--adapter',str(folder),'--split','test','--reuse-base',str(BASE_CACHE)],folder=folder)
            report(folder,source)
            set_status('completed','report-written',folder=folder)
            print(f"Completed: {folder/'repair-report.md'}")
            return 0
        except KeyboardInterrupt:
            print('Stopped. Re-run the same queue command to resume saved work.')
            return 130


if __name__ == '__main__':
    raise SystemExit(main())
