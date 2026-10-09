"""Continue a failed adapter using completion-only labels and joint head/body LoRA."""
import argparse
from functools import partial
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.benchmark import file_sha256, parse_calls, score_calls, run_generation
from evaluation.checkpoint import exclusive_prediction_file, write_json_atomic
from training.experiment_state import rows_sha256, recipe_signature, last_complete_checkpoint, progress_callback


def completion_example(row, tokenizer, max_length):
    if row['messages'][-1]['role'] != 'assistant':
        raise ValueError('The last message must be the reference assistant answer.')
    full = tokenizer.apply_chat_template(row['messages'],tools=row['tools'],enable_thinking=False,
        tokenize=True,add_generation_prompt=False,return_dict=True)
    prompt = tokenizer.apply_chat_template(row['messages'][:-1],tools=row['tools'],enable_thinking=False,
        tokenize=True,add_generation_prompt=True,return_dict=True)
    ids, start = full['input_ids'], len(prompt['input_ids'])
    if ids[:start] != prompt['input_ids'] or start >= len(ids):
        raise ValueError('Training text differs from the actual generation prompt.')
    if len(ids) > max_length:
        raise ValueError('A frozen repair example is overlong; refusing to truncate its target.')
    try:
        end = ids.index(tokenizer.eos_token_id,start)+1
    except ValueError:
        raise ValueError('Completion must contain the assistant EOS token.') from None
    return {'input_ids':ids,'attention_mask':full['attention_mask'],
            'labels':[-100]*start + ids[start:end] + [-100]*(len(ids)-end)}


def parameter_groups(model, head_lr, transformer_lr):
    groups = {'head': [], 'transformer': []}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if 'lora_' not in name:
                raise ValueError('A frozen Base parameter is unexpectedly trainable.')
            groups['head' if 'lm_head.lora_' in name else 'transformer'].append(parameter)
    if not all(groups.values()):
        raise ValueError('Joint repair requires transformer and output-head LoRA weights.')
    return [{'params':groups['head'],'lr':head_lr,'name':'head'},
            {'params':groups['transformer'],'lr':transformer_lr,'name':'transformer'}]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--config',type=Path,default=ROOT/'configs/protocol_repair/r16_pilot.json')
    cli.add_argument('--output-dir',type=Path)
    cli.add_argument('--prepare-only',action='store_true')
    cli.add_argument('--status',action='store_true')
    cli.add_argument('--stop-after-steps',type=int)
    args = cli.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    source = (ROOT/cfg['source_adapter']).resolve()
    output = args.output_dir.resolve() if args.output_dir else (ROOT/cfg['output_dir']).resolve()
    if not source.is_relative_to(ROOT) or not output.is_relative_to(ROOT) or source == output:
        raise ValueError('Source and separate repair output must stay within this project.')
    if args.status:
        path = output/'training_progress.json'
        print(path.read_text() if path.exists() else 'No repair training started.')
        return 0
    if args.stop_after_steps is not None and args.stop_after_steps < 1:
        cli.error('stop-after-steps must be positive.')
    output.mkdir(parents=True,exist_ok=True)
    with exclusive_prediction_file(output/'.training.lock'):
        original = json.loads((source/'training_manifest.json').read_text(encoding='utf-8'))
        train = [json.loads(line) for line in (source/'training_subset.jsonl').read_text(encoding='utf-8').splitlines()]
        validation = [json.loads(line) for line in (source/'validation.jsonl').read_text(encoding='utf-8').splitlines()]
        if rows_sha256(train) != original['train_sha256'] or rows_sha256(validation) != original['validation_sha256']:
            raise ValueError('Source frozen data changed.')
        if not 1 <= cfg['train_rows'] <= len(train):
            raise ValueError('Invalid repair subset size.')
        train = train[:cfg['train_rows']]
        effective = {**original['config'],**cfg,'output_dir':str(output.relative_to(ROOT))}
        manifest = {'config':effective,'source_adapter':str(source),'continuation':True,
                    'source_adapter_sha256':file_sha256(source/'adapter_model.safetensors'),
                    'source_config_sha256':file_sha256(source/'adapter_config.json'),
                    'base_files':original['base_files'],'train_rows':len(train),'validation_rows':len(validation),
                    'test_rows':original['test_rows'],'train_sha256':rows_sha256(train),
                    'validation_sha256':rows_sha256(validation),'test_sha256':original['test_sha256'],
                    'label_policy':'generation-prompt-masked-through-eos-v1',
                    'training_code_sha256':file_sha256(Path(__file__)),
                    'helper_code_sha256':{name:file_sha256(ROOT/name) for name in ('training/experiment_state.py','evaluation/checkpoint.py')},
                    'versions':{name:version(name) for name in ('torch','transformers','peft','trl','bitsandbytes')}}
        signature = recipe_signature(manifest)
        path = output/'training_manifest.json'
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError('Repair recipe changed; use a new output directory.')
        if not path.exists():
            if (output/'adapter_model.safetensors').exists():
                raise ValueError('Existing repair has no matching recipe.')
            write_json_atomic(path,manifest)
        if (output/'run_metrics.json').exists():
            metrics = json.loads((output/'run_metrics.json').read_text())
            if metrics['recipe_signature'] == signature and (output/'adapter_model.safetensors').exists():
                print('Repair training already completed; adapter preserved.')
                return 0
        from transformers import AutoTokenizer
        from datasets import Dataset
        tokenizer = AutoTokenizer.from_pretrained(source,local_files_only=True)
        train_ds = Dataset.from_list([completion_example(row,tokenizer,cfg['max_length']) for row in train])
        validation_ds = Dataset.from_list([completion_example(row,tokenizer,cfg['max_length']) for row in validation])
        for name,rows in (('training_subset.jsonl',train),('validation.jsonl',validation)):
            (output/name).write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')
        shutil.copyfile(source/'held_out_test.jsonl',output/'held_out_test.jsonl')
        (output/'training_example.txt').write_text(tokenizer.decode([v for v in train_ds[0]['labels'] if v != -100],skip_special_tokens=False),encoding='utf-8')
        print(f'Repair: {len(train)} training rows; {len(validation)} reserved internal validation rows; actual completions only.')
        if args.prepare_only:
            print('Completion masks and EOS supervision verified.')
            return 0
        import signal
        import torch
        from peft import PeftModel, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig, set_seed, TrainerCallback
        from trl import SFTConfig, SFTTrainer
        if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 12*2**30:
            raise RuntimeError('At least 12 GiB free CUDA memory is required.')
        if hasattr(signal,'SIGBREAK'):
            def interrupt(signum,frame):
                raise KeyboardInterrupt
            signal.signal(signal.SIGBREAK,interrupt)
        set_seed(cfg['seed'])
        base = AutoModelForCausalLM.from_pretrained(ROOT/effective['model_id'],local_files_only=True,
            dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa',
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16))
        base = prepare_model_for_kbit_training(base)
        model = PeftModel.from_pretrained(base,source,is_trainable=True,local_files_only=True)
        model.save_pretrained = partial(model.save_pretrained,save_embedding_layers=False)
        model.load_adapter = partial(model.load_adapter,autocast_adapter_dtype=False)
        callback = progress_callback(output,signature,args.stop_after_steps)

        class GenerationCheck(TrainerCallback):
            def on_step_end(self,training_args,state,control,model=None,**kwargs):
                if state.global_step % cfg['generation_check_every']:
                    return
                model.eval()
                records = []
                for index in (1,0,2,3):
                    row = validation[index]
                    record = run_generation(model,tokenizer,row,128)
                    predicted,error = parse_calls(record['output_text'])
                    expected = [{'name':c['function']['name'],'arguments':c['function']['arguments']} for c in row['messages'][-1]['tool_calls']]
                    record.update({'validation_index':index,'parse_error':error,**score_calls(predicted,expected,error)})
                    records.append(record)
                model.train()
                write_json_atomic(output/f'generation-step-{state.global_step}.json',records)
                print(f"Generation check step {state.global_step}: valid {sum(r['valid_tool_call'] for r in records)}/4; exact {sum(r['full_call_exact'] for r in records)}/4",flush=True)

        trainer = SFTTrainer(model=model,processing_class=tokenizer,train_dataset=train_ds,eval_dataset=validation_ds,
            callbacks=[callback,GenerationCheck()],args=SFTConfig(output_dir=str(output),max_steps=cfg['max_steps'],
                learning_rate=cfg['transformer_lr'],lr_scheduler_type='constant',weight_decay=0,
                per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
                max_length=cfg['max_length'],bf16=True,gradient_checkpointing=True,assistant_only_loss=False,
                eos_token='<|im_end|>',loss_type='nll',logging_steps=10,save_strategy='steps',save_steps=20,save_total_limit=3,
                eval_strategy='no',report_to='none',seed=cfg['seed'],dataloader_num_workers=0))
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        trainer.optimizer = torch.optim.AdamW(parameter_groups(model,cfg['head_lr'],cfg['transformer_lr']),weight_decay=0)
        model.print_trainable_parameters()
        checkpoint = last_complete_checkpoint(output,signature)
        print('Resume checkpoint:',checkpoint or 'none; continuation from saved failed r16 adapter')
        try:
            result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
        except KeyboardInterrupt:
            callback.save(trainer.state,'interrupted')
            print('Stopped; re-run the same repair command to resume the latest complete checkpoint.')
            return 130
        if result.global_step < trainer.state.max_steps:
            callback.save(trainer.state,'interrupted')
            print('Deliberate pause saved; repeat without --stop-after-steps to continue.')
            return 130
        progress = json.loads((output/'training_progress.json').read_text())
        validation_metrics = trainer.evaluate()
        trainer.save_model(str(output))
        tokenizer.save_pretrained(output)
        metrics = {**result.metrics,'recipe_signature':signature,'optimizer_steps':result.global_step,
                   'source_adapter':str(source),'continuation':True,'training_seconds':progress['training_seconds'],
                   'peak_allocated_vram_gib':progress['peak_allocated_vram_gib'],'validation':validation_metrics,
                   'adapter_bytes':(output/'adapter_model.safetensors').stat().st_size,
                   'trainable_dtype':'float32','label_policy':manifest['label_policy']}
        write_json_atomic(output/'run_metrics.json',metrics)
        callback.save(trainer.state,'completed')
        print('Repair saved. Use generated internal validation before a new official test.')
        print(json.dumps(metrics,indent=2))
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
