"""Fresh-Base LoRA training for the corrected, fixed-data rank experiment."""
import argparse
from functools import partial
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.benchmark import file_sha256,run_generation,parse_calls,score_calls
from evaluation.checkpoint import exclusive_prediction_file,write_json_atomic
from training.train_tool_lora import prepare_tokenizer
from training.repair_protocol import completion_example,parameter_groups
from training.experiment_state import rows_sha256,recipe_signature,last_complete_checkpoint,progress_callback


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--config',type=Path,required=True)
    cli.add_argument('--prepare-only',action='store_true')
    cli.add_argument('--stop-after-steps',type=int)
    args = cli.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    if args.stop_after_steps is not None and args.stop_after_steps < 1:
        cli.error('stop-after-steps must be positive.')
    source,output,base_path = [(ROOT/cfg[key]).resolve() for key in ('data_source','output_dir','model_id')]
    if not all(path.is_relative_to(ROOT) for path in (source,output,base_path)) or source == output:
        raise ValueError('Source and separate output must remain within this project.')
    output.mkdir(parents=True,exist_ok=True)
    with exclusive_prediction_file(output/'.training.lock'):
        original = json.loads((source/'training_manifest.json').read_text())
        train = [json.loads(line) for line in (source/'training_subset.jsonl').read_text(encoding='utf-8').splitlines()]
        validation = [json.loads(line) for line in (source/'validation.jsonl').read_text(encoding='utf-8').splitlines()]
        test = [json.loads(line) for line in (source/'held_out_test.jsonl').read_text(encoding='utf-8').splitlines()]
        for rows,key,size in ((train,'train_sha256',1000),(validation,'validation_sha256',100),(test,'test_sha256',961)):
            if len(rows) != size or rows_sha256(rows) != original[key]:
                raise ValueError('Frozen study data changed or the sample size differs.')
        base_identity = {
            'hashes':{name:file_sha256(base_path/name) for name in original['base_files']['hashes']},
            'weights':[{'name':w['name'],'bytes':(base_path/w['name']).stat().st_size,'mtime_ns':(base_path/w['name']).stat().st_mtime_ns} for w in original['base_files']['weights']],
        }
        if base_identity != original['base_files']:
            raise ValueError('The shared Base files changed.')
        manifest = {'config':cfg,'fresh_base':True,'train_rows':len(train),'validation_rows':len(validation),'test_rows':len(test),
                    'train_sha256':rows_sha256(train),'validation_sha256':rows_sha256(validation),'test_sha256':rows_sha256(test),
                    'base_files':base_identity,'data_source_manifest_sha256':file_sha256(source/'training_manifest.json'),
                    'training_code_sha256':file_sha256(Path(__file__)),
                    'helper_code_sha256':{name:file_sha256(ROOT/name) for name in ('training/repair_protocol.py','training/train_tool_lora.py','training/experiment_state.py','evaluation/checkpoint.py')},
                    'label_policy':'generation-prompt-masked-through-eos-v1',
                    'versions':{name:version(name) for name in ('torch','transformers','peft','trl','bitsandbytes')}}
        signature = recipe_signature(manifest)
        path = output/'training_manifest.json'
        if path.exists():
            if json.loads(path.read_text()) != manifest:
                raise ValueError('Fresh study recipe changed; use a new output directory.')
        elif (output/'adapter_model.safetensors').exists():
            raise ValueError('Existing model lacks an immutable recipe.')
        else:
            write_json_atomic(path,manifest)
        if (output/'run_metrics.json').exists():
            metrics = json.loads((output/'run_metrics.json').read_text())
            if metrics['recipe_signature'] == signature and (output/'adapter_model.safetensors').exists():
                print('Fresh rank training already complete; preserved its adapter.')
                return 0
        tokenizer = prepare_tokenizer(base_path)
        from datasets import Dataset
        train_ds = Dataset.from_list([completion_example(row,tokenizer,cfg['max_length']) for row in train])
        validation_ds = Dataset.from_list([completion_example(row,tokenizer,cfg['max_length']) for row in validation])
        for name,rows in (('training_subset.jsonl',train),('validation.jsonl',validation),('held_out_test.jsonl',test)):
            (output/name).write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')
        (output/'training_example.txt').write_text(tokenizer.decode([v for v in train_ds[0]['labels'] if v != -100],skip_special_tokens=False),encoding='utf-8')
        print(f"Fresh r{cfg['lora_r']}: {len(train)} train / {len(validation)} validation; {cfg['max_steps']} planned optimizer updates.")
        if args.prepare_only:
            print('Frozen rows, completion masks and shared Base identity verified.')
            return 0
        import hashlib
        import signal
        import torch
        from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM,BitsAndBytesConfig,set_seed,TrainerCallback
        from trl import SFTTrainer,SFTConfig
        if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 12*2**30:
            raise RuntimeError('At least 12 GiB free CUDA memory is required.')
        if hasattr(signal,'SIGBREAK'):
            def interrupt(signum,frame):
                raise KeyboardInterrupt
            signal.signal(signal.SIGBREAK,interrupt)
        set_seed(cfg['seed'])
        torch.cuda.reset_peak_memory_stats()
        model = AutoModelForCausalLM.from_pretrained(base_path,local_files_only=True,device_map={'':0},dtype=torch.bfloat16,
            attn_implementation='sdpa',quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16))
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model,LoraConfig(r=cfg['lora_r'],lora_alpha=cfg['lora_alpha'],lora_dropout=cfg['lora_dropout'],
                                               target_modules=cfg['target_modules'],bias='none',task_type='CAUSAL_LM'))
        model.save_pretrained = partial(model.save_pretrained,save_embedding_layers=False)
        model.load_adapter = partial(model.load_adapter,autocast_adapter_dtype=False)
        model.eval()
        inputs = torch.tensor([train_ds[0]['input_ids'][:128]],device='cuda')
        with torch.inference_mode():
            active = model(input_ids=inputs).logits[:,-1].float().cpu()
            with model.disable_adapter():
                base_logits = model(input_ids=inputs).logits[:,-1].float().cpu()
        torch.testing.assert_close(active,base_logits,rtol=0,atol=0)
        initial_sha = hashlib.sha256(base_logits.numpy().tobytes()).hexdigest()
        write_json_atomic(output/'initialization.json',{'initial_base_logits_sha256':initial_sha,'fresh_base_initialization_verified':True})
        model.train()
        callback = progress_callback(output,signature,args.stop_after_steps)

        class GenerationCheck(TrainerCallback):
            def on_step_end(self,training_args,state,control,model=None,**kwargs):
                if state.global_step not in (20,40,100):
                    return
                model.eval()
                records=[]
                for index in (1,0,2,3):
                    row=validation[index]
                    record=run_generation(model,tokenizer,row,128)
                    predicted,error=parse_calls(record['output_text'])
                    expected=[{'name':c['function']['name'],'arguments':c['function']['arguments']} for c in row['messages'][-1]['tool_calls']]
                    record.update({'validation_index':index,'parse_error':error,**score_calls(predicted,expected,error)})
                    records.append(record)
                model.train()
                write_json_atomic(output/f'generation-step-{state.global_step}.json',records)
                print(f"Step {state.global_step}: generated format {sum(r['valid_tool_call'] for r in records)}/4",flush=True)

        trainer=SFTTrainer(model=model,processing_class=tokenizer,train_dataset=train_ds,eval_dataset=validation_ds,
            callbacks=[callback,GenerationCheck()],args=SFTConfig(output_dir=str(output),num_train_epochs=cfg['epochs'],max_steps=cfg['max_steps'],
                learning_rate=cfg['transformer_lr'],lr_scheduler_type='constant',weight_decay=0,
                per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
                max_length=cfg['max_length'],bf16=True,gradient_checkpointing=True,assistant_only_loss=False,eos_token='<|im_end|>',loss_type='nll',
                logging_steps=10,save_strategy='steps',save_steps=20,save_total_limit=3,eval_strategy='no',report_to='none',seed=cfg['seed'],dataloader_num_workers=0))
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data=parameter.data.float()
        trainer.optimizer=torch.optim.AdamW(parameter_groups(model,cfg['head_lr'],cfg['transformer_lr']),weight_decay=0)
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
        model.print_trainable_parameters()
        checkpoint=last_complete_checkpoint(output,signature)
        print('Resume:',checkpoint or 'none; new zero-B LoRA on fresh Base')
        try:
            result=trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
        except KeyboardInterrupt:
            callback.save(trainer.state,'interrupted')
            print('Stopped; repeat this command to resume saved optimizer/RNG state.')
            return 130
        if result.global_step < trainer.state.max_steps:
            callback.save(trainer.state,'interrupted')
            print('Saved the deliberate pilot pause. Continue without --stop-after-steps.')
            return 130
        progress=json.loads((output/'training_progress.json').read_text())
        trainer.save_model(str(output))
        tokenizer.save_pretrained(output)
        model.eval()
        with torch.inference_mode():
            expected=model(input_ids=inputs).logits[:,-1].float().cpu()
        model.load_adapter(str(output),adapter_name='reload_check',is_trainable=False)
        model.set_adapter('reload_check')
        model.eval()
        with torch.inference_mode():
            actual=model(input_ids=inputs).logits[:,-1].float().cpu()
        torch.testing.assert_close(actual,expected,rtol=0,atol=.005)
        metrics={**result.metrics,'recipe_signature':signature,'optimizer_steps':result.global_step,'training_seconds':progress['training_seconds'],
                 'peak_allocated_vram_gib':progress['peak_allocated_vram_gib'],'trainable_parameters':trainable,
                 'adapter_bytes':(output/'adapter_model.safetensors').stat().st_size,'initial_base_logits_sha256':initial_sha,
                 'fresh_base_initialization_verified':True,'adapter_reload_max_logit_difference':(actual-expected).abs().max().item()}
        write_json_atomic(output/'run_metrics.json',metrics)
        callback.save(trainer.state,'completed')
        print(json.dumps(metrics,indent=2))
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
