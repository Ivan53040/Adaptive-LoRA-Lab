"""Train a fresh coding adapter with raw-prefix-masked loss and checkpoint recovery."""
import argparse
from functools import partial
import json
import math
from pathlib import Path
import signal
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from coding.data import DATA,read_rows
from evaluation.benchmark import file_sha256
from evaluation.checkpoint import exclusive_prediction_file,write_json_atomic
from training.experiment_state import recipe,recipe_signature,last_complete_checkpoint,progress_callback,rows_sha256
from training.repair_protocol import parameter_groups

def encode_example(row,tokenizer,max_length):
    prompt=tokenizer.encode(row['prompt'],add_special_tokens=False)
    target=tokenizer.encode(row['completion'],add_special_tokens=False)+[tokenizer.eos_token_id]
    if len(prompt)+len(target)>max_length or not prompt or not target:
        raise ValueError('Coding example is empty or overlong; targets must not be truncated.')
    return {'input_ids':prompt+target,'attention_mask':[1]*(len(prompt)+len(target)),
            'labels':[-100]*len(prompt)+target}

def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--config',type=Path,default=ROOT/'configs/coding/mbpp_r8.json')
    cli.add_argument('--smoke',action='store_true')
    cli.add_argument('--stop-after-steps',type=int)
    cli.add_argument('--output-dir',type=Path)
    cli.add_argument('--prepare-only',action='store_true')
    args=cli.parse_args()
    if args.stop_after_steps is not None and args.stop_after_steps<1:
        cli.error('stop-after-steps must be positive.')
    cfg=json.loads(args.config.read_text())
    if (ROOT/cfg['data_dir']).resolve()!=DATA.resolve():
        raise ValueError('This pilot requires its frozen data directory.')
    train,validation,test=[read_rows(name) for name in ('train','validation','test')]
    preparation=json.loads((DATA/'manifest.json').read_text())
    for name,rows in zip(('train','validation','test'),(train,validation,test)):
        if rows_sha256(rows)!=preparation['row_sha256'][name]:
            raise ValueError('Frozen coding split changed.')
    if args.smoke:
        train=train[:16]
        cfg.update(output_dir=cfg['output_dir']+'-smoke',max_steps=4,smoke=True)
    else:
        cfg['max_steps']=math.ceil(len(train)/cfg['gradient_accumulation_steps'])*cfg['epochs']
    if args.output_dir:
        cfg['output_dir']=str(args.output_dir.resolve().relative_to(ROOT))
    output=(ROOT/cfg['output_dir']).resolve(); base=(ROOT/cfg['model_id']).resolve()
    if not output.is_relative_to(ROOT) or not base.is_relative_to(ROOT):
        raise ValueError('Coding artifacts must remain in this project.')
    output.mkdir(parents=True,exist_ok=True)
    with exclusive_prediction_file(output/'.training.lock'):
        manifest=recipe(cfg,train,validation,test,ROOT/cfg['dataset_path'])
        manifest.update(fresh_base=True,label_policy='raw-prefix-masked-endoftext-v1',
                        training_code_sha256=file_sha256(Path(__file__)),
                        preparation_manifest_sha256=file_sha256(ROOT/cfg['data_dir']/'manifest.json'),
                        helper_code_sha256={name:file_sha256(ROOT/name) for name in
                            ('coding/data.py','training/experiment_state.py','training/repair_protocol.py','evaluation/checkpoint.py')})
        signature=recipe_signature(manifest)
        path=output/'training_manifest.json'
        if path.exists() and json.loads(path.read_text())!=manifest:
            raise ValueError('Frozen coding recipe changed; use a separate output directory.')
        write_json_atomic(path,manifest)
        metrics_path=output/'run_metrics.json'
        checkpoint=last_complete_checkpoint(output,signature)
        if not metrics_path.exists() and checkpoint is not None and json.loads((checkpoint/'trainer_state.json').read_text())['global_step']==cfg['max_steps']:
            # Recover final export without an extra optimizer update.
            for name in ('adapter_config.json','adapter_model.safetensors','tokenizer.json','tokenizer_config.json','chat_template.jinja'):
                shutil.copyfile(checkpoint/name,output/name)
            progress=json.loads((output/'training_progress.json').read_text())
            from safetensors import safe_open
            with safe_open(str(output/'adapter_model.safetensors'),framework='pt',device='cpu') as state:
                count=sum(math.prod(state.get_slice(key).get_shape()) for key in state.keys())
            write_json_atomic(metrics_path,{'recipe_signature':signature,'optimizer_steps':cfg['max_steps'],
                'trainable_parameters':count,'training_seconds':progress['training_seconds'],
                'peak_allocated_vram_gib':progress['peak_allocated_vram_gib'],
                'adapter_reload_max_logit_difference':None,'finalized_from_checkpoint':str(checkpoint)})
            write_json_atomic(output/'training_progress.json',{**progress,'status':'completed'})
        if metrics_path.exists():
            metrics=json.loads(metrics_path.read_text())
            if metrics['recipe_signature']==signature and (output/'adapter_model.safetensors').exists():
                print('Coding training already complete; preserved the saved adapter.'); return 0
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
        tokenizer.eos_token=cfg['eos_token']; tokenizer.pad_token=cfg['eos_token']
        examples=[encode_example(row,tokenizer,cfg['max_length']) for row in train]
        validation_examples=[encode_example(row,tokenizer,cfg['max_length']) for row in validation]
        tokenizer.save_pretrained(output)
        for name,rows in (('training_subset',train),('validation',validation),('held_out_test',test)):
            (output/f'{name}.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')
        print(f"Coding: {len(train)} train /{len(validation)} validation /{len(test)} held-out; {cfg['max_steps']} optimizer updates.",flush=True)
        if args.prepare_only:
            return 0
        import torch
        from datasets import Dataset
        from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM,BitsAndBytesConfig,set_seed
        from trl import SFTTrainer,SFTConfig
        if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<12*2**30:
            raise RuntimeError('At least12GiB free CUDA memory is required.')
        if hasattr(signal,'SIGBREAK'):
            signal.signal(signal.SIGBREAK,lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
        set_seed(cfg['seed']); torch.cuda.reset_peak_memory_stats()
        model=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,device_map={'':0},dtype=torch.bfloat16,
            attn_implementation='sdpa',quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16))
        model=prepare_model_for_kbit_training(model)
        model=get_peft_model(model,LoraConfig(r=cfg['lora_r'],lora_alpha=cfg['lora_alpha'],lora_dropout=cfg['lora_dropout'],
                    target_modules=cfg['target_modules'],bias='none',task_type='CAUSAL_LM'))
        model.save_pretrained=partial(model.save_pretrained,save_embedding_layers=False)
        model.load_adapter=partial(model.load_adapter,autocast_adapter_dtype=False)
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data=parameter.data.float()
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
        model.eval()
        inputs=torch.tensor([examples[0]['input_ids'][:128]],device='cuda')
        with torch.inference_mode():
            active=model(input_ids=inputs).logits[:,-1].float().cpu()
            with model.disable_adapter():
                untouched=model(input_ids=inputs).logits[:,-1].float().cpu()
        torch.testing.assert_close(active,untouched,rtol=0,atol=0)
        write_json_atomic(output/'initialization.json',{'fresh_zero_b_matches_base':True})
        model.train()
        callback=progress_callback(output,signature,args.stop_after_steps)
        trainer=SFTTrainer(model=model,processing_class=tokenizer,train_dataset=Dataset.from_list(examples),
            eval_dataset=Dataset.from_list(validation_examples),callbacks=[callback],
            args=SFTConfig(output_dir=str(output),max_steps=cfg['max_steps'],num_train_epochs=cfg['epochs'],
                per_device_train_batch_size=1,gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
                learning_rate=cfg['transformer_lr'],lr_scheduler_type='constant',weight_decay=0,
                max_length=cfg['max_length'],bf16=True,gradient_checkpointing=True,assistant_only_loss=False,
                eos_token=cfg['eos_token'],loss_type='nll',logging_steps=10,save_strategy='steps',save_steps=20,
                save_total_limit=3,eval_strategy='no',report_to='none',seed=cfg['seed'],dataloader_num_workers=0))
        # SFTTrainer initialization can recast a PEFT model to BF16. Keep trainable
        # LoRA tensors FP32 after that initialization, before constructing optimizer state.
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data=parameter.data.float()
        assert all(parameter.dtype==torch.float32 for parameter in model.parameters() if parameter.requires_grad)
        trainer.optimizer=torch.optim.AdamW(parameter_groups(model,cfg['head_lr'],cfg['transformer_lr']),weight_decay=0)
        checkpoint=last_complete_checkpoint(output,signature)
        print('Resume:',checkpoint or 'fresh Base, new zero-B coding LoRA',flush=True)
        try:
            result=trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
        except KeyboardInterrupt:
            callback.save(trainer.state,'interrupted'); print('Interrupted; repeat to resume complete optimizer/RNG checkpoints.'); return 130
        if result.global_step<trainer.state.max_steps:
            callback.save(trainer.state,'interrupted'); print('Deliberate pause saved.'); return 130
        assert all(parameter.dtype==torch.float32 for parameter in model.parameters() if parameter.requires_grad)
        trainer.save_model(str(output)); tokenizer.save_pretrained(output)
        model.eval()
        with torch.inference_mode():
            expected=model(input_ids=inputs).logits[:,-1].float().cpu()
        model.load_adapter(str(output),adapter_name='reload_check',is_trainable=False); model.set_adapter('reload_check'); model.eval()
        with torch.inference_mode():
            actual=model(input_ids=inputs).logits[:,-1].float().cpu()
        torch.testing.assert_close(actual,expected,rtol=0,atol=.005)
        progress=json.loads((output/'training_progress.json').read_text())
        metrics={**result.metrics,'optimizer_steps':result.global_step,'recipe_signature':signature,
                 'training_seconds':progress['training_seconds'],'peak_allocated_vram_gib':progress['peak_allocated_vram_gib'],
                 'adapter_reload_max_logit_difference':(actual-expected).abs().max().item(),
                 'trainable_parameters':trainable_parameters}
        write_json_atomic(metrics_path,metrics); callback.save(trainer.state,'completed')
        print(json.dumps(metrics,indent=2)); return 0

if __name__=='__main__':
    raise SystemExit(main())
