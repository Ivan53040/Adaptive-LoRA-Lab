# Separate expansion snapshot of evaluation/coding_benchmark.py; original pilot remains frozen.
"""Greedy raw Python completion: identical prompts and Docker tests for all adapters."""
import argparse
import ast
import json
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from coding.expansion_data import DATA,read_rows,verify_splits
from coding.log import CodingLog
from coding.sandbox import evaluate_program
from evaluation.benchmark import file_sha256
from evaluation.checkpoint import write_json_atomic
from training.experiment_state import recipe,rows_sha256

STOP_MARKERS=('\nclass ','\ndef ','\n#','\nif __name__','<|endoftext|>','<|im_end|>')

def completion_text(text):
    # Standard raw-completion boundary: preserve the target body and stop before unrelated top-level code.
    for marker in STOP_MARKERS:
        text=text.split(marker,1)[0]
    return text.rstrip()+'\n'

def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--config',type=Path,default=ROOT/'configs/coding/expansion_r8.json')
    cli.add_argument('--variant',choices=('base','tool','coding'),required=True)
    cli.add_argument('--split',choices=('validation','test'),default='validation')
    cli.add_argument('--adapter',type=Path)
    cli.add_argument('--limit',type=int,default=0)
    cli.add_argument('--output-dir',type=Path)
    args=cli.parse_args()
    verify_splits()
    cfg=json.loads(args.config.read_text()); rows=read_rows(args.split)
    if (ROOT/cfg['data_dir']).resolve()!=DATA.resolve():
        raise ValueError('This experiment requires its frozen expansion directory.')
    if args.limit<0:
        cli.error('limit must be zero or positive.')
    if args.limit:
        rows=rows[:args.limit]
    data_manifest=json.loads((DATA/'manifest.json').read_text())
    if rows_sha256(read_rows(args.split))!=data_manifest['row_sha256'][args.split]:
        raise ValueError('Frozen coding evaluation rows changed.')
    base=(ROOT/cfg['model_id']).resolve()
    adapter=args.adapter.resolve() if args.adapter else (ROOT/(cfg['tool_adapter'] if args.variant=='tool' else cfg['output_dir'])).resolve()
    folder=args.output_dir.resolve() if args.output_dir else ROOT/cfg['output_dir']/'evaluation'
    if not folder.is_relative_to(ROOT) or not adapter.is_relative_to(ROOT):
        raise ValueError('Coding output/model paths must stay within the project.')
    identity=recipe(cfg,[],[],[],ROOT/cfg['dataset_path'])
    fingerprint={'variant':args.variant,'split':args.split,'rows_sha256':rows_sha256(rows),
                 'base_files':identity['base_files'],'versions':identity['versions'],'max_new_tokens':cfg['max_new_tokens'],
                 'base_generation_config_sha256':file_sha256(base/'generation_config.json') if (base/'generation_config.json').exists() else None,
                 'seed':cfg['seed'],'generation':'raw-prefix-greedy-fp32-adapter-nf4-bf16-v1','eos_token':cfg['eos_token'],
                 'docker_image':cfg['docker_image'],'evaluation_code':file_sha256(Path(__file__)),
                 'data_manifest':file_sha256(DATA/'manifest.json'),'data_reader':file_sha256(ROOT/'coding/expansion_data.py'),
                 'baseline_evaluation_code':file_sha256(ROOT/'evaluation/coding_benchmark.py'),
                 'sandbox_code':file_sha256(ROOT/'coding/sandbox.py'),'log_code':file_sha256(ROOT/'coding/log.py')}
    if args.variant!='base':
        fingerprint['adapter']={name:file_sha256(adapter/name) for name in ('adapter_config.json','adapter_model.safetensors')}
    stem=f'{args.variant}-{args.split}-{len(rows)}-tokens{cfg["max_new_tokens"]}'
    interrupted=False; torch=None
    with CodingLog(folder/f'{stem}.jsonl',rows,fingerprint) as log:
        pending=[i for i in range(len(rows)) if i not in log.records]
        try:
            if pending:
                image_check=subprocess.run(['docker','image','inspect',cfg['docker_image'],'--format','{{.Id}}'],
                    text=True,capture_output=True,timeout=10)
                if image_check.returncode or image_check.stdout.strip()!=cfg['docker_image']:
                    raise RuntimeError('Pinned Docker sandbox image/engine is unavailable; no host execution fallback.')
                import torch
                from peft import PeftModel
                from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,set_seed,StoppingCriteria,StoppingCriteriaList
                if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<8*2**30:
                    raise RuntimeError('At least8GiB free CUDA memory required for coding evaluation.')
                if hasattr(signal,'SIGBREAK'):
                    def interrupt(*args): raise KeyboardInterrupt
                    signal.signal(signal.SIGBREAK,interrupt)
                set_seed(cfg['seed'])
                tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
                tokenizer.eos_token=cfg['eos_token']; tokenizer.pad_token=cfg['eos_token']
                model=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,device_map={'':0},dtype=torch.bfloat16,
                    attn_implementation='sdpa',quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16))
                if args.variant!='base':
                    adapter_cfg=json.loads((adapter/'adapter_config.json').read_text())
                    declared=Path(adapter_cfg['base_model_name_or_path'])
                    if not declared.is_absolute(): declared=ROOT/declared
                    if declared.resolve()!=base:
                        raise ValueError('Adapter belongs to a different Base.')
                    model=PeftModel.from_pretrained(model,adapter,is_trainable=False,local_files_only=True)
                model.eval(); torch.cuda.reset_peak_memory_stats()
                class StopAtBoundary(StoppingCriteria):
                    def __init__(self,prefix_length): self.prefix_length=prefix_length
                    def __call__(self,input_ids,scores,**kwargs):
                        text=tokenizer.decode(input_ids[0,self.prefix_length:],skip_special_tokens=False)
                        return any(marker in text for marker in STOP_MARKERS)
                warm=tokenizer(rows[pending[0]]['prompt'],return_tensors='pt',add_special_tokens=False).to('cuda')
                with torch.inference_mode():
                    model.generate(**warm,max_new_tokens=1,do_sample=False,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id)
                for index in pending:
                    row=rows[index]
                    inputs=tokenizer(row['prompt'],return_tensors='pt',add_special_tokens=False).to('cuda')
                    torch.cuda.synchronize(); start=time.monotonic()
                    with torch.inference_mode():
                        output=model.generate(**inputs,max_new_tokens=cfg['max_new_tokens'],do_sample=False,
                                              eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id,
                                              stopping_criteria=StoppingCriteriaList([StopAtBoundary(inputs['input_ids'].shape[1])]))
                    torch.cuda.synchronize(); latency=time.monotonic()-start
                    tokens=output[0,inputs['input_ids'].shape[1]:]
                    text=tokenizer.decode(tokens,skip_special_tokens=False); completion=completion_text(text)
                    program=row['prompt']+completion
                    try:
                        ast.parse(program); syntax=True
                    except (SyntaxError,ValueError):
                        syntax=False
                    result=evaluate_program(program,row['test_code'],cfg['docker_image']) if syntax else {'passed':False,'error_type':'SyntaxError'}
                    if result.get('error_type')=='ContainerError' and any(message in result.get('error','').lower()
                        for message in ('docker:','error response from daemon','error during connect','cannot connect to')):
                        raise RuntimeError('Docker infrastructure failed; this coding case was not scored or saved.')
                    record={'test_index':index,'task_id':row['task_id'],'task_sha256':rows_sha256([row]),
                            'output_text':text,'completion':completion,'generated_tokens':len(tokens),
                            'latency_seconds':latency,'hit_token_limit':len(tokens)==cfg['max_new_tokens'],
                            'syntax_valid':syntax,**result}
                    log.append(record)
                    print(f'{args.variant} {args.split} {len(log.records)}/{len(rows)}: passed={result["passed"]}, syntax={syntax}, {latency:.2f}s',flush=True)
        except KeyboardInterrupt:
            interrupted=True; print('Interrupted; completed coding cases are saved.'); return 130
        finally:
            log.save_progress(interrupted,torch.cuda.max_memory_allocated()/2**30 if torch else None)
        records=list(log.records.values()); n=len(records)
        summary={'variant':args.variant,'split':args.split,'examples':n,'rows_sha256':rows_sha256(rows),
                 'passed':sum(r['passed'] for r in records),'pass_at_1':sum(r['passed'] for r in records)/n,
                 'syntax_valid_rate':sum(r['syntax_valid'] for r in records)/n,
                 'token_limit_hit_rate':sum(r['hit_token_limit'] for r in records)/n,
                 'median_latency_seconds':statistics.median(r['latency_seconds'] for r in records),
                 'peak_allocated_vram_gib':log.peak,
                 'generation':'one greedy candidate per frozen task; raw function completion,512-token budget, stop at next top-level definition/comment',
                 'benchmark_scope':'eligible MBPP465 held-out subset' if args.split=='test' and not args.limit else 'unchanged eligible MBPP32 validation subset' if not args.limit else 'explicit expansion smoke subset'}
        write_json_atomic(folder/f'{stem}-summary.json',summary)
        print(json.dumps(summary,indent=2)); return 0

if __name__=='__main__':
    raise SystemExit(main())
