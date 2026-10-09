"""Resumable routed versus oracle and always-Base integration checks on isolated cases."""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from runtime.router import route_request,VERSION
from evaluation.benchmark import file_sha256,score_calls
from evaluation.checkpoint import exclusive_prediction_file,write_json_atomic

DATA=ROOT/'data/evaluation/router-v1.json'
OUT=ROOT/'outputs/router-v1'

def api(path,body=None):
    request=urllib.request.Request('http://127.0.0.1:18765'+path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={'Content-Type':'application/json'} if body is not None else {})
    with urllib.request.urlopen(request,timeout=120) as response: return json.load(response)

def score(case,payload):
    if case['domain']=='clarify': return payload.get('kind')=='clarify'
    if case['domain']=='tools':
        record=payload['responses'][0]
        return score_calls(record['calls'],case['calls'],None if record['format_valid'] else 'invalid')['full_call_exact'] and record['execution']['ok']
    if case['domain']=='python':
        record=payload['responses'][0]
        if record.get('checks_status')=='unavailable': raise RuntimeError('Docker checks unavailable; case was not scored.')
        return record['syntax_valid'] and record['execution']['passed']
    answer=payload['response']['answer'].splitlines()[0] if payload['response']['answer'] else ''
    return re.sub(r'[^a-z0-9]','',answer.lower())==case['answer']

def execute(case,mode,reference):
    common={'max_new_tokens':256 if case['domain'] in ('tools','python') else 64,
            'isolated':True,'record_history':False,'reference_time':reference}
    if mode=='auto':
        return api('/api/route',{**common,'prompt':case['prompt'],'run_tests':case['domain']=='python','tests':case.get('tests','')})
    if case['domain']=='clarify': return {'kind':'clarify'}
    if case['domain']=='tools': return api('/api/run',{**common,'prompt':case['prompt'],'mode':'base' if mode=='always_base' else 'lora'})
    if case['domain']=='python': return api('/api/code',{**common,'prefix':case['prompt'],'mode':'base','run_tests':True,'tests':case['tests']})
    return api('/api/general',{'prompt':case['prompt'],'max_new_tokens':64})

def report(records,routes,data,status):
    modes={}
    for mode in ('auto','oracle','always_base'):
        values=[r for r in records if r['mode']==mode]
        modes[mode]={'completed':len(values),'passed':sum(r['passed'] for r in values),
                     'pass_rate':sum(r['passed'] for r in values)/len(values) if values else None,
                     'median_end_to_end_seconds':statistics.median(r['seconds'] for r in values) if values else None}
    correct=sum(r['domain']==c['domain'] and r['adapter']==c['adapter'] for r,c in zip(routes,data['cases']))
    value={'status':status,'scope':data['scope'],'router_version':VERSION,'routing_correct':correct,'routing_cases':len(routes),
           'results':modes,'cases':records,'routing_decisions':routes}
    write_json_atomic(OUT/'comparison.json',value)
    lines=['# Router v1: isolated integration pilot','',f'Status: {status}.','',data['scope'],
           f'Routing contract matches: {correct}/{len(routes)}. Python defaults to Base; Coding requires explicit experimental opt-in.',
           'Oracle uses the predetermined domain policy, not per-case test outcomes. Always-Base retains the same domain prompt builders.',
           'Clarification cases require no model generation. Generation cases use fresh mock worlds and isolated Docker checks; demo history is not recorded.','',
           '| Mode | Completed | Task checks passed | Median total seconds |','|---|---:|---:|---:|']
    for mode,metrics in modes.items():
        lines.append(f"| {mode} | {metrics['completed']} | {metrics['passed']} | {metrics['median_end_to_end_seconds']:.2f} |" if metrics['completed'] else f'| {mode} | pending | pending | pending |')
    lines+=['','General answers use strict normalized first-line matching on three simple questions, not a broad QA score.',
            'Timing includes HTTP, generation and optional Docker checks, unlike earlier generation-only latency.',
            'This authored finite pilot checks routing/integration behavior, not general routing intelligence or Coding specialist superiority.',
            'All original HumanEval and Tool-use official benchmarks remain unchanged.','']
    (OUT/'comparison.md').write_text('\n'.join(lines),encoding='utf-8')
    return value

def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--status',action='store_true');args=cli.parse_args()
    if args.status:
        path=OUT/'progress.json'; print(path.read_text() if path.exists() else '{"status":"not_started"}');return
    data=json.loads(DATA.read_text()); OUT.mkdir(exist_ok=True)
    fingerprint={'data':file_sha256(DATA),'router':file_sha256(ROOT/'runtime/router.py'),
                 'runtime':file_sha256(ROOT/'demo/server.py'),'general':file_sha256(ROOT/'runtime/general_engine.py'),
                 'code':file_sha256(ROOT/'runtime/coding_engine.py'),'evaluator':file_sha256(Path(__file__)),
                 'tool_weights':file_sha256(ROOT/'outputs/data-size-r8/3k/adapter_model.safetensors'),
                 'coding_weights':file_sha256(ROOT/'outputs/coding-mbpp-r8-fp32/adapter_model.safetensors')}
    records=[]; path=OUT/'predictions.jsonl';meta=OUT/'metadata.json'
    routes=[route_request(case['prompt']) for case in data['cases']]
    generated=[case for case in data['cases'] if not case.get('classification_only')]
    with exclusive_prediction_file(path) as stream:
        if meta.exists() and json.loads(meta.read_text())!=fingerprint: raise ValueError('Frozen router pilot changed; use a separate output directory.')
        raw=stream.read();offset=0
        for index,line in enumerate(raw.splitlines(keepends=True)):
            try: record=json.loads(line)
            except ValueError:
                if index!=len(raw.splitlines())-1:raise
                stream.seek(offset);stream.truncate();break
            records.append(record);offset+=len(line)
        completed={(r['mode'],r['id']) for r in records}
        if len(completed)!=len(records):raise ValueError('Duplicate router result.')
        write_json_atomic(meta,fingerprint); stream.seek(0,os.SEEK_END)
        if stream.tell() and not raw[:stream.tell()].endswith(b'\n'):stream.write(b'\n')
        before=api('/api/status'); interrupted=False
        try:
            for mode in ('auto','oracle','always_base'):
                for case in generated:
                    if (mode,case['id']) in completed:continue
                    start=time.monotonic();payload=execute(case,mode,data['reference_time']);elapsed=time.monotonic()-start
                    route_ok=mode!='auto' or (payload['route']['domain']==case['domain'] and payload['route']['adapter']==case['adapter'])
                    record={'mode':mode,'id':case['id'],'domain':case['domain'],'passed':bool(route_ok and score(case,payload)),
                            'seconds':elapsed,'route':payload.get('route'),'output':payload.get('responses',payload.get('response'))}
                    stream.write((json.dumps(record,ensure_ascii=False)+'\n').encode());stream.flush();os.fsync(stream.fileno())
                    records.append(record);completed.add((mode,case['id']))
                    write_json_atomic(OUT/'progress.json',{'status':'running','mode':mode,'case':case['id'],'completed':len(records),'total':len(generated)*3})
                    report(records,routes,data,'running'); print(f'{mode}: {case["id"]} passed={record["passed"]} [{len(records)}/{len(generated)*3}]',flush=True)
        except KeyboardInterrupt: interrupted=True;raise
        finally:
            after=api('/api/status')
            if before['worlds']!=after['worlds'] or before['history']!=after['history'] or before['coding']['history']!=after['coding']['history']:
                raise RuntimeError('Isolated routing evaluation changed demo worlds/history.')
            if after['shared_base_load_count']!=before['shared_base_load_count'] or not after['inference_parameters_frozen']:
                raise RuntimeError('Shared inference runtime changed during evaluation.')
            write_json_atomic(OUT/'progress.json',{'status':'interrupted' if interrupted else 'completed' if len(records)==len(generated)*3 else 'failed',
                'completed':len(records),'total':len(generated)*3,'shared_base_load_count':after['shared_base_load_count'],'worlds_and_history_preserved':True})
        report(records,routes,data,'completed')

if __name__=='__main__': main()
