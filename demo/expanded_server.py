"""Extended demo with an explicit 3k Coding choice; frozen legacy server is retained."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import threading
import time
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from demo import server as legacy
from runtime.coding_engine import validate_coding_request
from runtime.router import route_request

EXPANDED_ADAPTER=ROOT/'outputs/coding-expansion-3k/r8'

def coding_choice(request):
    choice=request.get('coding_variant','pilot')
    if choice not in ('pilot','expanded') or not isinstance(choice,str):
        raise ValueError('Coding variant must be pilot or expanded.')
    return choice

class ExpandedDemo(legacy.Demo):
    def __init__(self,adapter):
        super().__init__(adapter)
        self.expanded_ready=False;self.expanded_error=None;self.expanded_loading=False
        self.expanded_config=json.loads((ROOT/'configs/coding/expansion_r8.json').read_text())
        review_path=ROOT/'outputs/coding-expansion-3k/test-case-review.json'
        self.expanded_review=json.loads(review_path.read_text(encoding='utf-8'))

    def load(self):
        self.expanded_loading=True
        # The same lock also prevents generation during the legacy loader's ready transition.
        with self.lock:
            try:
                super().load()
                if self.status!='ready':return
                from transformers import AutoTokenizer
                original=json.loads((self.adapter/'adapter_config.json').read_text())
                expanded=json.loads((EXPANDED_ADAPTER/'adapter_config.json').read_text())
                def base_path(config):
                    path=Path(config['base_model_name_or_path'])
                    return (path if path.is_absolute() else ROOT/path).resolve()
                if base_path(original)!=base_path(expanded):raise ValueError('Expanded adapter belongs to a different Base.')
                self.expanded_tokenizer=AutoTokenizer.from_pretrained(EXPANDED_ADAPTER,local_files_only=True)
                if self.expanded_tokenizer.get_vocab()!=self.tokenizer.get_vocab():raise ValueError('Expanded vocabulary differs from the shared Base.')
                self.model.load_adapter(EXPANDED_ADAPTER,adapter_name='coding_expanded',is_trainable=False,local_files_only=True)
                self.expanded_ready=True
                print('Expanded Coding ready; Tool, original Coding and 3k Coding share one Base.',flush=True)
            except Exception as error:
                self.expanded_error=str(error);traceback.print_exc()
            finally:
                if hasattr(self,'model'):
                    self.model.set_adapter('default',inference_mode=True);self.model.eval()
                self.expanded_loading=False

    def snapshot(self):
        value=super().snapshot()
        if self.expanded_loading and value['status']=='ready':value['status']='loading'
        value['coding']['expanded']={'ready':self.expanded_ready,'error':self.expanded_error,
            'adapter_path':str(EXPANDED_ADAPTER.relative_to(ROOT)),'experimental':True,
            'metrics':{name:{'validation':self.expanded_review['validation'][name],
                            'test':self.expanded_review['summaries'][name]} for name in ('base','pilot','expanded')}}
        value['router']['runtime_profile']='expanded-choice-v1'
        return value

    def generate_coding(self,request):
        choice=coding_choice(request);values=validate_coding_request(request)
        record_history=request.get('record_history',True)
        if type(record_history) is not bool:raise ValueError('Invalid history setting.')
        ready=self.coding_ready if choice=='pilot' else self.expanded_ready
        if self.status!='ready' or self.expanded_loading or (values['mode']!='base' and not ready):
            raise ValueError('The shared model or selected Coding adapter is not ready.')
        if not self.lock.acquire(blocking=False):raise ValueError('Another request is running. Wait for it to finish.')
        try:
            responses=[]
            for variant in ('base','coding') if values['mode']=='both' else (values['mode'],):
                name='default' if variant=='base' else 'coding' if choice=='pilot' else 'coding_expanded'
                start=time.perf_counter();self.model.set_adapter(name,inference_mode=True);self.model.eval()
                switch_ms=(time.perf_counter()-start)*1000
                tokenizer=self.tokenizer if variant=='base' else self.coding_tokenizer if choice=='pilot' else self.expanded_tokenizer
                config=self.coding_config if choice=='pilot' else self.expanded_config
                with self.model.disable_adapter() if variant=='base' else nullcontext():
                    result=legacy.run_coding_generation(self.model,tokenizer,self.torch,values['prefix'],
                        values['max_new_tokens'],values['tests'] if values['run_tests'] else None,config['docker_image'])
                response={**result,'variant':variant,'coding_variant':choice if variant=='coding' else None,
                    'selected_adapter':name,'adapter_switch_ms':switch_ms,'experimental':variant=='coding'}
                responses.append(response)
                with self.state_lock:
                    if record_history:self.coding_history.append(response)
                print(f'coding/{name}: {result["generated_tokens"]} tokens, {result["latency_seconds"]:.2f}s',flush=True)
        finally:
            try:
                self.model.set_adapter('default',inference_mode=True);self.model.eval();final_state=self.snapshot()
            finally:self.lock.release()
        return {'responses':responses,'prefix':values['prefix'],'coding_variant':choice,'state':final_state}

    def generate_routed(self,request):
        choice=coding_choice(request)
        route=route_request(request.get('prompt'),request.get('allow_experimental_coding',False))
        route['coding_variant']=choice if route['adapter']=='coding' else None
        generate=request.get('generate',True)
        if type(generate) is not bool:raise ValueError('Generate must be boolean.')
        if route['domain']=='clarify' or not generate:
            return {'route':route,'kind':'clarify' if route['domain']=='clarify' else 'preview','state':self.snapshot()}
        if request.get('run_tests',False) and route['domain']!='python':
            raise ValueError('Python assertions require a valid function prefix.')
        start=time.perf_counter()
        options={key:request[key] for key in ('record_history','isolated','reference_time') if key in request}
        if route['domain']=='python':
            payload=self.generate_coding({**options,'prefix':request['prompt'],'mode':route['adapter'],'coding_variant':choice,
                'max_new_tokens':request.get('max_new_tokens',512),'run_tests':request.get('run_tests',False),'tests':request.get('tests','')})
        elif route['domain']=='tools':
            payload=self.generate({**options,'prompt':request['prompt'],'mode':'lora','max_new_tokens':request.get('max_new_tokens',256)})
        else:payload=self.generate_general({'prompt':request['prompt'],'max_new_tokens':request.get('max_new_tokens',64)})
        return {**payload,'route':route,'kind':route['domain'],'end_to_end_seconds':time.perf_counter()-start}

class ExpandedHandler(legacy.Handler):
    def do_GET(self):
        if self.path=='/api/coding-expansion-review':self.json_response(200,self.server.app.expanded_review)
        else:super().do_GET()

def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--port',type=int,default=18765);cli.add_argument('--no-model',action='store_true');args=cli.parse_args()
    app=ExpandedDemo(legacy.DEFAULT_DEMO_ADAPTER)
    http=legacy.LocalServer(('127.0.0.1',args.port),ExpandedHandler);http.app=app
    if args.no_model:app.status='paused'
    else:threading.Thread(target=app.load,daemon=True).start()
    try:http.serve_forever()
    except KeyboardInterrupt:pass
    finally:http.server_close()

if __name__=='__main__':main()
