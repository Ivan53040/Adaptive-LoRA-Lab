"""Local browser demo: a persistent GPU model and isolated Base/LoRA mock worlds."""

import argparse
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.benchmark import DEFAULT_ADAPTER, load_evaluation_model, parse_calls, run_generation
from runtime.mock_tools import MockWorld, TOOL_SCHEMAS
from runtime.coding_engine import validate_coding_request,run_coding_generation
from runtime.router import route_request,VERSION as ROUTER_VERSION
from runtime.general_engine import run_general_generation

BRISBANE = timezone(timedelta(hours=10))
DEFAULT_DEMO_ADAPTER = ROOT / 'outputs/data-size-r8/3k'
CODING_ADAPTER = ROOT / 'outputs/coding-mbpp-r8-fp32'


class Demo:
    def __init__(self, adapter):
        self.adapter = adapter
        self.lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.worlds = {variant: MockWorld() for variant in ("base", "lora")}
        self.status = "loading"
        self.error = None
        self.history = []
        self.coding_history = []
        self.coding_ready = False
        self.coding_error = None
        self.base_load_count = 0
        coding_folder = CODING_ADAPTER / 'evaluation'
        review_path = coding_folder / 'test-case-review.json'
        self.coding_review = json.loads(review_path.read_text(encoding='utf-8')) if review_path.exists() else {}
        self.coding_metrics = self.coding_review.get('summaries',{})
        self.coding_config = json.loads((ROOT/'configs/coding/mbpp_r8.json').read_text(encoding='utf-8'))
        self.folder = adapter / "evaluation"
        self.adapter_rank = json.loads((adapter / 'adapter_config.json').read_text(encoding='utf-8'))['r']
        manifest_path = adapter / 'training_manifest.json'
        self.training_rows = json.loads(manifest_path.read_text(encoding='utf-8')).get('train_rows') if manifest_path.exists() else None
        summary = self.read_json('test-summary.json',None)
        if summary is not None:
            self.benchmark = {'results':{'base':summary.get('base_metrics',{}),'lora':summary['metrics']}}
            audit = self.read_json('test-error-audit.json',{})
            counts = audit.get('new_counts',{})
            self.mock_summary = {'lora':{'examples':summary['examples'],
                'strict_mock_execution_success':counts.get('mock_execution_ok',round(summary['metrics']['mock_execution_success_rate']*summary['examples']))}}
            categories = {'datetime_difference':'datetime difference',
                          'only_case_spacing_punctuation':'only case/spacing/punctuation'}
            self.review = [{**row,'category':categories.get(row['category'],row['category']),
                            'mock_execution':{'ok':row['mock_execution_ok'],'errors':row['mock_errors']}}
                           for row in audit.get('errors',[])]
        else:
            self.benchmark = self.read_json("benchmark-961-seed42-tokens256-summary.json", {})
            self.mock_summary = self.read_json("benchmark-961-mock-execution-summary.json", {})
            errors = self.read_json("benchmark-961-seed42-tokens256-error-analysis.json", {}).get("errors", [])
            review = self.read_json("benchmark-961-mock-execution-review.json", [])
            mock_by_index = {row["test_index"]: row["mock_execution"] for row in review}
            self.review = [{**row, "mock_execution": mock_by_index.get(row["test_index"])} for row in errors]
        self.review.sort(key=lambda row: (bool(row["mock_execution"] and row["mock_execution"]["ok"]), row["test_index"]))

    def read_json(self, name, fallback):
        path = self.folder / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else fallback

    def load(self):
        try:
            self.model, self.tokenizer, self.torch = load_evaluation_model(self.adapter)
            self.base_load_count += 1
            if (CODING_ADAPTER/'adapter_model.safetensors').exists():
                try:
                    from transformers import AutoTokenizer
                    tool_config=json.loads((self.adapter/'adapter_config.json').read_text())
                    coding_config=json.loads((CODING_ADAPTER/'adapter_config.json').read_text())
                    def base_path(config):
                        path=Path(config['base_model_name_or_path'])
                        return (path if path.is_absolute() else ROOT/path).resolve()
                    if base_path(tool_config)!=base_path(coding_config):
                        raise ValueError('Coding and Tool adapters belong to different Base models.')
                    self.coding_tokenizer=AutoTokenizer.from_pretrained(CODING_ADAPTER,local_files_only=True)
                    if self.coding_tokenizer.get_vocab()!=self.tokenizer.get_vocab():
                        raise ValueError('Coding vocabulary differs from the shared Base.')
                    self.model.load_adapter(CODING_ADAPTER,adapter_name='coding',is_trainable=False,local_files_only=True)
                    self.coding_ready=True
                except Exception as error:
                    self.coding_error=str(error)
                    traceback.print_exc()
            self.model.set_adapter('default',inference_mode=True)
            self.model.eval()
            self.status = "ready"
            print(f"Model ready. One frozen NF4 Base; Tool adapter and Coding available={self.coding_ready}.", flush=True)
        except Exception as error:
            self.status, self.error = "error", str(error)
            traceback.print_exc()

    def snapshot(self):
        with self.state_lock:
            return {
                "status": self.status, "error": self.error,
                "adapter": self.adapter.name, "adapter_rank":self.adapter_rank, "training_rows":self.training_rows,
                "adapter_path":str(self.adapter.relative_to(ROOT)) if self.adapter.is_relative_to(ROOT) else str(self.adapter),
                "model": "Qwen3-8B-Base", "review_count":len(self.review),
                "shared_base_load_count":self.base_load_count,
                "active_adapter":getattr(getattr(self,'model',None),'active_adapter',None),
                "inference_parameters_frozen":not any(p.requires_grad for p in self.model.parameters()) if hasattr(getattr(self,'model',None),'parameters') else None,
                "now": datetime.now(BRISBANE).strftime("%Y-%m-%dT%H:%M:%S"),
                "worlds": {v: w.snapshot() for v, w in self.worlds.items()},
                "benchmark": self.benchmark.get("results", {}), "mock_summary": self.mock_summary,
                "history": self.history[-20:],
                "coding": {"ready":self.coding_ready,"error":self.coding_error,
                    "adapter_path":str(CODING_ADAPTER.relative_to(ROOT)),"experimental":True,
                    "metrics":self.coding_metrics,"review_count":self.coding_review.get('coding_failures',0),
                    "history":self.coding_history[-10:]},
                "router":{"version":ROUTER_VERSION,"coding_default":"base","coding_opt_in":True},
            }

    def reset(self):
        if not self.lock.acquire(blocking=False):
            raise ValueError("A request is running; wait before resetting.")
        try:
            with self.state_lock:
                self.worlds = {v: MockWorld() for v in ("base", "lora")}
                self.history = []
        finally:
            self.lock.release()

    def generate(self, request):
        if self.status == 'paused':
            raise ValueError('Model inference is paused to free GPU memory for training.')
        if self.status != "ready":
            raise ValueError("The model is still loading or failed to load. Check the status.")
        prompt = request.get("prompt")
        mode = request.get("mode", "lora")
        tokens = request.get("max_new_tokens", 256)
        isolated=request.get('isolated',False); record_history=request.get('record_history',True)
        if type(isolated) is not bool or type(record_history) is not bool:
            raise ValueError('Invalid request isolation/history settings.')
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4000:
            raise ValueError("Enter a request containing 1–4,000 characters.")
        if mode not in {"base", "lora", "both"}:
            raise ValueError("Choose Base, LoRA or Both.")
        if type(tokens) is not int or not 32 <= tokens <= 512:
            raise ValueError("Output limit must be between 32 and 512 tokens.")
        reference = request.get("reference_time")
        now = datetime.fromisoformat(reference) if reference else datetime.now(BRISBANE)
        date = now.strftime("%Y-%m-%dT%H:%M:%S")
        row = {
            "tools": TOOL_SCHEMAS,
            "messages": [
                {"role": "system", "content": f"Current date and time given in YYYY-MM-DDTHH:MM:SS format: {date}\nDay of week is {now.strftime('%A')}\nYou are a model that can do function calling with the following functions\n"},
                {"role": "user", "content": prompt.strip()}, {"role": "assistant", "content": ""},
            ],
        }
        if not self.lock.acquire(blocking=False):
            raise ValueError("Another request is running. Wait for it to finish.")
        try:
            responses = []
            self.model.set_adapter('default',inference_mode=True)
            for variant in ("base", "lora") if mode == "both" else (mode,):
                with self.model.disable_adapter() if variant == "base" else nullcontext():
                    result = run_generation(self.model, self.tokenizer, row, tokens)
                calls, error = parse_calls(result["output_text"])
                with self.state_lock:
                    world = MockWorld(row['tools']) if isolated else self.worlds[variant]
                    native_errors = world.validate(calls, strict=False) if error is None else [
                        {"code": "parse_error", "message": error}]
                    execution = world.execute(calls) if error is None else {
                        "ok": False, "mode": "mock", "errors": [{"code": "parse_error", "message": error}],
                        "results": [], "state": world.snapshot()}
                    response = {
                        **result, "variant": variant, "prompt": prompt.strip(),
                        "calls": calls, "format_valid": error is None,
                        "native_schema_valid": not native_errors, "native_schema_errors": native_errors,
                        "execution": execution,
                    }
                    if record_history: self.history.append(response)
                responses.append(response)
                print(f"{variant}: {result['generated_tokens']} tokens, {result['latency_seconds']:.2f}s, mock execution={execution['ok']}", flush=True)
            return {"responses": responses, "reference_time": date, "state": self.snapshot()}
        finally:
            self.lock.release()

    def generate_coding(self, request):
        values=validate_coding_request(request)
        record_history=request.get('record_history',True)
        if type(record_history) is not bool: raise ValueError('Invalid history setting.')
        if self.status!='ready' or not self.coding_ready:
            raise ValueError(self.coding_error or 'The shared model and Coding adapter are not ready.')
        if not self.lock.acquire(blocking=False):
            raise ValueError('Another request is running. Wait for it to finish.')
        try:
            responses=[]
            for variant in ('base','coding') if values['mode']=='both' else (values['mode'],):
                start=time.perf_counter()
                self.model.set_adapter('coding',inference_mode=True)
                switch_ms=(time.perf_counter()-start)*1000
                self.model.eval()
                with self.model.disable_adapter() if variant=='base' else nullcontext():
                    result=run_coding_generation(self.model,self.coding_tokenizer,self.torch,values['prefix'],
                        values['max_new_tokens'],values['tests'] if values['run_tests'] else None,
                        self.coding_config['docker_image'])
                response={**result,'variant':variant,'adapter_switch_ms':switch_ms,'experimental':variant=='coding'}
                responses.append(response)
                with self.state_lock:
                    if record_history: self.coding_history.append(response)
                print(f"coding/{variant}: {result['generated_tokens']} tokens, {result['latency_seconds']:.2f}s, switch={switch_ms:.2f}ms",flush=True)
        finally:
            try:
                self.model.set_adapter('default',inference_mode=True)
                self.model.eval()
                final_state=self.snapshot()
            finally:
                self.lock.release()
        return {'responses':responses,'prefix':values['prefix'],'state':final_state}

    def generate_general(self,request):
        prompt=request.get('prompt'); tokens=request.get('max_new_tokens',64)
        if not isinstance(prompt,str) or not prompt.strip() or len(prompt)>4000:
            raise ValueError('Enter a request containing1–4000 characters.')
        if type(tokens) is not int or not 16<=tokens<=256: raise ValueError('General output limit must be16–256 tokens.')
        if self.status!='ready': raise ValueError('The shared Base is not ready.')
        if not self.lock.acquire(blocking=False): raise ValueError('Another request is running. Wait for it to finish.')
        try:
            self.model.set_adapter('default',inference_mode=True)
            with self.model.disable_adapter():
                result=run_general_generation(self.model,self.tokenizer,self.torch,prompt,tokens)
        finally:
            try: self.model.set_adapter('default',inference_mode=True); self.model.eval()
            finally: self.lock.release()
        return {'response':result,'state':self.snapshot()}

    def generate_routed(self,request):
        prompt=request.get('prompt')
        route=route_request(prompt,request.get('allow_experimental_coding',False))
        generate=request.get('generate',True)
        if type(generate) is not bool: raise ValueError('Generate must be boolean.')
        if route['domain']=='clarify' or not generate:
            return {'route':route,'kind':'clarify' if route['domain']=='clarify' else 'preview','state':self.snapshot()}
        start=time.perf_counter()
        options={key:request[key] for key in ('record_history','isolated','reference_time') if key in request}
        if request.get('run_tests',False) and route['domain']!='python':
            raise ValueError('Python assertions require a valid function prefix.')
        if route['domain']=='tools':
            payload=self.generate({**options,'prompt':prompt,'mode':'lora','max_new_tokens':request.get('max_new_tokens',256)})
            kind='tools'
        elif route['domain']=='python':
            payload=self.generate_coding({**options,'prefix':prompt,'mode':route['adapter'],
                'max_new_tokens':request.get('max_new_tokens',512),'run_tests':request.get('run_tests',False),
                'tests':request.get('tests','')})
            kind='python'
        else:
            payload=self.generate_general({'prompt':prompt,'max_new_tokens':request.get('max_new_tokens',64)})
            kind='general'
        return {**payload,'route':route,'kind':kind,'end_to_end_seconds':time.perf_counter()-start}


class LocalServer(ThreadingHTTPServer):
    allow_reuse_address = False


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def json_response(self, status, value):
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/api/status":
            self.json_response(200, self.server.app.snapshot())
        elif self.path == "/api/review":
            self.json_response(200, self.server.app.review)
        elif self.path == "/api/tools":
            self.json_response(200, TOOL_SCHEMAS)
        elif self.path == "/api/coding-review":
            self.json_response(200, self.server.app.coding_review.get('cases',[]))
        elif self.path in {"/", "/index.html", "/app.js", "/i18n.js", "/pipeline.js", "/coding.js", "/style.css"}:
            name = "index.html" if self.path == "/" else self.path[1:]
            payload = (ROOT / "demo" / name).read_bytes()
            mime = {"html": "text/html", "js": "text/javascript", "css": "text/css"}[name.rsplit(".", 1)[1]]
            self.send_response(200)
            self.send_header("Content-Type", mime + "; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.json_response(404, {"error": "Not found."})

    def do_POST(self):
        try:
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                raise ValueError("Use application/json.")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 32768:
                raise ValueError("Invalid request size.")
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict):
                raise ValueError("The request must be a JSON object.")
            if self.path == "/api/run":
                self.json_response(200, self.server.app.generate(request))
            elif self.path == "/api/code":
                self.json_response(200, self.server.app.generate_coding(request))
            elif self.path == "/api/route":
                self.json_response(200, self.server.app.generate_routed(request))
            elif self.path == "/api/general":
                self.json_response(200, self.server.app.generate_general(request))
            elif self.path == "/api/reset":
                self.server.app.reset()
                self.json_response(200, self.server.app.snapshot())
            else:
                self.json_response(404, {"error": "Not found."})
        except (ValueError, TypeError) as error:
            self.json_response(400, {"error": str(error)})
        except Exception:
            traceback.print_exc()
            self.json_response(500, {"error": "Model request failed; see the terminal for details."})


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_DEMO_ADAPTER)
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument('--no-model',action='store_true',help='Serve reports and pipeline without using GPU memory.')
    parser.add_argument('--runner-log',type=Path,help='Write stdout/stderr for independent background startup.')
    args = parser.parse_args()
    if args.runner_log:
        log_path = args.runner_log.resolve()
        if not log_path.is_relative_to(ROOT):
            parser.error('Demo log must remain within this project.')
        log_path.parent.mkdir(parents=True,exist_ok=True)
        sys.stdout = sys.stderr = log_path.open('a',encoding='utf-8',buffering=1)
    app = Demo(args.adapter.resolve())
    server = LocalServer(("127.0.0.1", args.port), Handler)
    server.app = app
    if args.no_model:
        app.status = 'paused'
    else:
        threading.Thread(target=app.load, daemon=True).start()
    print(f"Demo: http://127.0.0.1:{args.port} — mock tools only.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
