"""Raw Python completion using the already-loaded shared Base and selected adapter."""
import ast
import subprocess
import time

from coding.sandbox import evaluate_program
from evaluation.coding_benchmark import STOP_MARKERS,completion_text

def checks_status(execution):
    if execution is None: return 'not_requested'
    if execution.get('error_type')=='SandboxUnavailable': return 'unavailable'
    if execution.get('error_type')=='ContainerError' and any(message in execution.get('error','').lower()
        for message in ('docker api','error response from daemon','error during connect','cannot connect to')):
        return 'unavailable'
    return 'passed' if execution['passed'] else 'failed'

def validate_coding_request(request):
    prefix=request.get('prefix')
    mode=request.get('mode','coding')
    tokens=request.get('max_new_tokens',512)
    run_tests=request.get('run_tests',False)
    tests=request.get('tests','')
    if not isinstance(prefix,str) or not prefix.strip() or len(prefix)>4000:
        raise ValueError('Enter a Python function prefix containing1–4000 characters.')
    if mode not in ('base','coding','both'):
        raise ValueError('Choose Base, Coding or Both.')
    if type(tokens) is not int or not 32<=tokens<=512:
        raise ValueError('Output limit must be between32 and512 tokens.')
    if type(run_tests) is not bool or not isinstance(tests,str) or len(tests)>4000:
        raise ValueError('Invalid coding test settings.')
    if not prefix.endswith('\n'): prefix+='\n'
    try:
        tree=ast.parse(prefix+'    pass\n')
    except SyntaxError as error:
        raise ValueError('Finish the function signature and docstring before generating.') from error
    functions=[item for item in tree.body if isinstance(item,ast.FunctionDef)]
    if len(functions)!=1 or tree.body[-1] is not functions[0] or any(not isinstance(item,(ast.FunctionDef,ast.Import,ast.ImportFrom)) for item in tree.body):
        raise ValueError('This pilot completes one Python function, with optional imports.')
    if run_tests:
        try: test_tree=ast.parse(tests)
        except SyntaxError as error: raise ValueError('The supplied assertions have invalid Python syntax.') from error
        if not any(isinstance(item,ast.Assert) for item in ast.walk(test_tree)):
            raise ValueError('Enter at least one assert statement to test the generated function.')
    return {'prefix':prefix,'mode':mode,'max_new_tokens':tokens,'run_tests':run_tests,'tests':tests}

def run_coding_generation(model,tokenizer,torch,prefix,max_new_tokens,tests=None,image=None):
    from transformers import StoppingCriteria,StoppingCriteriaList
    inputs=tokenizer(prefix,return_tensors='pt',add_special_tokens=False).to('cuda')
    length=inputs['input_ids'].shape[1]
    class StopAtBoundary(StoppingCriteria):
        def __call__(self,input_ids,scores,**kwargs):
            text=tokenizer.decode(input_ids[0,length:],skip_special_tokens=False)
            return any(marker in text for marker in STOP_MARKERS)
    torch.cuda.synchronize(); start=time.monotonic()
    with torch.inference_mode():
        output=model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False,
            eos_token_id=tokenizer.convert_tokens_to_ids('<|endoftext|>'),
            pad_token_id=tokenizer.convert_tokens_to_ids('<|endoftext|>'),
            stopping_criteria=StoppingCriteriaList([StopAtBoundary()]))
    torch.cuda.synchronize(); latency=time.monotonic()-start
    tokens=output[0,length:]; text=tokenizer.decode(tokens,skip_special_tokens=False)
    completion=completion_text(text); program=prefix+completion
    try: ast.parse(program); syntax=True; syntax_error=None
    except SyntaxError as error: syntax=False; syntax_error=str(error)
    execution=None
    if tests is not None:
        try:
            execution=evaluate_program(program,tests,image) if syntax else {'passed':False,'error_type':'SyntaxError','error':syntax_error}
        except (subprocess.TimeoutExpired,OSError):
            execution={'passed':False,'error_type':'SandboxUnavailable',
                       'error':'Docker could not complete the sandbox operation or cleanup. Start or repair Docker Desktop; no host execution was attempted.'}
    return {'output_text':text,'completion':completion,'program':program,'syntax_valid':syntax,
            'syntax_error':syntax_error,'tests_requested':tests is not None,'execution':execution,
            'checks_status':checks_status(execution),
            'latency_seconds':latency,'generated_tokens':len(tokens),'hit_token_limit':len(tokens)==max_new_tokens}
