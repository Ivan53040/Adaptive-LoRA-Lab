"""Execute benchmark programs only in constrained, disposable Docker containers."""
import json
import subprocess
import uuid

RUNNER = r'''
import contextlib,io,json,sys
item=json.load(sys.stdin)
output=io.StringIO()
result={'passed':False}
try:
    scope={}
    with contextlib.redirect_stdout(output),contextlib.redirect_stderr(output):
        exec(compile(item['program'],'candidate.py','exec'),scope)
        exec(compile(item['test_code'],'tests.py','exec'),scope)
    result={'passed':True}
except BaseException as error:
    result={'passed':False,'error_type':type(error).__name__,'error':str(error)[:500]}
print('CODING_RESULT:'+json.dumps(result),flush=True)
'''

def evaluate_program(program,test_code,image,timeout=8):
    if not image.startswith('sha256:'):
        raise ValueError('Use the pinned local Docker image ID.')
    name='adaptive-coding-'+uuid.uuid4().hex
    command=['docker','run','--rm','--name',name,'--pull','never','--network','none',
             '--read-only','--cap-drop','ALL','--security-opt','no-new-privileges',
             '--user','65534:65534','--pids-limit','32','--memory','256m','--memory-swap','256m',
             '--cpus','1','--ulimit','nofile=64:64','--tmpfs','/tmp:rw,noexec,nosuid,size=16m',
             '-i',image,'python','-I','-B','-u','-c',RUNNER]
    try:
        result=subprocess.run(command,input=json.dumps({'program':program,'test_code':test_code}),
                              text=True,encoding='utf-8',capture_output=True,timeout=timeout)
    except subprocess.TimeoutExpired:
        return {'passed':False,'error_type':'Timeout','error':f'Exceeded {timeout}s including container startup.'}
    finally:
        # Only this call's exact disposable container can be removed.
        subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=15)
    lines=[line[len('CODING_RESULT:'):] for line in result.stdout.splitlines() if line.startswith('CODING_RESULT:')]
    if result.returncode or len(lines)!=1:
        return {'passed':False,'error_type':'ContainerError','error':result.stderr[-500:] or 'No trusted harness result.'}
    value=json.loads(lines[0])
    if type(value.get('passed')) is not bool:
        raise ValueError('Invalid sandbox result.')
    return value

def verify_sandbox(image):
    checks={
        'correct':evaluate_program('def answer(): return 42','assert answer()==42',image),
        'incorrect':evaluate_program('def answer(): return 0','assert answer()==42',image),
        'readonly':evaluate_program("open('/forbidden-write','w').write('x')",'assert True',image),
        'network':evaluate_program("import socket; socket.create_connection(('1.1.1.1',443),timeout=1)",'assert True',image),
        'timeout':evaluate_program('while True: pass','assert True',image,timeout=3),
    }
    assert checks['correct']['passed'] and all(not checks[key]['passed'] for key in checks if key!='correct'),checks
    return checks
