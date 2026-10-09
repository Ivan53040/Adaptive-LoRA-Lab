"""Freeze a small MBPP training pilot and untouched HumanEval final test."""
import ast
import copy
import json
from pathlib import Path
import random
import textwrap

from evaluation.benchmark import file_sha256
from evaluation.checkpoint import write_json_atomic
from training.experiment_state import rows_sha256
from coding.sandbox import evaluate_program,verify_sandbox

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/processed/coding-mbpp-pilot'

def body_key(code):
    tree=ast.parse(code)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    if len(functions)!=1:
        return None
    body=copy.deepcopy(functions[0].body)
    if body and isinstance(body[0],ast.Expr) and isinstance(body[0].value,ast.Constant) and isinstance(body[0].value.value,str):
        body=body[1:]
    return '\n'.join(ast.dump(n,include_attributes=False) for n in body)

def mbpp_example(row,split):
    tree=ast.parse(row['code'])
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    if len(functions)!=1 or any(not isinstance(n,(ast.FunctionDef,ast.Import,ast.ImportFrom)) for n in tree.body):
        raise ValueError('Pilot supports one top-level function and imports only.')
    function=functions[0]
    if function.decorator_list or tree.body[-1] is not function:
        raise ValueError('Decorated functions or trailing imports are outside this pilot.')
    header=copy.deepcopy(function); header.body=[ast.Pass()]
    signature=ast.unparse(header).splitlines()[0]
    imports='\n'.join(ast.unparse(n) for n in tree.body[:-1])
    description=row['text'].replace('\\','\\\\').replace('"""','\\"\\"\\"')
    prompt=(imports+'\n\n' if imports else '')+signature+'\n    """'+description+'\n    """\n'
    body=copy.deepcopy(function.body)
    if body and isinstance(body[0],ast.Expr) and isinstance(body[0].value,ast.Constant) and isinstance(body[0].value.value,str):
        body=body[1:]
    completion=textwrap.indent('\n'.join(ast.unparse(n) for n in body),'    ')+'\n'
    tests=row.get('test_setup_code','')+'\n'+'\n'.join(row['test_list'])
    return {'task_id':f"MBPP/{row['task_id']}",'dataset':'mbpp','split':split,'prompt':prompt,
            'completion':completion,'entry_point':function.name,'test_code':tests,
            'canonical_code':prompt+completion}

def read_rows(name):
    return [json.loads(line) for line in (DATA/f'{name}.jsonl').read_text(encoding='utf-8').splitlines()]

def prepare(image):
    import pyarrow.parquet as pq
    source=ROOT/'data/raw/coding'
    provenance=json.loads((source/'sources.json').read_text())
    files=[source/'sources.json']+[source/name/p for name,entry in zip(('mbpp','humaneval'),provenance) for p in entry['files']]
    protocol={'seed':42,'raw_files':{str(p.relative_to(ROOT)):file_sha256(p) for p in files},
              'preparation_code_sha256':file_sha256(Path(__file__)),
              'sandbox_code_sha256':file_sha256(ROOT/'coding/sandbox.py'),'docker_image':image,
              'scope':'Single-function MBPP training pilot; raw function-completion prompts; untouched HumanEval164 final test.'}
    DATA.mkdir(parents=True,exist_ok=True)
    manifest_path=DATA/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest['protocol']!=protocol:
            raise ValueError('Frozen coding preparation changed; start a separate data directory.')
        for split,digest in manifest['row_sha256'].items():
            if rows_sha256(read_rows(split))!=digest:
                raise ValueError('Frozen coding rows changed.')
        return manifest
    checks=verify_sandbox(image)
    human_raw=pq.read_table(source/'humaneval/openai_humaneval/test-00000-of-00001.parquet').to_pylist()
    assert len(human_raw)==164
    test=[{'task_id':r['task_id'],'dataset':'humaneval','split':'test','prompt':r['prompt'],
           'completion':r['canonical_solution'],'entry_point':r['entry_point'],
           'test_code':r['test']+f"\ncheck({r['entry_point']})\n",'canonical_code':r['prompt']+r['canonical_solution']} for r in human_raw]
    human_bodies={body_key(r['canonical_code']) for r in test}
    excluded=[]; selected={}
    for split in ('validation','train'):
        raw=pq.read_table(source/f'mbpp/full/{split}-00000-of-00001.parquet').to_pylist()
        pool=[]
        for item in raw:
            try:
                row=mbpp_example(item,split)
                key=body_key(row['canonical_code'])
                if key in human_bodies or any(key==body_key(r['canonical_code']) for r in selected.get('validation',[])+pool):
                    raise ValueError('Exact canonical body overlaps held-out/reference or another task.')
                result=evaluate_program(row['canonical_code'],row['test_code'],image)
                if not result['passed']:
                    raise ValueError('Reference failed isolated tests: '+result.get('error_type','unknown'))
                pool.append(row)
            except (ValueError,SyntaxError) as error:
                excluded.append({'task_id':f"MBPP/{item['task_id']}",'split':split,'reason':str(error)})
        random.Random(42).shuffle(pool)
        selected[split]=pool
        print(f'{split}: {len(pool)}/{len(raw)} reference-verified eligible single-function tasks.',flush=True)
    if len(selected['train'])<100 or len(selected['validation'])<20:
        raise ValueError('Insufficient eligible coding pilot data.')
    # The full eligible validation set is frozen; a deterministic first32 is the pilot evaluation budget.
    selected['validation']=selected['validation'][:32]
    selected['test']=test
    for split,rows in selected.items():
        (DATA/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    manifest={'protocol':protocol,'sources':provenance,'counts':{s:len(r) for s,r in selected.items()},
              'row_sha256':{s:rows_sha256(r) for s,r in selected.items()},'excluded_mbpp':excluded,
              'sandbox_preflight':checks,'overlap_check':'Exact AST function-body overlap removed from project training. This does not certify absence of Base pretraining contamination.'}
    write_json_atomic(manifest_path,manifest)
    return manifest

if __name__=='__main__':
    import argparse
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--image',required=True)
    args=cli.parse_args()
    result=prepare(args.image)
    print(json.dumps({'counts':result['counts'],'manifest':str(DATA/'manifest.json')},indent=2))
