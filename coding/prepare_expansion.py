"""Prepare a separate broader corpus; never alter the completed coding pilot."""
from collections import Counter
import json
from pathlib import Path
import random
import re
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from coding.data import read_rows,mbpp_example,body_key
from coding.sandbox import evaluate_program
from evaluation.benchmark import file_sha256
from evaluation.checkpoint import write_json_atomic
from training.experiment_state import rows_sha256

OUT=ROOT/'data/processed/coding-expansion-3k'
IMAGE='sha256:dddfd7e07f9d15aeeca61529320492139d21cac7f0070c00609243e51e4e0016'

def main():
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    source=ROOT/'data/raw/coding/magicoder/data-oss_instruct-decontaminated.jsonl'
    mbpp=ROOT/'data/raw/coding/mbpp/full/test-00000-of-00001.parquet'
    protocol={'magicoder_revision':'5f839b1f368a76b161028bb9edff055db34022b2','magicoder_sha256':file_sha256(source),
              'mbpp_revision':'4bb6404fdc6cacfda99d4ac4205087b89d32030c','mbpp_test_sha256':file_sha256(mbpp),
              'preparation_code':file_sha256(Path(__file__)),'image':IMAGE,'seed':42,'target_train_rows':3000}
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').exists():
        if json.loads((OUT/'manifest.json').read_text())['protocol']!=protocol:raise ValueError('Expansion protocol changed.')
        print('Expansion corpus already frozen.');return
    train=read_rows('train');validation=read_rows('validation');human=read_rows('test')
    raw_test=pq.read_table(mbpp).to_pylist()
    blocked={body_key(row['canonical_code']) for row in train+validation+human}
    test=[];test_excluded=[]
    for index,item in enumerate(raw_test):
        try:
            row=mbpp_example(item,'test');key=body_key(row['canonical_code'])
            if key in blocked:raise ValueError('Exact body overlaps pilot train/validation/HumanEval.')
            checked=evaluate_program(row['canonical_code'],row['test_code'],IMAGE)
            if not checked['passed']:raise ValueError('Reference failed pinned isolated tests: '+checked.get('error_type','unknown'))
            test.append(row);blocked.add(key)
        except (ValueError,SyntaxError) as error:test_excluded.append({'id':item['task_id'],'reason':str(error)})
        if (index+1)%50==0:print(f'New MBPP reference preparation {index+1}/{len(raw_test)}; eligible={len(test)}.',flush=True)
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/Qwen3-8B-Base',local_files_only=True)
    candidates=[];excluded=Counter();seen=set(blocked)
    for line in source.open(encoding='utf-8'):
        raw=json.loads(line)
        if raw['lang'] not in ('python','py'):continue
        blocks=re.findall(r'```(?:python|py)?[ \t]*\n(.*?)```',raw['solution'],re.S)
        code=blocks[0] if blocks else raw['solution']
        try:
            row=mbpp_example({'task_id':raw['index'],'text':raw['problem'],'code':code,'test_list':[],'test_setup_code':''},'train')
            key=body_key(row['canonical_code'])
            if key in seen:raise ValueError('Exact body duplicate or held-out overlap.')
            if len(tokenizer.encode(row['prompt'],add_special_tokens=False))+len(tokenizer.encode(row['completion'],add_special_tokens=False))+1>2048:
                raise ValueError('Over2048 tokens.')
            row.update(task_id=f"Magicoder/{raw['index']}",dataset='magicoder',reference_tests_verified=False)
            candidates.append(row);seen.add(key)
        except (ValueError,SyntaxError) as error:excluded[str(error)]+=1
    random.Random(42).shuffle(candidates)
    if len(candidates)<3000-len(train) or len(test)<100:raise ValueError('Insufficient eligible expansion data.')
    combined=train+candidates[:3000-len(train)];random.Random(42).shuffle(combined)
    groups={'train':combined,'validation':validation,'test':test}
    for name,rows in groups.items():
        (OUT/f'{name}.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')
    manifest={'protocol':protocol,'counts':{name:len(rows) for name,rows in groups.items()},
              'row_sha256':{name:rows_sha256(rows) for name,rows in groups.items()},
              'magicoder_eligible_pool':len(candidates),'magicoder_exclusions':dict(excluded),'mbpp_test_exclusions':test_excluded,
              'training_quality':'339 reference-tested MBPP plus synthetic MIT Magicoder single-function references. Synthetic correctness is not execution-certified.',
              'evaluation_scope':'New eligible MBPP held-out subset, not full official MBPP. HumanEval remains a frozen historical result; exact body filtering does not certify semantic/pretraining decontamination.'}
    write_json_atomic(OUT/'manifest.json',manifest)
    print(json.dumps({'counts':manifest['counts'],'magicoder_pool':len(candidates)},indent=2))

if __name__=='__main__':main()
