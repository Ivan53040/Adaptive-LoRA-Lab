"""Read the separate frozen expansion corpus and validate its split boundaries."""
import json
from pathlib import Path

from coding.data import body_key,read_rows as pilot_rows
from training.experiment_state import rows_sha256

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/processed/coding-expansion-3k'

def read_rows(name):
    if name not in ('train','validation','test'):
        raise ValueError('Unknown expansion split.')
    return [json.loads(line) for line in (DATA/f'{name}.jsonl').read_text(encoding='utf-8').splitlines()]

def verify_splits():
    manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    groups={name:read_rows(name) for name in ('train','validation','test')}
    if manifest['counts']!={'train':3000,'validation':32,'test':465}:
        raise ValueError('Unexpected frozen expansion counts.')
    keys={}
    for name,rows in groups.items():
        if len(rows)!=manifest['counts'][name] or rows_sha256(rows)!=manifest['row_sha256'][name]:
            raise ValueError(f'Frozen expansion {name} changed.')
        if len({row['task_id'] for row in rows})!=len(rows):
            raise ValueError('Duplicate task identifiers.')
        keys[name]={body_key(row['canonical_code']) for row in rows}
        if None in keys[name] or len(keys[name])!=len(rows):
            raise ValueError('Invalid or duplicate function bodies.')
        if any(row['canonical_code']!=row['prompt']+row['completion'] for row in rows):
            raise ValueError('Canonical completion differs from training representation.')
    if any(keys[a]&keys[b] for a,b in (('train','validation'),('train','test'),('validation','test'))):
        raise ValueError('Exact function bodies overlap splits.')
    if keys['train']&{body_key(row['canonical_code']) for row in pilot_rows('test')}:
        raise ValueError('Expansion training overlaps frozen HumanEval bodies.')
    if groups['validation']!=pilot_rows('validation'):
        raise ValueError('Expansion validation must preserve the pilot validation.')
    old_train={row['task_id']:row for row in pilot_rows('train')}
    actual={row['task_id']:row for row in groups['train']}
    if any(actual.get(task)!=row for task,row in old_train.items()):
        raise ValueError('Expansion must retain all original training rows unchanged.')
    return manifest
