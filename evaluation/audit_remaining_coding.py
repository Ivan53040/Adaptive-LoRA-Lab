"""Replay a completed remaining-study log in CPU Docker; never load model weights."""
import argparse
import ast
import json
from pathlib import Path
import signal
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from coding import data as pilot
from coding import expansion_data as expanded
from coding.log import CodingLog
from coding.sandbox import evaluate_program
from evaluation.benchmark import file_sha256
from evaluation.checkpoint import write_json_atomic
from evaluation.coding_benchmark import completion_text
from evaluation.review_coding_expansion import load_verified_records
from training.experiment_state import rows_sha256

def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--predictions',type=Path,required=True);cli.add_argument('--config',type=Path,required=True)
    cli.add_argument('--family',choices=('pilot','expanded'),required=True);cli.add_argument('--split',choices=('validation','test'),required=True)
    args=cli.parse_args();path=args.predictions.resolve()
    if not path.is_relative_to(ROOT):raise ValueError('Audit must remain in this project.')
    family=pilot if args.family=='pilot' else expanded
    rows=family.read_rows(args.split);manifest=json.loads((family.DATA/'manifest.json').read_text())
    if rows_sha256(rows)!=manifest['row_sha256'][args.split]:raise ValueError('Frozen audit rows changed.')
    cfg=json.loads(args.config.read_text());source={'examples':len(rows),'prediction_sha256':file_sha256(path),'rows_sha256':rows_sha256(rows)}
    original=load_verified_records(path,rows,source)
    meta=json.loads(path.with_suffix('.metadata.json').read_text())
    if meta['rows_sha256']!=source['rows_sha256'] or meta['max_new_tokens']!=512 or meta['docker_image']!=cfg['docker_image']:
        raise ValueError('Evaluation protocol does not match this audit.')
    fp={'max_new_tokens':512,**source,'image':cfg['docker_image'],'audit_code':file_sha256(Path(__file__)),
        'sandbox_code':file_sha256(ROOT/'coding/sandbox.py'),'review_guard':file_sha256(ROOT/'evaluation/review_coding_expansion.py')}
    if hasattr(signal,'SIGBREAK'):
        def interrupt(*args):raise KeyboardInterrupt
        signal.signal(signal.SIGBREAK,interrupt)
    with CodingLog(path.parent/'audit'/path.name,rows,fp) as log:
        try:
            for index,r in original.items():
                if index in log.records:continue
                program=rows[index]['prompt']+r['completion']
                try:ast.parse(program);syntax=True
                except (SyntaxError,ValueError):syntax=False
                result=evaluate_program(program,rows[index]['test_code'],cfg['docker_image']) if syntax else {'passed':False}
                if syntax!=r['syntax_valid'] or result['passed']!=r['passed']:
                    raise ValueError(f'Case did not independently reproduce: {rows[index]["task_id"]}.')
                log.append({**r,'independent_replay_verified':True});print(f'audit {path.stem} {len(log.records)}/{len(rows)}',flush=True)
        except KeyboardInterrupt:
            log.save_progress(interrupted=True);raise
    summary=json.loads(path.with_name(path.stem+'-summary.json').read_text())
    if summary['examples']!=len(rows) or summary['passed']!=sum(r['passed'] for r in original.values()):
        raise ValueError('Summary differs from reproduced predictions.')
    write_json_atomic(path.with_name(path.stem+'-audit.json'),{'coverage_and_scores_verified':True,'passed':summary['passed'],**fp})

if __name__=='__main__':main()
