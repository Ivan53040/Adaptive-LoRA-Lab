"""CPU/Docker replay of the three completed follow-up validation conditions."""
import ast
import json
from pathlib import Path
import signal
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from coding.data import read_rows
from coding.log import CodingLog
from coding.sandbox import evaluate_program
from evaluation.benchmark import file_sha256
from evaluation.checkpoint import write_json_atomic
from evaluation.coding_benchmark import completion_text
from evaluation.review_coding_expansion import load_verified_records
from training.experiment_state import rows_sha256

STUDY=ROOT/'outputs/coding-followup'
NAMES=('seed43','seed44','attention-head-seed42')

def main():
    study=json.loads((STUDY/'study_status.json').read_text())
    if study['status']!='completed':raise ValueError('Follow-up training and validation must be completed first.')
    rows=read_rows('validation');audited={}
    if hasattr(signal,'SIGBREAK'):
        def interrupt(*args):raise KeyboardInterrupt
        signal.signal(signal.SIGBREAK,interrupt)
    for name in NAMES:
        cfg=json.loads((ROOT/f'configs/coding_followup/{name}.json').read_text())
        folder=STUDY/name/'evaluation';path=folder/'coding-validation-32-tokens512.jsonl'
        source={'examples':len(rows),'prediction_sha256':file_sha256(path),'rows_sha256':rows_sha256(rows)}
        original=load_verified_records(path,rows,source)
        summary=json.loads(path.with_name(path.stem+'-summary.json').read_text())
        if (summary['examples']!=32 or summary['passed']!=sum(r['passed'] for r in original.values())
            or summary['rows_sha256']!=rows_sha256(rows)):
            raise ValueError('Follow-up summary does not match saved predictions.')
        fp={'max_new_tokens':512,**source,'image':cfg['docker_image'],
            'audit_code':file_sha256(Path(__file__)),'sandbox_code':file_sha256(ROOT/'coding/sandbox.py')}
        with CodingLog(STUDY/'audit'/f'{name}-validation.jsonl',rows,fp) as log:
            try:
                for index,r in original.items():
                    if index in log.records:continue
                    program=rows[index]['prompt']+completion_text(r['output_text'])
                    try:ast.parse(program);syntax=True
                    except (SyntaxError,ValueError):syntax=False
                    result=evaluate_program(program,rows[index]['test_code'],cfg['docker_image']) if syntax else {'passed':False}
                    if syntax!=r['syntax_valid'] or result['passed']!=r['passed']:
                        raise ValueError(f'Follow-up score did not reproduce: {name}/{rows[index]["task_id"]}.')
                    log.append({**r,'independent_replay_verified':True})
                    print(f'audit {name} {len(log.records)}/32 verified.',flush=True)
            except KeyboardInterrupt:
                log.save_progress(interrupted=True);raise
        audited[name]={'examples':len(rows),'passed':summary['passed'],**fp}
    write_json_atomic(STUDY/'independent-audit.json',{'coverage_and_scores_verified':True,'examples':96,'results':audited})
    (STUDY/'independent-audit.md').write_text('# Coding follow-up: independent replay\n\nAll96 validation predictions in seed43, seed44 and attention+head seed42 reproduced in pinned CPU Docker containers. No model text regenerated.\n',encoding='utf-8')
    print('All96 follow-up validation predictions independently reproduced.')

if __name__=='__main__':main()
