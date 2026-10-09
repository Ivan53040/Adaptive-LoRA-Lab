"""Independently replay saved coding outputs in Docker; never regenerate model text."""
import ast
import json
from pathlib import Path
import signal
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from coding.data import read_rows
from coding.log import CodingLog
from coding.sandbox import evaluate_program
from evaluation.coding_benchmark import completion_text
from evaluation.benchmark import file_sha256
from evaluation.checkpoint import write_json_atomic
from training.experiment_state import rows_sha256

def main():
    cfg=json.loads((ROOT/'configs/coding/mbpp_r8.json').read_text())
    folder=ROOT/cfg['output_dir']/'evaluation'
    if hasattr(signal,'SIGBREAK'):
        def interrupt(*args): raise KeyboardInterrupt
        signal.signal(signal.SIGBREAK,interrupt)
    audited={}
    for variant in ('base','tool','coding'):
        for split in ('validation','test'):
            rows=read_rows(split); stem=f'{variant}-{split}-{len(rows)}-tokens512'
            path=folder/f'{stem}.jsonl'
            original=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            if len(original)!=len(rows) or {r['test_index'] for r in original}!=set(range(len(rows))):
                raise ValueError('Coding audit requires complete unique coverage.')
            fp={'max_new_tokens':512,'prediction_sha256':file_sha256(path),'rows_sha256':rows_sha256(rows),
                'image':cfg['docker_image'],'audit_code':file_sha256(Path(__file__)),
                'sandbox_code':file_sha256(ROOT/'coding/sandbox.py')}
            with CodingLog(folder/'audit'/f'{stem}.jsonl',rows,fp) as log:
                try:
                    for record in original:
                        index=record['test_index']
                        if index in log.records: continue
                        if completion_text(record['output_text'])!=record['completion']:
                            raise ValueError('Saved extracted completion differs.')
                        row=rows[index]; program=row['prompt']+record['completion']
                        try: ast.parse(program); syntax=True
                        except (SyntaxError,ValueError): syntax=False
                        result=evaluate_program(program,row['test_code'],cfg['docker_image']) if syntax else {'passed':False}
                        if syntax!=record['syntax_valid'] or result['passed']!=record['passed']:
                            raise ValueError(f'Saved coding score did not reproduce: {variant}/{split}/{row["task_id"]}.')
                        log.append({**record,'independent_replay_verified':True})
                        print(f'audit {variant}/{split} {len(log.records)}/{len(rows)} verified.',flush=True)
                except KeyboardInterrupt:
                    log.save_progress(interrupted=True); raise
                summary=json.loads((folder/f'{stem}-summary.json').read_text())
                if summary['passed']!=sum(r['passed'] for r in original):
                    raise ValueError('Coding summary differs from saved records.')
                audited[f'{variant}/{split}']={'examples':len(rows),'passed':summary['passed'],**fp}
    write_json_atomic(folder/'independent-audit.json',{'coverage_and_scores_verified':True,'results':audited})
    print('All coding validation/test records independently reproduced.')

if __name__=='__main__': main()
