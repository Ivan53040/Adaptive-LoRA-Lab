"""Independently replay the new MBPP comparisons in pinned Docker containers."""
import ast
import json
from pathlib import Path
import signal
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from coding.expansion_data import read_rows,verify_splits
from coding.log import CodingLog
from coding.sandbox import evaluate_program
from evaluation.coding_expansion import completion_text
from evaluation.benchmark import file_sha256
from evaluation.checkpoint import write_json_atomic
from training.experiment_state import rows_sha256

STUDY=ROOT/'outputs/coding-expansion-3k'
CONDITIONS={'base':('base',STUDY/'base/evaluation'),
            'pilot':('coding',STUDY/'pilot/evaluation'),
            'expanded':('coding',STUDY/'r8/evaluation')}

def main():
    verify_splits()
    cfg=json.loads((ROOT/'configs/coding/expansion_r8.json').read_text())
    if hasattr(signal,'SIGBREAK'):
        def interrupt(*args):raise KeyboardInterrupt
        signal.signal(signal.SIGBREAK,interrupt)
    audited={}
    for condition,(variant,folder) in CONDITIONS.items():
        for split in ('validation','test'):
            rows=read_rows(split);stem=f'{variant}-{split}-{len(rows)}-tokens512';path=folder/f'{stem}.jsonl'
            original=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            if len(original)!=len(rows) or {r['test_index'] for r in original}!=set(range(len(rows))):
                raise ValueError('Expansion audit requires complete unique coverage.')
            summary=json.loads((folder/f'{stem}-summary.json').read_text())
            passed=sum(r['passed'] for r in original)
            if (summary['examples']!=len(rows) or summary['passed']!=passed
                or summary['pass_at_1']!=passed/len(rows) or summary['rows_sha256']!=rows_sha256(rows)
                or summary['syntax_valid_rate']!=sum(r['syntax_valid'] for r in original)/len(rows)):
                raise ValueError('Expansion summary differs from saved predictions.')
            fp={'max_new_tokens':512,'prediction_sha256':file_sha256(path),'rows_sha256':rows_sha256(rows),
                'image':cfg['docker_image'],'audit_code':file_sha256(Path(__file__)),
                'data_reader':file_sha256(ROOT/'coding/expansion_data.py'),
                'sandbox_code':file_sha256(ROOT/'coding/sandbox.py')}
            with CodingLog(STUDY/'audit'/f'{condition}-{split}.jsonl',rows,fp) as log:
                try:
                    for record in original:
                        index=record['test_index']
                        if index in log.records:continue
                        if completion_text(record['output_text'])!=record['completion']:
                            raise ValueError('Saved extracted completion differs.')
                        row=rows[index];program=row['prompt']+record['completion']
                        try:ast.parse(program);syntax=True
                        except (SyntaxError,ValueError):syntax=False
                        result=evaluate_program(program,row['test_code'],cfg['docker_image']) if syntax else {'passed':False}
                        if syntax!=record['syntax_valid'] or result['passed']!=record['passed']:
                            raise ValueError(f'Prediction did not reproduce: {condition}/{split}/{row["task_id"]}.')
                        log.append({**record,'independent_replay_verified':True})
                        print(f'audit {condition}/{split} {len(log.records)}/{len(rows)} verified.',flush=True)
                except KeyboardInterrupt:
                    log.save_progress(interrupted=True);raise
            audited[f'{condition}/{split}']={'examples':len(rows),'passed':passed,**fp}
    write_json_atomic(STUDY/'independent-audit.json',{'coverage_and_scores_verified':True,'results':audited})

if __name__=='__main__':main()
