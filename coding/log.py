"""Durable coding predictions with recipe validation and single-writer recovery."""
import json
import os
from evaluation.checkpoint import exclusive_prediction_file,write_json_atomic
from training.experiment_state import rows_sha256

class CodingLog:
    def __init__(self,path,rows,fingerprint):
        self.path=path; self.rows=rows; self.fingerprint=fingerprint; self.records={}; self.peak=None
        self.metadata=path.with_suffix('.metadata.json'); self.progress=path.with_suffix('.progress.json')
    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.context=exclusive_prediction_file(self.path); self.stream=self.context.__enter__()
        try:
            raw=self.stream.read()
            if self.metadata.exists() and json.loads(self.metadata.read_text())!=self.fingerprint:
                raise ValueError('Coding model, data or evaluation settings changed; use a new output directory.')
            if raw and not self.metadata.exists():
                raise ValueError('Coding predictions lack their immutable fingerprint.')
            offset=0
            for i,line in enumerate(raw.splitlines(keepends=True)):
                try:
                    record=json.loads(line)
                except (ValueError,UnicodeDecodeError):
                    if i!=len(raw.splitlines())-1:
                        raise ValueError('Invalid record inside the coding log.')
                    self.stream.seek(offset); self.stream.truncate(); break
                self.validate(record)
                if record['test_index'] in self.records:
                    raise ValueError('Duplicate coding result.')
                self.records[record['test_index']]=record; offset+=len(line)
            self.stream.seek(0,os.SEEK_END)
            if self.stream.tell() and not raw[:self.stream.tell()].endswith(b'\n'):
                self.stream.write(b'\n')
            write_json_atomic(self.metadata,self.fingerprint)
            if self.progress.exists():
                self.peak=json.loads(self.progress.read_text()).get('peak_allocated_vram_gib')
            self.save_progress(); return self
        except BaseException:
            self.context.__exit__(None,None,None); raise
    def __exit__(self,*args):
        return self.context.__exit__(*args)
    def validate(self,record):
        index=record.get('test_index')
        if type(index) is not int or not 0<=index<len(self.rows):
            raise ValueError('Invalid coding test index.')
        row=self.rows[index]
        if record.get('task_sha256')!=rows_sha256([row]) or record.get('task_id')!=row['task_id']:
            raise ValueError('Saved coding question/reference changed.')
        if type(record.get('passed')) is not bool or not {'output_text','completion','generated_tokens','syntax_valid','latency_seconds','hit_token_limit'}.issubset(record):
            raise ValueError('Incomplete coding prediction.')
        if record['generated_tokens']>self.fingerprint['max_new_tokens']:
            raise ValueError('Saved coding token budget differs.')
    def append(self,record):
        self.validate(record)
        if record['test_index'] in self.records:
            raise ValueError('Cannot overwrite a completed coding case.')
        self.stream.write((json.dumps(record,ensure_ascii=False)+'\n').encode())
        self.stream.flush(); os.fsync(self.stream.fileno())
        self.records[record['test_index']]=record; self.save_progress()
    def save_progress(self,interrupted=False,peak=None):
        if peak is not None:
            self.peak=max(self.peak or 0,peak)
        write_json_atomic(self.progress,{'status':'completed' if len(self.records)==len(self.rows) else 'interrupted' if interrupted else 'in_progress',
            'completed':len(self.records),'remaining':len(self.rows)-len(self.records),'peak_allocated_vram_gib':self.peak})
