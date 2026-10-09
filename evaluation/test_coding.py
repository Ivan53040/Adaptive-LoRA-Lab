import json
from pathlib import Path
import tempfile
import unittest

from coding.data import mbpp_example
from coding.log import CodingLog
from evaluation.coding_benchmark import completion_text
from training.train_coding_lora import encode_example
from training.experiment_state import rows_sha256

ROOT=Path(__file__).resolve().parents[1]
class Tokenizer:
    eos_token_id=999
    def encode(self,text,add_special_tokens=False): return [ord(char) for char in text]

class CodingTests(unittest.TestCase):
    def row(self):
        return mbpp_example({'task_id':601,'text':'Return twice the number.',
            'code':'def double(x):\n    return x * 2\n','test_list':['assert double(3)==6'],'test_setup_code':''},'train')
    def record(self,row,index=0):
        return {'test_index':index,'task_id':row['task_id'],'task_sha256':rows_sha256([row]),
                'passed':True,'output_text':'    return x * 2\n','completion':'    return x * 2\n',
                'generated_tokens':12,'syntax_valid':True,'latency_seconds':0.1,'hit_token_limit':False}
    def test_raw_prompt_contains_signature_and_request_but_no_reference_body(self):
        row=self.row()
        self.assertIn('def double(x):',row['prompt'])
        self.assertIn('Return twice the number.',row['prompt'])
        self.assertNotIn('x * 2',row['prompt'])
        compile(row['canonical_code'],'reference','exec')
    def test_labels_mask_exact_generation_prefix_and_keep_eos(self):
        row=self.row(); encoded=encode_example(row,Tokenizer(),2048)
        start=len(row['prompt'])
        self.assertEqual(encoded['input_ids'][:start],Tokenizer().encode(row['prompt']))
        self.assertTrue(all(v==-100 for v in encoded['labels'][:start]))
        self.assertEqual(encoded['labels'][start:],Tokenizer().encode(row['completion'])+[999])
    def test_overlong_targets_are_rejected_instead_of_truncated(self):
        with self.assertRaisesRegex(ValueError,'overlong'):
            encode_example(self.row(),Tokenizer(),5)
    def test_next_top_level_function_is_excluded_but_nested_functions_remain(self):
        text='    def helper():\n        return 1\n    return helper()\n\ndef unrelated():\n    return 0'
        self.assertEqual(completion_text(text),'    def helper():\n        return 1\n    return helper()\n')
    def test_torn_final_record_recovery_does_not_repeat_completed_case(self):
        row=self.row()
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(ROOT))
            path=Path(folder)/'cases.jsonl'; fp={'max_new_tokens':32}
            with CodingLog(path,[row],fp) as log:
                log.append(self.record(row)); log.save_progress(peak=6)
            with path.open('ab') as stream: stream.write(b'{"partial":')
            with CodingLog(path,[row],fp) as log:
                self.assertEqual(len(log.records),1); self.assertEqual(log.peak,6)
            self.assertEqual(len(path.read_text().splitlines()),1)
    def test_changed_recipe_and_duplicate_case_are_rejected(self):
        row=self.row()
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(ROOT))
            path=Path(folder)/'cases.jsonl'
            with CodingLog(path,[row],{'max_new_tokens':32}) as log:
                log.append(self.record(row))
                with self.assertRaisesRegex(ValueError,'overwrite'): log.append(self.record(row))
            with self.assertRaisesRegex(ValueError,'settings changed'):
                with CodingLog(path,[row],{'max_new_tokens':64}): pass

if __name__=='__main__': unittest.main()
