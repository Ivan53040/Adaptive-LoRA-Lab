import copy
import unittest
from unittest.mock import patch

from coding import expansion_data
from evaluation.coding_expansion import completion_text
from training.train_coding_expansion import encode_example

class Tokenizer:
    eos_token_id=999
    def encode(self,text,add_special_tokens=False):return [ord(char) for char in text]

class ExpansionTests(unittest.TestCase):
    def test_frozen_corpus_has_verified_boundaries_and_preserved_pilot_rows(self):
        manifest=expansion_data.verify_splits()
        self.assertEqual(manifest['counts'],{'train':3000,'validation':32,'test':465})
    def test_changed_reference_cannot_enter_training(self):
        original=expansion_data.read_rows
        changed=copy.deepcopy(original('train'));changed[0]['completion']+='    return None\n'
        with patch.object(expansion_data,'read_rows',side_effect=lambda split:changed if split=='train' else original(split)):
            with self.assertRaisesRegex(ValueError,'train changed'):expansion_data.verify_splits()
    def test_function_prefix_is_masked_and_completion_eos_is_supervised(self):
        row={'prompt':'def double(x):\n    """Double x."""\n','completion':'    return 2*x\n'}
        encoded=encode_example(row,Tokenizer(),2048)
        self.assertEqual(encoded['labels'],[-100]*len(row['prompt'])+[ord(c) for c in row['completion']]+[999])
        with self.assertRaisesRegex(ValueError,'overlong'):encode_example(row,Tokenizer(),8)
        self.assertEqual(completion_text(row['completion']+'\ndef next_task():\n    pass'),row['completion'])

if __name__=='__main__':unittest.main()
