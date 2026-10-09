import unittest
from training.repair_protocol import completion_example, parameter_groups
from experiments.protocol_repair import validation_passes


class TokenizerStub:
    eos_token_id = 9
    def __init__(self,full=None,prompt=None):
        self.full = [1,2,3,4,5,9,7] if full is None else full
        self.prompt = [1,2,3] if prompt is None else prompt
    def apply_chat_template(self,messages,add_generation_prompt=False,**kwargs):
        ids = self.prompt if add_generation_prompt else self.full
        return {'input_ids':ids,'attention_mask':[1]*len(ids)}


class ParameterStub:
    def __init__(self,trainable=True):
        self.requires_grad = trainable


class ModelStub:
    def __init__(self,values):
        self.values = values
    def named_parameters(self):
        return iter(self.values)


class ProtocolRepairTests(unittest.TestCase):
    row = {'messages':[{'role':'user','content':'question'},{'role':'assistant','content':'answer'}], 'tools':[]}

    def test_official_test_gate_requires_format_tools_mock_and_termination(self):
        metrics = {'valid_tool_call_rate':1,'tool_selection_accuracy':.99,
                   'mock_execution_success_rate':.99,'token_limit_hit_rate':0}
        self.assertTrue(validation_passes({'metrics':metrics}))
        for key,value in (('valid_tool_call_rate',.94),('tool_selection_accuracy',.94),
                          ('mock_execution_success_rate',.94),('token_limit_hit_rate',.06)):
            self.assertFalse(validation_passes({'metrics':{**metrics,key:value}}))

    def test_masks_entire_prefill_and_tokens_after_eos_but_keeps_completion(self):
        example = completion_example(self.row,TokenizerStub(),20)
        self.assertEqual(example['labels'],[-100,-100,-100,4,5,9,-100])
        self.assertEqual(example['input_ids'],[1,2,3,4,5,9,7])

    def test_rejects_mismatched_prompt_missing_eos_and_overlong_reference(self):
        for tokenizer,length in ((TokenizerStub(prompt=[8,2,3]),20),
                                 (TokenizerStub(full=[1,2,3,4]),20),(TokenizerStub(),4)):
            with self.assertRaises(ValueError):
                completion_example(self.row,tokenizer,length)

    def test_optimizer_covers_every_trainable_lora_and_keeps_two_learning_rates(self):
        head,body,frozen = ParameterStub(),ParameterStub(),ParameterStub(False)
        model = ModelStub([('model.lm_head.lora_A.default.weight',head),
                           ('model.q_proj.lora_B.default.weight',body),('model.weight',frozen)])
        groups = parameter_groups(model,.001,.0001)
        self.assertEqual(groups[0]['params'],[head])
        self.assertEqual(groups[1]['params'],[body])
        self.assertEqual([group['lr'] for group in groups],[.001,.0001])

    def test_rejects_accidental_base_training(self):
        with self.assertRaises(ValueError):
            parameter_groups(ModelStub([('model.weight',ParameterStub())]),.001,.0001)


if __name__ == '__main__':
    unittest.main()
