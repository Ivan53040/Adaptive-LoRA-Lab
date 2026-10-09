import unittest
from experiments.fair_rank import configurations,pilot_indices,pilot_passes


class FairRankTests(unittest.TestCase):
    def test_all_common_conditions_are_identical_and_alpha_ratio_is_fixed(self):
        configs=configurations()
        self.assertEqual([cfg['lora_r'] for cfg in configs],[8,16,32])
        self.assertEqual([cfg['lora_alpha'] for cfg in configs],[16,32,64])
        self.assertTrue(all(cfg['max_steps']==1000 and cfg['epochs']==2 for cfg in configs))

    def test_pilot_is_fixed_unique_and_contains_additional_multicall_examples(self):
        rows=[{'messages':[{'tool_calls':[{}]*(2 if i>=8 else 1)}]} for i in range(30)]
        indices=pilot_indices(rows)
        self.assertEqual(indices,list(range(16)))
        self.assertEqual(len(set(indices)),16)
        self.assertEqual(sum(len(rows[i]['messages'][-1]['tool_calls'])>1 for i in indices),8)

    def test_pilot_rejects_bad_protocol_and_token_limit_hits(self):
        metrics={'valid_tool_call_rate':1,'tool_selection_accuracy':1,'mock_execution_success_rate':1,'token_limit_hit_rate':0}
        self.assertTrue(pilot_passes({'metrics':metrics}))
        for key,value in (('valid_tool_call_rate',.8),('tool_selection_accuracy',.8),('mock_execution_success_rate',.8),('token_limit_hit_rate',.01)):
            self.assertFalse(pilot_passes({'metrics':{**metrics,key:value}}))


if __name__=='__main__':
    unittest.main()
