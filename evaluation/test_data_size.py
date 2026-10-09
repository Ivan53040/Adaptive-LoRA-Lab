import json
from pathlib import Path
import unittest
from experiments.data_size import configurations,select_size,verify_baseline_and_splits


class DataSizeTests(unittest.TestCase):
    def test_fixed_epochs_and_shared_corrected_settings(self):
        configs=configurations()
        self.assertEqual([cfg['training_rows'] for cfg in configs],[1000,3000,8258])
        self.assertEqual([cfg['max_steps'] for cfg in configs],[1000,3000,8258])
        self.assertTrue(all(cfg['lora_r']==8 and cfg['epochs']==2 for cfg in configs))

    def test_baseline_reuse_has_identical_kernel_and_excluded_eval_rows(self):
        plan=verify_baseline_and_splits(configurations())
        self.assertTrue(plan['kernel_preserved'])
        self.assertEqual(plan['validation_rows'],100)
        self.assertEqual(plan['test_rows'],961)
        self.assertEqual(plan['train_sha256']['1k'],'b6655b007b63455214dcac5a59114aa5cedc434d82594056c6e11d1538a3b999')

    def test_selection_uses_validation_and_fewer_rows_on_tie(self):
        def result(name,count,exact,mock):
            return {'name':name,'training_rows':count,'validation':{'metrics':{'full_call_exact_match_rate':exact,'mock_execution_success_rate':mock}},
                    'test':{'full_call_exact_match_rate':1}}
        rows=[result('1k',1000,.9,1),result('3k',3000,.91,.99),result('full',8258,.91,1)]
        self.assertEqual(select_size(rows),'full')
        rows[1]=result('3k',3000,.91,1)
        self.assertEqual(select_size(rows),'3k')
        for row in rows:
            row['validation']['metrics']['full_call_exact_match_rate']=.9
        self.assertEqual(select_size(rows),'1k')
        with self.assertRaises(ValueError):
            select_size(rows[:2])


if __name__=='__main__':
    unittest.main()
