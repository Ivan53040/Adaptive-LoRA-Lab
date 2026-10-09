import unittest
from evaluation.final_research_summary import range_summary

class FinalSummaryTests(unittest.TestCase):
    def test_mean_and_range_account_for_all_conditions(self):
        value=range_summary([90,88,90])
        self.assertAlmostEqual(value['mean'],89+1/3)
        self.assertEqual((value['minimum'],value['maximum'],value['conditions']),(88,90,3))
    def test_empty_input_is_not_a_completed_study(self):
        with self.assertRaisesRegex(ValueError,'at least one'):range_summary([])

if __name__=='__main__':unittest.main()
