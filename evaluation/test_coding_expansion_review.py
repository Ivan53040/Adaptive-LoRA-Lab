import json
from pathlib import Path
import tempfile
import unittest

from evaluation.benchmark import file_sha256
from evaluation.review_coding_expansion import load_verified_records,paired_counts
from training.experiment_state import rows_sha256

ROOT=Path(__file__).resolve().parents[1]

class ExpansionReviewTests(unittest.TestCase):
    def test_pairing_accounts_for_every_task_and_net_gain(self):
        left={i:{'passed':value} for i,value in enumerate((True,False,True,False,False))}
        right={i:{'passed':value} for i,value in enumerate((True,True,False,False,True))}
        p=paired_counts(left,right)
        self.assertEqual([p[key] for key in ('both_passed','right_only_passed','left_only_passed','neither_passed')],[1,2,1,1])
        self.assertEqual(p['net_gain'],1);self.assertEqual(p['percentage_point_change'],20)
    def test_different_or_empty_coverage_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'coverage'):paired_counts({0:{'passed':True}},{1:{'passed':True}})
        with self.assertRaisesRegex(ValueError,'coverage'):paired_counts({},{})
    def test_altered_predictions_cannot_be_presented_as_audited(self):
        rows=[{'task_id':'toy','prompt':'def f():\n','completion':'    return 1\n'}]
        record={'test_index':0,'task_id':'toy','task_sha256':rows_sha256(rows),
                'passed':True,'syntax_valid':True,'output_text':'    return 1\n','completion':'    return 1\n'}
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as folder:
            path=Path(folder)/'predictions.jsonl';path.write_text(json.dumps(record)+'\n')
            audited={'prediction_sha256':file_sha256(path),'rows_sha256':rows_sha256(rows),'examples':1}
            self.assertEqual(load_verified_records(path,rows,audited)[0]['passed'],True)
            record['passed']=False;path.write_text(json.dumps(record)+'\n')
            with self.assertRaisesRegex(ValueError,'changed after'):load_verified_records(path,rows,audited)

if __name__=='__main__':unittest.main()
