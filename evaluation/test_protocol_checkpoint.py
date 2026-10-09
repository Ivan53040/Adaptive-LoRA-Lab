import json
from pathlib import Path
import shutil
import unittest
import uuid

from experiments.protocol_repair import recover_finished_checkpoint, ROOT
from evaluation.benchmark import file_sha256
from training.experiment_state import recipe_signature, REQUIRED_CHECKPOINT_FILES


class ProtocolCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.root = (ROOT/'.cache'/('protocol-checkpoint-'+uuid.uuid4().hex)).resolve()
        assert self.root.is_relative_to(ROOT.resolve())
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree,self.root)
        source = self.root/'source'
        source.mkdir()
        (source/'adapter_model.safetensors').write_bytes(b'original source weights')
        self.folder = self.root/'output'
        self.folder.mkdir()
        self.cfg = {'source_adapter':str(source.relative_to(ROOT)),'max_steps':4,'head_lr':.001}
        self.manifest = {'config':self.cfg,'training_code_sha256':file_sha256(ROOT/'training/repair_protocol.py'),
                         'source_adapter_sha256':file_sha256(source/'adapter_model.safetensors'),
                         'label_policy':'generation-prompt-masked-through-eos-v1'}
        (self.folder/'training_manifest.json').write_text(json.dumps(self.manifest))

    def checkpoint(self,step):
        path = self.folder/f'checkpoint-{step}'
        path.mkdir()
        for name in set(REQUIRED_CHECKPOINT_FILES) | {'tokenizer.json','tokenizer_config.json','chat_template.jinja'}:
            (path/name).write_bytes(b'saved '+name.encode())
        (path/'trainer_state.json').write_text(json.dumps({'global_step':step}))
        (path/'checkpoint_complete.json').write_text(json.dumps({'global_step':step,'recipe_signature':recipe_signature(self.manifest)}))
        return path

    def test_finalization_preserves_saved_weights_and_does_not_add_an_update(self):
        checkpoint = self.checkpoint(4)
        payload = (checkpoint/'adapter_model.safetensors').read_bytes()
        self.assertTrue(recover_finished_checkpoint(self.folder,self.cfg))
        self.assertEqual((self.folder/'adapter_model.safetensors').read_bytes(),payload)
        metrics = json.loads((self.folder/'run_metrics.json').read_text())
        self.assertEqual(metrics['optimizer_steps'],4)
        self.assertIn('finalized_from_checkpoint',metrics)
        self.assertFalse(recover_finished_checkpoint(self.folder,self.cfg))

    def test_partial_training_needs_normal_resume(self):
        self.checkpoint(3)
        self.assertFalse(recover_finished_checkpoint(self.folder,self.cfg))
        self.assertFalse((self.folder/'run_metrics.json').exists())

    def test_changed_learning_rate_is_not_silently_adopted(self):
        self.checkpoint(4)
        with self.assertRaisesRegex(ValueError,'settings changed'):
            recover_finished_checkpoint(self.folder,{**self.cfg,'head_lr':.002})
        self.assertFalse((self.folder/'run_metrics.json').exists())


if __name__ == '__main__':
    unittest.main()
