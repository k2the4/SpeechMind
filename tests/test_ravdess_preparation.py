import unittest
import numpy as np
from scripts.train_voice_emotion import (
    BASE_MODEL_NAME, LABELS, assign_speaker_splits,
    parse_ravdess_filename, parse_crema_d_filename, summarize,
)
from models.voice_features import POOLING_VERSION, EXPECTED_FEATURE_DIMENSION, pool_hidden_states


class MultiCorpusPreparationTests(unittest.TestCase):
    def test_training_backbone_is_wavlm(self):
        self.assertEqual(BASE_MODEL_NAME,"microsoft/wavlm-base-plus")

    def test_richer_feature_pooling(self):
        self.assertIn("q10_q90",POOLING_VERSION)
        hidden=np.asarray([[1.,2.],[3.,4.],[5.,6.]],dtype=np.float32)
        pooled=pool_hidden_states(hidden)
        self.assertEqual(pooled.shape,(8,))
        np.testing.assert_allclose(pooled[:2],[3.,4.])
        np.testing.assert_allclose(pooled[2:4],np.std(hidden,axis=0))
        self.assertEqual(EXPECTED_FEATURE_DIMENSION,3072)

    def test_parse_ravdess_including_surprise(self):
        row=parse_ravdess_filename("03-01-08-01-02-01-12.wav")
        self.assertEqual(row["dataset"],"ravdess")
        self.assertEqual(row["label"],"surprise")
        self.assertEqual(row["speaker_id"],"ravdess_actor_12")

    def test_parse_crema_d(self):
        row=parse_crema_d_filename("1001_DFA_ANG_XX.wav")
        self.assertEqual(row["dataset"],"crema_d")
        self.assertEqual(row["label"],"anger")
        self.assertEqual(row["speaker_id"],"crema_d_actor_1001")

    def test_invalid_dataset_files_ignored(self):
        self.assertIsNone(parse_ravdess_filename("sample.wav"))
        self.assertIsNone(parse_crema_d_filename("1001_DFA_SUR_XX.wav"))

    def test_speakers_are_disjoint_per_corpus(self):
        rows=[]
        for dataset,prefix in (("ravdess","r"),("crema_d","c")):
            for actor in range(1,26):
                rows.append({"path":f"{prefix}-{actor}","dataset":dataset,
                    "speaker_id":f"{prefix}_{actor}","label":LABELS[actor%len(LABELS)]})
        split=assign_speaker_splits(rows,seed=7)
        for dataset in ("ravdess","crema_d"):
            groups={name:{r["speaker_id"] for r in split if r["dataset"]==dataset and r["split"]==name}
                    for name in ("train","validation","test")}
            self.assertFalse(groups["train"]&groups["validation"])
            self.assertFalse(groups["train"]&groups["test"])
            self.assertFalse(groups["validation"]&groups["test"])

    def test_summary_shows_datasets(self):
        summary=summarize([
            {"dataset":"ravdess","speaker_id":"r1","label":"joy","split":"train"},
            {"dataset":"crema_d","speaker_id":"c1","label":"anger","split":"test"},
        ])
        self.assertEqual(summary["train"]["datasets"],{"ravdess":1})
        self.assertEqual(summary["test"]["datasets"],{"crema_d":1})


if __name__=="__main__":
    unittest.main()
