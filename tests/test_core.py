import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from bgm.timeline import validate, draft
from bgm.fusion import fuse, temporal_masks, sample_fused
from bgm.audio import mix, read_wav, write_wav
from bgm.backends import DiagnosticBackend
from bgm.service import render
from bgm.dataset import audit


def example():
    return json.loads((Path(__file__).resolve().parents[1]/"examples/timeline.json").read_text())


class TimelineTests(unittest.TestCase):
    def test_overlap_accepted(self):
        self.assertEqual(len(validate(example())["cues"]), 3)

    def test_invalid_boundaries_and_nan(self):
        for key, value in (("start_s", -1), ("end_s", 11), ("end_s", 0), ("gain_db", float("nan"))):
            plan = example()
            plan["cues"][0][key] = value
            with self.assertRaises(ValueError):
                validate(plan)

    def test_unsafe_duplicate_ids(self):
        for bad in ("../file", "rain_01"):
            plan = example()
            plan["cues"][1]["id"] = bad
            with self.assertRaises(ValueError):
                validate(plan)

    def test_draft_explicitly_estimated(self):
        plan = draft("rain then a dog and piano", 10)
        self.assertIn("estimate", plan["timing_source"])
        self.assertEqual({c["category"] for c in plan["cues"]}, {"rain", "dog", "piano"})


class FusionTests(unittest.TestCase):
    def test_overlap_gaps_and_single_expert(self):
        base = np.full((2, 4), 10.)
        experts = np.stack([np.full_like(base, 2), np.full_like(base, 6)])
        masks = np.array([[0, 1, 1, 0.5], [0, 0, 1, 0]])
        np.testing.assert_allclose(fuse(base, experts, masks), [[10, 2, 4, 6]]*2)

    def test_identity_for_identical_predictions(self):
        base = np.arange(24).reshape(2, 3, 4)
        for mode in ("normalized", "residual"):
            np.testing.assert_allclose(fuse(base, np.stack([base]*3), np.ones((3, 4)), mode), base)

    def test_empty_expert_bank_keeps_base(self):
        base = np.ones((2, 5))
        np.testing.assert_equal(fuse(base, np.empty((0, 2, 5)), np.empty((0, 5))), base)

    def test_padding_masks_are_zero(self):
        masks = temporal_masks([{"start_s": 0, "end_s": 2, "fade_s": 0}], 500, 0.02)
        self.assertEqual(int(masks.sum()), 100)
        self.assertTrue((masks[:, 100:] == 0).all())

    def test_model_config_frame_count_types(self):
        cues = [{"start_s": 1, "end_s": 3, "fade_s": 0.25}]
        expected = temporal_masks(cues, 1024, 2048/44100)
        for frames in (1024.0, np.int64(1024), np.float64(1024)):
            np.testing.assert_array_equal(temporal_masks(cues, frames, 2048/44100), expected)
        for frames in (True, np.bool_(True), 1024.5, float("nan"), float("inf"), 0, -1, "1024"):
            with self.assertRaises(ValueError):
                temporal_masks(cues, frames, 2048/44100)

    def test_soft_edges_and_bad_shape(self):
        masks = temporal_masks([{"start_s": 1, "end_s": 3, "fade_s": 0.5}], 40, 0.1)
        self.assertTrue(0 < masks[0, 10] < masks[0, 15])
        with self.assertRaises(ValueError):
            fuse(np.zeros((2, 4)), np.zeros((2, 2, 4)), np.zeros((4, 2)))

    def test_one_update_and_shared_latent(self):
        calls, seen = [], []
        def base(z, t):
            seen.append((t, z.copy()))
            return np.zeros_like(z)
        def expert(i, z, t):
            np.testing.assert_equal(z, seen[-1][1])
            return np.full_like(z, i+1)
        def step(prediction, t, z):
            calls.append(t)
            return z-prediction
        result = sample_fused(np.zeros((1, 4)), [2, 1, 0], base, expert, step, np.ones((2, 4)))
        self.assertEqual(calls, [2, 1, 0])
        np.testing.assert_allclose(result, -4.5)


class AudioTests(unittest.TestCase):
    def test_placement_and_headroom(self):
        plan = example()
        plan["duration_s"] = 2
        plan["cues"] = [dict(plan["cues"][1], start_s=1, end_s=2, fade_s=0, gain_db=12)]
        output, report = mix(plan, {"dog_01": np.ones((24000, 2))})
        self.assertTrue((output[:24000] == 0).all())
        self.assertLessEqual(report["sample_peak_after_gain"], 10**(-1/20)+1e-10)

    def test_ducking_reduces_bed(self):
        plan = example()
        plan["cues"] = [plan["cues"][0]]
        stem = np.full((240000, 2), 0.1)
        narration = np.full_like(stem, 0.03)
        dry, _ = mix(plan, {"rain_01": stem})
        ducked, _ = mix(plan, {"rain_01": stem}, narration)
        self.assertLess(np.mean(ducked[24000:48000]-narration[24000:48000]), np.mean(dry[24000:48000])*0.5)

    def test_narration_and_short_stem_rejected(self):
        plan = example()
        stems = {c["id"]: np.zeros((1, 2)) for c in plan["cues"]}
        with self.assertRaises(ValueError):
            mix(plan, stems)
        with self.assertRaises(ValueError):
            mix(plan, stems, np.zeros((240001, 2)))

    def test_run_roundtrip_determinism_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder)/"a", Path(folder)/"b"
            first = render(example(), a, backend=DiagnosticBackend())
            second = render(example(), b, backend=DiagnosticBackend())
            self.assertEqual(first["sha256"], second["sha256"])
            self.assertEqual(read_wav(a/"mix.wav", 24000).shape, (240000, 2))
            with self.assertRaises(FileExistsError):
                render(example(), a, backend=DiagnosticBackend())

    def test_failed_run_has_status(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)/"failed"
            with self.assertRaises(FileNotFoundError):
                render(example(), out, stems_dir=Path(folder)/"missing")
            self.assertEqual(json.loads((out/"status.json").read_text())["state"], "failed")


class DatasetTests(unittest.TestCase):
    def test_source_and_duplicate_leakage(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_wav(root/"a.wav", np.zeros((100, 2)), 24000)
            write_wav(root/"b.wav", np.ones((100, 2))*0.1, 24000)
            first = dict(id="a", path="a.wav", source_id="video1", category="dog", caption="Dog bark", license="CC0", split="train")
            second = dict(first, id="b", path="b.wav", split="test")
            manifest = root/"manifest.jsonl"
            def save(rows):
                manifest.write_text("\n".join(json.dumps(r) for r in rows))
            save([first, second])
            with self.assertRaisesRegex(ValueError, "leakage"):
                audit(manifest)
            second.update(source_id="video2", path="a.wav")
            save([first, second])
            with self.assertRaisesRegex(ValueError, "leakage"):
                audit(manifest)
            second["path"] = "b.wav"
            save([first, second])
            self.assertEqual(audit(manifest)["clips"], 2)


if __name__ == "__main__":
    unittest.main()
