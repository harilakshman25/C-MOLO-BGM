import csv
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import wave
from scripts.compare_prompts import cases, SEEDS
from scripts import prepare_esc50_dog


class ExperimentTests(unittest.TestCase):
    def test_only_prompt_changes_within_seed_pair(self):
        template = json.loads((Path(__file__).resolve().parents[1]/"examples/dog-baseline.json").read_text())
        plans = list(cases(template))
        self.assertEqual(len(plans), 6)
        self.assertEqual({row[1] for row in plans}, set(SEEDS))
        for index in range(0, 6, 2):
            outdoor, isolated = plans[index][2], plans[index+1][2]
            self.assertNotEqual(outdoor["cues"][0]["prompt"], isolated["cues"][0]["prompt"])
            isolated["cues"][0]["prompt"] = outdoor["cues"][0]["prompt"]
            self.assertEqual(outdoor, isolated)
        self.assertEqual(template["seed"], 42)

    def test_dataset_splits_revision_and_review_preserved(self):
        metadata = io.StringIO()
        writer = csv.writer(metadata)
        writer.writerow(["filename", "fold", "category", "src_file"])
        for i in range(40):
            writer.writerow([f"clip-{i}.wav", i//8+1, "dog", str(i)])
        def fetch(url):
            if "api.github.com" in url:
                return json.dumps({"sha": "a"*40}).encode()
            if url.endswith("esc50.csv"):
                return metadata.getvalue().encode()
            if url.endswith(".wav"):
                i = int(url.rsplit("-", 1)[1].split(".")[0])
                buffer = io.BytesIO()
                with wave.open(buffer, "wb") as handle:
                    handle.setnchannels(1)
                    handle.setsampwidth(2)
                    handle.setframerate(44100)
                    handle.writeframes(struct.pack("<h", i)*100)
                return buffer.getvalue()
            return b"Test fixture license/readme"
        with tempfile.TemporaryDirectory() as folder:
            with patch("sys.argv", ["prepare", "--out", folder]), patch.object(prepare_esc50_dog, "fetch", side_effect=fetch), patch("builtins.print"):
                prepare_esc50_dog.main()
                root = Path(folder)
                report = json.loads((root/"audit.json").read_text())
                self.assertEqual(report["counts"], {"dog:train": 24, "dog:validation": 8, "dog:test": 8})
                (root/"review.csv").write_text("my existing review")
                prepare_esc50_dog.main()
                self.assertEqual((root/"review.csv").read_text(), "my existing review")


if __name__ == "__main__":
    unittest.main()
