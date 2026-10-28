"""Download English MMS TTS, pinning the resolved revision for offline inference."""
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

root = Path("checkpoints/mms-tts-eng")
marker = root/"download_manifest.json"
repo = "facebook/mms-tts-eng"
revision = json.loads(marker.read_text())["revision"] if marker.exists() else HfApi().model_info(repo).sha
snapshot_download(repo, revision=revision, local_dir=str(root), max_workers=2,
                  allow_patterns=["*.json", "*.safetensors", "pytorch_model.bin", "*.txt", "README.md", "LICENSE*"])
marker.write_text(json.dumps(dict(repo_id=repo, revision=revision, license="CC-BY-NC-4.0"), indent=2))
print(f"TTS DOWNLOAD COMPLETE: {root}")
