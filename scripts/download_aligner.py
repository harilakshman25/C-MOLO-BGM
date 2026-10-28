"""Pin a local Whisper English checkpoint for CPU word-timestamp estimation."""
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

root=Path('checkpoints/whisper-base-en')
marker=root/'download_manifest.json'
repo='openai/whisper-base.en'
revision=json.loads(marker.read_text())['revision'] if marker.exists() else HfApi().model_info(repo).sha
snapshot_download(repo,revision=revision,local_dir=str(root),max_workers=2,
                  allow_patterns=['*.json','*.txt','*.safetensors','README.md','LICENSE*'])
marker.write_text(json.dumps(dict(repo_id=repo,revision=revision),indent=2))
print(f'ALIGNER DOWNLOAD COMPLETE: {root}')
