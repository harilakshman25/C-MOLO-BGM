"""Pin a local Whisper English checkpoint for CPU word-timestamp estimation."""
import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

p=argparse.ArgumentParser()
p.add_argument('--size',choices=('base','small'),default='base')
args=p.parse_args()
root=Path(f'checkpoints/whisper-{args.size}-en')
marker=root/'download_manifest.json'
repo=f'openai/whisper-{args.size}.en'
identity=json.loads(marker.read_text()) if marker.exists() else None
if identity and identity['repo_id']!=repo:
    raise ValueError('Existing checkpoint directory belongs to a different model')
revision=identity['revision'] if identity else HfApi().model_info(repo).sha
snapshot_download(repo,revision=revision,local_dir=str(root),max_workers=2,
                  allow_patterns=['*.json','*.txt','*.safetensors','README.md','LICENSE*'])
marker.write_text(json.dumps(dict(repo_id=repo,revision=revision),indent=2))
print(f'ALIGNER DOWNLOAD COMPLETE: {root}')
