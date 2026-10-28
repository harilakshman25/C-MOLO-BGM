"""Optional independent semantic evaluator; pin locally, no training or generation."""
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

repo='laion/clap-htsat-unfused'
root=Path('checkpoints/clap-htsat-unfused')
marker=root/'download_manifest.json'
identity=json.loads(marker.read_text()) if marker.exists() else None
if identity and identity['repo_id']!=repo:
    raise ValueError('Wrong checkpoint identity')
info=HfApi().model_info(repo,revision=identity['revision'] if identity else None)
names={s.rfilename for s in info.siblings}
weight='model.safetensors' if 'model.safetensors' in names else 'pytorch_model.bin'
if weight not in names: raise ValueError('Expected single-file weights unavailable')
snapshot_download(repo,revision=info.sha,local_dir=str(root),max_workers=2,
                  allow_patterns=['*.json','*.txt',weight,'README.md','LICENSE*'])
marker.write_text(json.dumps(dict(repo_id=repo,revision=info.sha),indent=2))
print(f'CLAP EVALUATOR READY: {root}')
