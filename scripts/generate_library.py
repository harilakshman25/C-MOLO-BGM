"""Generate two selectable candidates per category, saving each immediately."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from bgm.backends import StableAudioBackend, cue_seed
from bgm.lora import set_strength
from bgm.audio import write_wav
from bgm.story import level_audio


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--adapter",type=Path,required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    status=args.out/"status.json"
    status.write_text(json.dumps(dict(state="running")))
    try:
        if torch.cuda.device_count()!=1: raise RuntimeError("Use one allocated GPU")
        bank=json.loads(Path("examples/sound_library.json").read_text())
        backend=StableAudioBackend("checkpoints/stable-audio-open-1.0",adapter_path=args.adapter)
        report=dict(model=backend.adapter_metadata["model"],dog_adapter=backend.adapter_metadata,
                    adapter_sha256=hashlib.sha256(args.adapter.read_bytes()).hexdigest(),steps=100,
                    guidance_scale=7.0,candidates=[],review_status="Pending listening; candidate 0 selected provisionally")
        selection={}
        for category,spec in bank.items():
            set_strength(backend.pipe.transformer,1 if category=="dog" else 0)
            for candidate in (0,1):
                seed=cue_seed(42+candidate,category)
                cue=dict(start_s=0,end_s=spec["seconds"],prompt=spec["prompt"])
                print(f"Generating {category} candidate {candidate}",flush=True)
                torch.manual_seed(seed)
                sound=backend.generate(cue,44100,seed)
                sound,gain=level_audio(sound,target_db=-22)
                name=f"{category}-{candidate}.wav"
                write_wav(args.out/name,sound,44100)
                report["candidates"].append(dict(category=category,candidate=candidate,file=name,seed=seed,
                    prompt=spec["prompt"],gain_db=gain,sha256=hashlib.sha256((args.out/name).read_bytes()).hexdigest()))
                (args.out/"library-report.json").write_text(json.dumps(report,indent=2))
            selection[category]=f"{category}-0.wav"
        (args.out/"selection.json").write_text(json.dumps(selection,indent=2))
        status.write_text(json.dumps(dict(state="completed")))
        print("SOUND LIBRARY COMPLETE",flush=True)
    except Exception as exc:
        status.write_text(json.dumps(dict(state="failed",error=str(exc))))
        raise

if __name__=="__main__": main()
