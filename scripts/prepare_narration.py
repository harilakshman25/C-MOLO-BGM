"""CPU-only local TTS. Saves measured sentence boundaries and a ready timeline."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from scipy.signal import resample_poly
from transformers import AutoTokenizer, VitsModel
from bgm.audio import write_wav
from bgm.story import level_audio, build_plan


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--story",type=Path,default=Path("examples/forest_cabin.json"))
    p.add_argument("--out",type=Path,default=Path("outputs/forest-narration"))
    p.add_argument("--model",type=Path,default=Path("checkpoints/mms-tts-eng"))
    args=p.parse_args()
    story=json.loads(args.story.read_text())
    args.out.mkdir(parents=True,exist_ok=False)
    status=args.out/"status.json"
    status.write_text(json.dumps(dict(state="running")))
    try:
        torch.set_num_threads(4)
        model=VitsModel.from_pretrained(str(args.model),local_files_only=True).cpu().eval()
        tokenizer=AutoTokenizer.from_pretrained(str(args.model),local_files_only=True)
        sr=44100
        pieces=[np.zeros((sr,2))]
        cursor=sr
        timings=[]
        for i,scene in enumerate(story["segments"]):
            torch.manual_seed(story["seed"]+i)
            with torch.inference_mode():
                mono=model(**tokenizer(scene["text"],return_tensors="pt")).waveform[0].float().numpy()
            original_sr=model.config.sampling_rate
            divisor=math.gcd(sr,original_sr)
            mono=resample_poly(mono,sr//divisor,original_sr//divisor)
            sound=np.repeat(mono[:,None],2,axis=1)
            if len(sound)==0 or not np.isfinite(sound).all():
                raise ValueError("Invalid narration waveform")
            timings.append(dict(index=i,text=scene["text"],start_s=cursor/sr,end_s=(cursor+len(sound))/sr))
            pieces.extend([sound,np.zeros((round(2.5*sr),2))])
            cursor+=len(sound)+round(2.5*sr)
            print(f"Narrated sentence {i+1}/{len(story['segments'])}",flush=True)
        narration,gain=level_audio(np.concatenate(pieces),target_db=-19,max_boost_db=12)
        write_wav(args.out/"narration.wav",narration,sr)
        library=json.loads(Path("examples/sound_library.json").read_text())
        plan=build_plan(story,timings,library,sr)
        (args.out/"timeline.json").write_text(json.dumps(plan,indent=2))
        (args.out/"segments.json").write_text(json.dumps(timings,indent=2))
        (args.out/"tts-report.json").write_text(json.dumps(dict(model=json.loads((args.model/"download_manifest.json").read_text()),
            story_sha256=hashlib.sha256(args.story.read_bytes()).hexdigest(),sample_rate=sr,gain_db=gain,
            method="Sentence synthesis with 2.5-second gaps; polyphase resampling; CPU",seed=story["seed"]),indent=2))
        status.write_text(json.dumps(dict(state="completed")))
        print(f"NARRATION COMPLETE: {args.out}")
    except Exception as exc:
        status.write_text(json.dumps(dict(state="failed",error=str(exc))))
        raise

if __name__=="__main__": main()
