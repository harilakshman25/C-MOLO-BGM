"""Synthesize the study story on CPU into fixed scene slots, preserving pauses."""
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
from bgm.story import level_audio


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--spec',type=Path,default=Path('examples/composition_study.json'))
    p.add_argument('--model',type=Path,default=Path('checkpoints/mms-tts-eng'))
    p.add_argument('--out',type=Path,default=Path('outputs/study-narration'))
    args=p.parse_args()
    spec=json.loads(args.spec.read_text())
    args.out.mkdir(parents=True,exist_ok=False)
    status=args.out/'status.json'
    status.write_text(json.dumps(dict(state='running')))
    try:
        torch.set_num_threads(4)
        model=VitsModel.from_pretrained(str(args.model),local_files_only=True).cpu().eval()
        tokenizer=AutoTokenizer.from_pretrained(str(args.model),local_files_only=True)
        sr=44100
        output=np.zeros((round(len(spec['scenes'])*spec['duration_s']*sr),2))
        segments=[]
        for i, scene in enumerate(spec['scenes']):
            torch.manual_seed(42+i)
            with torch.inference_mode():
                mono=model(**tokenizer(scene['text'],return_tensors='pt')).waveform[0].float().numpy()
            original_sr=model.config.sampling_rate
            div=math.gcd(sr,original_sr)
            mono=resample_poly(mono,sr//div,original_sr//div)
            if len(mono)/sr > spec['duration_s']-2:
                raise ValueError(f'{scene["id"]} narration exceeds slot; shorten text, never truncate speech')
            start=round((i*spec['duration_s']+1)*sr)
            output[start:start+len(mono)]=np.repeat(mono[:,None],2,axis=1)
            segments.append(dict(scene=scene['id'],start_s=start/sr,end_s=(start+len(mono))/sr,text=scene['text']))
            print(f'Narrated {scene["id"]}: {len(mono)/sr:.2f}s',flush=True)
        output,gain=level_audio(output,target_db=-19,max_boost_db=12)
        write_wav(args.out/'narration.wav',output,sr)
        (args.out/'segments.json').write_text(json.dumps(segments,indent=2))
        (args.out/'report.json').write_text(json.dumps(dict(spec_sha256=hashlib.sha256(args.spec.read_bytes()).hexdigest(),
            model=json.loads((args.model/'download_manifest.json').read_text()),gain_db=gain,
            timing='Authored 20-second scene slots. Cues are scene-relative, not aligned to individual words.'),indent=2))
        status.write_text(json.dumps(dict(state='completed')))
    except Exception as exc:
        status.write_text(json.dumps(dict(state='failed',error=str(exc))))
        raise


if __name__=='__main__': main()
