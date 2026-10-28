"""Optional windowed CLAP cosine analysis. Scores are NOT detection probabilities."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from scipy.signal import resample_poly
from transformers import ClapModel, ClapProcessor
from bgm.audio import read_wav
from bgm.study import DESCRIPTIONS


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--model',type=Path,default=Path('checkpoints/clap-htsat-unfused'))
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    args=p.parse_args()
    report=json.loads((args.study/'experiment.json').read_text())
    if report['smoke_test_only'] or json.loads((args.study/'status.json').read_text())['state']!='completed':
        raise ValueError('Completed non-smoke study required')
    spec=json.loads((args.study/'study-spec.json').read_text())
    scenes={s['id']:s for s in spec['scenes']}
    torch.set_num_threads(4); torch.manual_seed(0); np.random.seed(0)
    model=ClapModel.from_pretrained(str(args.model),local_files_only=True).to(args.device).eval()
    processor=ClapProcessor.from_pretrained(str(args.model),local_files_only=True)
    categories=list(DESCRIPTIONS)
    inputs=processor(text=['The sound of '+DESCRIPTIONS[c]+'.' for c in categories],padding=True,return_tensors='pt').to(args.device)
    with torch.inference_mode():
        text=model.get_text_features(**inputs)
        text=torch.nn.functional.normalize(text,dim=-1)
    rows=[]
    grouped=defaultdict(list)
    sr=report['sample_rate']
    for sample in report['samples']:
        path=args.study/sample['listen_file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=sample['listen_sha256']:
            raise ValueError('Audio hash mismatch')
        wave=read_wav(path,sr).mean(axis=1)
        # Whole-recording resampling before fixed 5s crops; no random long-clip truncation.
        wave=resample_poly(wave,160,147).astype(np.float32) if sr==44100 else wave
        if sr!=44100: raise ValueError('Expected 44.1kHz study')
        times=np.arange(0,spec['duration_s']-5+1e-6,2.5)
        windows=[wave[round(t*48000):round((t+5)*48000)] for t in times]
        scores=[]
        for begin in range(0,len(windows),4):
            inputs=processor(audios=windows[begin:begin+4],sampling_rate=48000,return_tensors='pt').to(args.device)
            with torch.inference_mode():
                emb=model.get_audio_features(**inputs)
                emb=torch.nn.functional.normalize(emb,dim=-1)
                scores.extend((emb@text.T).cpu().numpy())
        for c_index,category in enumerate(categories):
            targets=[(a,b) for c,a,b in scenes[sample['scene']]['cues'] if c==category]
            inside,outside=[],[]
            for t,values in zip(times,scores):
                # Union coverage handles repeated events without double-counting.
                intervals=sorted((max(t,a),min(t+5,b)) for a,b in targets if a<t+5 and b>t)
                covered=0.; end=t
                for a,b in intervals:
                    covered+=max(0.,b-max(a,end)); end=max(end,b)
                fraction=covered/5
                score=float(values[c_index])
                rows.append(dict(scene=sample['scene'],seed=sample['seed'],mode=sample['mode'],category=category,
                                 start_s=float(t),end_s=float(t+5),target_fraction=fraction,cosine=score))
                if fraction>=.8: inside.append(score)
                if fraction<=.2: outside.append(score)
            if targets:
                grouped[(sample['mode'],category)].append(dict(scene=sample['scene'],seed=sample['seed'],
                    in_target_mean=float(np.mean(inside)) if inside else None,
                    off_target_mean=float(np.mean(outside)) if outside else None,
                    contrast=float(np.mean(inside)-np.mean(outside)) if inside and outside else None))
        print(f'Scored {sample["scene"]} {sample["seed"]} {sample["mode"]}',flush=True)
    with (args.study/'clap-windows.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary=[]
    for (mode,category),values in grouped.items():
        item=dict(mode=mode,category=category,n_samples=len(values),samples=values)
        for field in ('in_target_mean','off_target_mean','contrast'):
            numbers=[v[field] for v in values if v[field] is not None]
            item[field]=float(np.mean(numbers)) if numbers else None
            item[field+'_n']=len(numbers)
        summary.append(item)
    (args.study/'clap-analysis.json').write_text(json.dumps(dict(model=json.loads((args.model/'download_manifest.json').read_text()),
        window_s=5,hop_s=2.5,results=summary,
        limitation='Uncalibrated semantic cosine, not event accuracy, quality, or precise timing. Correlated categories and overlapping windows; no threshold or significance claim.'),indent=2))
    print('CLAP ANALYSIS COMPLETE')


if __name__=='__main__': main()
