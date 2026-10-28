"""CPU: attach identical narration to all four soundscape methods."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from bgm.audio import read_wav, write_wav, duck_envelope


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--narration',type=Path,default=Path('outputs/study-narration'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    report=json.loads((args.study/'experiment.json').read_text())
    if report['smoke_test_only'] or json.loads((args.study/'status.json').read_text())['state']!='completed':
        raise ValueError('A completed non-smoke study is required')
    if json.loads((args.narration/'status.json').read_text())['state']!='completed':
        raise ValueError('Narration incomplete')
    nr=json.loads((args.narration/'report.json').read_text())
    if nr['spec_sha256']!=report['spec_sha256']:
        raise ValueError('Narration and study specifications differ')
    spec=json.loads((args.study/'study-spec.json').read_text())
    sr=report['sample_rate']
    voice=read_wav(args.narration/'narration.wav',sr)
    samples=round(spec['duration_s']*sr)
    if len(voice)!=samples*len(spec['scenes']):
        raise ValueError('Narration length differs from complete story')
    mixes={}
    for mode in report['methods']:
        parts=[]
        for scene in spec['scenes']:
            rows=[r for r in report['samples'] if r['mode']==mode and r['seed']==args.seed and r['scene']==scene['id']]
            if len(rows)!=1:
                raise ValueError(f'Missing or duplicate scene {scene["id"]} {mode}; run full study first')
            path=args.study/rows[0]['listen_file']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=rows[0]['listen_sha256']:
                raise ValueError('Study audio hash mismatch')
            audio=read_wav(path,sr)
            if len(audio)!=samples: raise ValueError('Wrong scene duration')
            # Same 150ms boundary fades in every arm; no overlapping story scenes.
            fade=round(.15*sr)
            ramp=np.linspace(0,1,fade)[:,None]
            audio[:fade]*=ramp; audio[-fade:]*=ramp[::-1]
            parts.append(audio)
        bed=np.concatenate(parts)*10**(-3/20)
        mixes[mode]=voice+bed*duck_envelope(voice,sr,-8)[:,None]
    gain=min(1.,.89/max(float(np.max(np.abs(a))) for a in mixes.values()))
    args.out.mkdir(parents=True,exist_ok=False)
    hashes={}
    for mode,audio in mixes.items():
        path=args.out/f'{mode}-story.wav'
        write_wav(path,audio*gain,sr)
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    (args.out/'report.json').write_text(json.dumps(dict(seed=args.seed,duration_s=len(voice)/sr,
        common_master_gain=gain,sha256=hashes,
        narration_sha256=hashlib.sha256((args.narration/'narration.wav').read_bytes()).hexdigest(),
        method='Ten scene generations concatenated with boundary fades; same separately generated narration in all arms.',
        limitation='Not a single 200-second diffusion trajectory. Narration is mixed after soundscape generation.'),indent=2))
    print(f'FOUR STORY MIXES COMPLETE: {args.out}')


if __name__=='__main__': main()
