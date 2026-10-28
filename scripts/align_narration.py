"""CPU word timestamps on existing speech; preserve original narration and timeline."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import numpy as np
import torch
from scipy.signal import resample_poly
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
from bgm.audio import read_wav
from bgm.alignment import align_cues


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--narration',type=Path,default=Path('outputs/forest-narration'))
    p.add_argument('--out',type=Path,default=Path('outputs/forest-aligned'))
    p.add_argument('--model',type=Path,default=Path('checkpoints/whisper-base-en'))
    p.add_argument('--anchors',type=Path,default=Path('examples/forest_anchors.json'))
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    status=args.out/'status.json'; status.write_text(json.dumps(dict(state='running')))
    try:
        torch.set_num_threads(4)
        plan=json.loads((args.narration/'timeline.json').read_text())
        segments=json.loads((args.narration/'segments.json').read_text())
        sound=read_wav(args.narration/'narration.wav',plan['sample_rate']).mean(axis=1)
        model=AutoModelForSpeechSeq2Seq.from_pretrained(str(args.model),local_files_only=True,
                    torch_dtype=torch.float32,attn_implementation='eager').cpu().eval()
        processor=AutoProcessor.from_pretrained(str(args.model),local_files_only=True)
        asr=pipeline('automatic-speech-recognition',model=model,tokenizer=processor.tokenizer,
                     feature_extractor=processor.feature_extractor,device=-1)
        transcripts=[]
        sr=plan['sample_rate']; divisor=math.gcd(sr,16000)
        for i,segment in enumerate(segments):
            clip=sound[round(segment['start_s']*sr):round(segment['end_s']*sr)]
            if len(clip)/sr>=30: raise ValueError('Sentence too long for this alignment pilot')
            audio=resample_poly(clip,16000//divisor,sr//divisor).astype(np.float32)
            result=asr({'raw':audio,'sampling_rate':16000},return_timestamps='word',generate_kwargs={'do_sample':False})
            transcripts.append(result)
            (args.out/'recognized-words.json').write_text(json.dumps(transcripts,indent=2))
            print(f"Aligned sentence {i+1}: {result['text']}",flush=True)
        revised,changes=align_cues(plan,segments,transcripts,json.loads(args.anchors.read_text()))
        (args.out/'timeline.json').write_text(json.dumps(revised,indent=2))
        for name in ('narration.wav','segments.json','tts-report.json'):
            shutil.copyfile(args.narration/name,args.out/name)
        (args.out/'alignment-report.json').write_text(json.dumps(dict(
            model=json.loads((args.model/'download_manifest.json').read_text()),
            narration_sha256=hashlib.sha256((args.out/'narration.wav').read_bytes()).hexdigest(),
            changes=changes,note='ASR-estimated timing; not forced alignment. Source-stem lead-in can delay audible onset.'),indent=2))
        status.write_text(json.dumps(dict(state='completed')))
        print(f'ALIGNMENT COMPLETE: {args.out}')
    except Exception as exc:
        status.write_text(json.dumps(dict(state='failed',error=str(exc))))
        raise

if __name__=='__main__': main()
