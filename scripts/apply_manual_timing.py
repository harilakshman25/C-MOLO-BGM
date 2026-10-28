"""Apply explicitly supplied human timestamps without modifying the baseline."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
from bgm.timeline import validate


def apply_marks(plan,segments,anchors,marks):
    if set(marks)!=set(anchors): raise ValueError('Provide exactly all six cue IDs')
    result=copy.deepcopy(plan); cues={c['id']:c for c in result['cues']}; changes=[]
    for cid,anchor in anchors.items():
        i=anchor['segment']; t=marks[cid]; segment=segments[i]; cue=cues[cid]
        if isinstance(t,bool) or not isinstance(t,(int,float)) or not math.isfinite(t) or not segment['start_s']<=t<=segment['end_s']:
            raise ValueError(f'{cid}: word end must be inside its narration segment')
        start=t+.08; old=cue['start_s']; duration=cue['end_s']-old
        limit=segments[i+1]['start_s'] if i+1<len(segments) else plan['duration_s']
        end=cue['end_s'] if cue['kind']=='music' else min(start+duration,limit)
        if end-start<.3: raise ValueError(f'{cid}: insufficient space after trigger')
        cue.update(start_s=start,end_s=end,fade_s=min(cue['fade_s'],(end-start)/2))
        changes.append(dict(cue_id=cid,word_end_s=t,old_start_s=old,new_start_s=start,new_end_s=end))
    result['timing_source']='Human-annotated trigger word ends + 80 ms; ambience unchanged'
    return validate(result),changes


def main():
    p=argparse.ArgumentParser();p.add_argument('--marks',type=Path,required=True)
    p.add_argument('--narration',type=Path,default=Path('outputs/forest-narration'))
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    marks=json.loads(args.marks.read_text());digest=hashlib.sha256((args.narration/'narration.wav').read_bytes()).hexdigest()
    if marks.get('method')!='human-listening-word-end' or marks.get('narration_sha256')!=digest:
        raise ValueError('Timing notes do not identify this narration')
    plan,changes=apply_marks(json.loads((args.narration/'timeline.json').read_text()),
         json.loads((args.narration/'segments.json').read_text()),
         json.loads(Path('examples/forest_anchors.json').read_text()),marks['word_end_s'])
    args.out.mkdir(parents=True,exist_ok=False)
    for name in ('narration.wav','segments.json','tts-report.json'):shutil.copyfile(args.narration/name,args.out/name)
    (args.out/'timeline.json').write_text(json.dumps(plan,indent=2))
    (args.out/'alignment-report.json').write_text(json.dumps(dict(method=marks['method'],narration_sha256=digest,changes=changes),indent=2))
    (args.out/'status.json').write_text(json.dumps(dict(state='completed')))
    print(f'MANUAL TIMING COMPLETE: {args.out}')

if __name__=='__main__':main()
