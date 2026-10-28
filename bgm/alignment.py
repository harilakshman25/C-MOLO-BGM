"""Conservative trigger matching; ASR estimates are not forced-alignment truth."""
import copy
import math
import re
from difflib import SequenceMatcher
from .timeline import validate


def words(text):
    return re.findall(r"[a-z]+",text.lower())


def align_cues(plan, segments, transcripts, anchors, delay=.08):
    if not math.isfinite(delay) or not 0<=delay<=1:
        raise ValueError('Delay must be within zero to one second')
    revised=copy.deepcopy(plan)
    by_id={c['id']:c for c in revised['cues']}
    changes=[]
    for cid,anchor in anchors.items():
        cue=by_id[cid]
        i=anchor['segment']; segment=segments[i]; prediction=transcripts[i]
        similarity=SequenceMatcher(None,words(segment['text']),words(prediction['text']),autojunk=False).ratio()
        if similarity<.75:
            raise ValueError(f'{cid}: transcript mismatch ({similarity:.2f}); review recognized text')
        target=words(anchor['word'])
        hits=[x for x in prediction['chunks'] if words(x['text'])==target]
        if len(hits)!=1:
            raise ValueError(f'{cid}: expected one exact trigger {target}; found {len(hits)}')
        times=hits[0]['timestamp']
        if len(times)!=2 or any(x is None or not math.isfinite(x) for x in times):
            raise ValueError(f'{cid}: missing word timestamp')
        if not 0<=times[0]<times[1]<=segment['end_s']-segment['start_s']+.1:
            raise ValueError(f'{cid}: word timestamp outside segment')
        start=segment['start_s']+times[1]+delay
        old_start,old_end=cue['start_s'],cue['end_s']
        # Preserve music's ending; let a discrete effect use its original duration,
        # capped before the next narration scene.
        limit=segments[i+1]['start_s'] if i+1<len(segments) else plan['duration_s']
        end=old_end if cue['kind']=='music' else min(start+old_end-old_start,limit)
        if end-start<.3:
            raise ValueError(f'{cid}: insufficient room after trigger; revise cue manually')
        cue.update(start_s=start,end_s=end,fade_s=min(cue['fade_s'],(end-start)/2))
        changes.append(dict(cue_id=cid,trigger=anchor['word'],old_start_s=old_start,new_start_s=start,
                            new_end_s=end,word_end_s=segment['start_s']+times[1],transcript_similarity=similarity))
    revised['timing_source']='Whisper estimated word endpoints + 80 ms; authored triggers; ambience unchanged; not ground-truth forced alignment'
    return validate(revised),changes
