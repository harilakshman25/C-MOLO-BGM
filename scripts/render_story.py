"""Mix existing narration and selected sound library; no model needed."""
import argparse
import json
from pathlib import Path
from bgm.audio import read_wav
from bgm.service import render
from bgm.story import fit_stem
from bgm.timeline import validate


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--narration",type=Path,required=True)
    p.add_argument("--library",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    plan=validate(json.loads((args.narration/"timeline.json").read_text()))
    if json.loads((args.narration/"status.json").read_text())["state"]!="completed":
        raise ValueError("Narration run did not complete")
    selection=json.loads((args.library/"selection.json").read_text())
    class LibraryBackend:
        name="separate-stem-story-demo-NOT-latent-fusion"
        def generate(self,cue,sample_rate,seed):
            filename=selection[cue["category"]]
            if Path(filename).name!=filename or not filename.endswith('.wav'):
                raise ValueError("Selection must name local WAV files")
            sound=read_wav(args.library/filename,sample_rate)
            count=round(cue["end_s"]*sample_rate)-round(cue["start_s"]*sample_rate)
            return fit_stem(sound,count,sample_rate,loop=cue["kind"] in ("music","ambience"))
    report=render(plan,args.out,backend=LibraryBackend(),narration_path=args.narration/"narration.wav")
    (args.out/"selection.json").write_text(json.dumps(selection,indent=2))
    print(json.dumps(report,indent=2))
    print(f"STORY DEMO COMPLETE: {args.out/'mix.wav'}")

if __name__=="__main__": main()
