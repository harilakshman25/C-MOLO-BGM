"""Summarize measured runtime and optional human ratings without inventing scores."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import statistics


def average(values):
    return statistics.mean(values) if values else None


def summarize(root):
    report=json.loads((root/'experiment.json').read_text())
    if report['smoke_test_only']:
        raise ValueError('Smoke audio is not evaluation data')
    key={r['sample']:r for r in csv.DictReader((root/'blind-key.csv').open())}
    ratings=list(csv.DictReader((root/'ratings.csv').open()))
    groups=defaultdict(list)
    seen=set()
    for row in ratings:
        identity=(row['sample'],row['cue_index'])
        if identity in seen or row['sample'] not in key:
            raise ValueError('Duplicate or unknown rating')
        seen.add(identity)
        if row['present'].strip() not in ('','0','1') or row['outside_window'].strip() not in ('','0','1'):
            raise ValueError('Presence and leakage must be blank, 0 or 1')
        for field in ('fidelity_1to5','cohesion_1to5','artifacts_1to5'):
            if row[field].strip() and row[field].strip() not in ('1','2','3','4','5'):
                raise ValueError(f'{field} must be an integer from 1 to 5')
        groups[key[row['sample']]['mode']].append(row)
    result={}
    for mode in report['methods']:
        samples=[r for r in report['samples'] if r['mode']==mode]
        rows=groups[mode]
        rated=[r for r in rows if r['present'].strip()]
        present=[r for r in rated if r['present'].strip()=='1']
        onset_errors=[]
        for row in present:
            if row['first_onset_s'].strip():
                onset=float(row['first_onset_s'])
                duration=json.loads((root/'study-spec.json').read_text())['duration_s']
                if not 0 <= onset <= duration:
                    raise ValueError('Onset outside scene')
                onset_errors.append(abs(onset-float(row['target_start_s'])))
        # Scene-level ratings count once per sample, not once per cue.
        scene_scores={}
        for field in ('cohesion_1to5','artifacts_1to5'):
            values={}
            for row in rows:
                if row[field].strip():
                    if row['sample'] in values and values[row['sample']]!=int(row[field]):
                        raise ValueError('Conflicting scene ratings across cue rows')
                    values[row['sample']]=int(row[field])
            scene_scores[field]=dict(n=len(values),mean=average(list(values.values())))
        by_category={}
        for category in sorted({r['category'] for r in rows}):
            cr=[r for r in rated if r['category']==category]
            by_category[category]=dict(n_rated=len(cr),presence_rate=average([int(r['present']) for r in cr]))
        result[mode]=dict(n_samples=len(samples),mean_elapsed_s=average([r['elapsed_s'] for r in samples]),
            max_allocated_gib=max(r['peak_allocated_gib'] for r in samples),
            near_silent_samples=sum(r['near_silent'] for r in samples),
            cues_rated=len(rated),cues_total=len(rows),presence_rate=average([int(r['present']) for r in rated]),
            timed_present_cues=len(onset_errors),mean_absolute_onset_error_s=average(onset_errors),
            leakage_rate_among_rated_present=average([int(r['outside_window']) for r in present if r['outside_window'].strip()]),
            fidelity_mean=average([int(r['fidelity_1to5']) for r in present if r['fidelity_1to5'].strip()]),
            scene_scores=scene_scores,by_category=by_category)
    return dict(methods=result,interpretation='Null means unmeasured. Ratings are descriptive; seeds are repeated observations, not independent listeners. Missing sounds excluded from onset error, counted in presence.',
                quality_claim='No winner inferred from runtime, loudness, or unfilled ratings.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--study',type=Path,required=True)
    args=p.parse_args()
    result=summarize(args.study)
    (args.study/'analysis.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
