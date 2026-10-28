"""Four-arm, paired-seed study. Run on one allocated GPU; no downloads."""
import argparse
from contextlib import nullcontext
import csv
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from bgm.audio import write_wav
from bgm.backends import StableAudioBackend, cue_seed
from bgm.lora import set_strength
from bgm.window_composition import conditioning, window_consensus
from bgm.study import DESCRIPTIONS

MODES = ('stems', 'joint', 'window', 'window_lora')


def prompt(categories):
    return ', together with '.join(DESCRIPTIONS[c] for c in dict.fromkeys(categories)) + '. Clear detailed audio, no speech or vocals.'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--spec', type=Path, default=Path('examples/composition_study.json'))
    p.add_argument('--model', type=Path, default=Path('checkpoints/stable-audio-open-1.0'))
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--scenes', nargs='+', help='Scene IDs; omit for all ten')
    p.add_argument('--seeds', nargs='+', type=int)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--smoke', action='store_true', help='Two steps; engineering only, never quality evidence')
    args = p.parse_args()
    spec = json.loads(args.spec.read_text())
    duration = spec['duration_s']
    scenes = [s for s in spec['scenes'] if not args.scenes or s['id'] in args.scenes]
    if not scenes or (args.scenes and set(args.scenes)-{s['id'] for s in scenes}):
        raise ValueError('Unknown scene IDs')
    if not 1 <= duration <= 47 or args.steps < 2:
        raise ValueError('Invalid duration or steps')
    for scene in scenes:
        for category, start, end in scene['cues']:
            if category not in DESCRIPTIONS or not 0 <= start < end <= duration:
                raise ValueError('Invalid cue')
    seeds = args.seeds or spec['seeds']
    steps = 2 if args.smoke else args.steps
    args.out.mkdir(parents=True, exist_ok=False)
    status = args.out/'status.json'
    status.write_text(json.dumps(dict(state='running')))
    (args.out/'study-spec.json').write_text(json.dumps(spec, indent=2))
    records, key, ratings = [], [], []
    report = dict(smoke_test_only=args.smoke, steps=steps, guidance_scale=7., seeds=seeds,
                  spec_sha256=digest(args.spec), adapter_sha256=digest(args.adapter),
                  methods=list(MODES), local_weight=spec['local_weight'], samples=records,
                  timing='Authored scene-relative cue windows; no word alignment',
                  limitation='Local cropping is untrained/out-of-distribution; no timing guarantee. Only dog has LoRA.')
    started = time.perf_counter()
    try:
        import diffusers
        if diffusers.__version__ != '0.32.2' or torch.cuda.device_count() != 1:
            raise RuntimeError('Use diffusers 0.32.2 and one allocated GPU')
        backend = StableAudioBackend(args.model, adapter_path=args.adapter)
        pipe = backend.pipe
        if pipe.scheduler.config.prediction_type != 'v_prediction':
            raise RuntimeError('Expected v prediction')
        sr = pipe.vae.sampling_rate
        spacing = pipe.vae.hop_length/sr
        report.update(sample_rate=sr, frame_seconds=spacing, torch=torch.__version__,
                      diffusers=diffusers.__version__, gpu=torch.cuda.get_device_name(),
                      model=backend.adapter_metadata['model'], scheduler=dict(pipe.scheduler.config))

        def generate(text, seconds, seed, views=None):
            set_strength(pipe.transformer, 0)
            torch.manual_seed(seed)
            context = nullcontext({}) if views is None else window_consensus(pipe.transformer, views, spec['local_weight'])
            with torch.inference_mode(), context as counts:
                audio = pipe(prompt=text, audio_end_in_s=float(seconds), guidance_scale=7.,
                             num_inference_steps=steps, output_type='pt',
                             generator=torch.Generator('cuda').manual_seed(seed)).audios[0].float().cpu().numpy().T
            if audio.shape != (int(seconds*sr), 2) or not np.isfinite(audio).all():
                raise ValueError('Invalid generated audio')
            if views is not None and counts != dict(steps=steps, global_passes=steps, local_passes=steps*len(views)):
                raise ValueError('Unexpected prediction call counts')
            return audio, dict(counts)

        # Exact full-view check validates local text+duration+rotary wiring against
        # the installed real pipeline. Uses two steps, with identical prompt/duration.
        saved_steps = steps
        steps = 2
        text = prompt(['rain', 'dog'])
        cross, glob = conditioning(pipe, text, duration)
        reference, _ = generate(text, duration, 17)
        full_view = dict(begin=0, end=int(pipe.transformer.config.sample_size), cross=cross)
        full_view['global'] = glob
        actual, _ = generate(text, duration, 17, [full_view])
        parity = float(np.max(np.abs(reference-actual)))
        report['full_view_parity_max_abs'] = parity
        if parity > 1e-4:
            raise RuntimeError(f'Full-view parity failed: {parity}')
        steps = saved_steps
        print('FULL-VIEW PARITY PASSED', flush=True)

        for scene in scenes:
            joint_prompt = prompt([c[0] for c in scene['cues']])
            background_categories = [c for c, a, b in scene['cues'] if a == 0 and b == duration]
            background_prompt = prompt(background_categories) if background_categories else 'Quiet indoor room tone, no speech or vocals.'
            views = []
            for category, start, end in scene['cues']:
                # Local event plus overlapping full-scene beds, so a barking view
                # does not explicitly ask the model to remove the rain.
                beds = [c for c, a, b in scene['cues'] if a == 0 and b == duration and c != category]
                begin, finish = round(start/spacing), round(end/spacing)
                cross, glob = conditioning(pipe, prompt([category]+beds), (finish-begin)*spacing)
                views.append(dict(begin=begin, end=finish, cross=cross, category=category,
                                  **{'global':glob}, fade_frames=5))
            for seed in seeds:
                outputs, batch_records = {}, []
                for mode in MODES:
                    print(f'Generating {scene["id"]} seed={seed} mode={mode}', flush=True)
                    torch.cuda.reset_peak_memory_stats()
                    tick = time.perf_counter()
                    if mode == 'stems':
                        sound = np.zeros((round(duration*sr), 2), dtype=np.float32)
                        stem_records = []
                        for index, (category, start, end) in enumerate(scene['cues']):
                            stem_seed = cue_seed(seed, f'{scene["id"]}-{index}')
                            stem, _ = generate(prompt([category]), end-start, stem_seed)
                            rms = float(np.sqrt(np.mean(stem.astype(float)**2)))
                            # Fixed baseline recipe: equal RMS stems, bounded boost.
                            level_gain = min(10., .05/max(rms, 1e-8))
                            stem *= level_gain
                            fade = min(round(.15*sr), len(stem)//2)
                            ramp = np.linspace(0, 1, fade)
                            stem[:fade] *= ramp[:, None]
                            stem[-fade:] *= ramp[::-1, None]
                            stem_path = args.out/'stems'/f'{scene["id"]}-{seed}-{index}-{category}.wav'
                            storage_gain = min(1., .89/max(float(np.max(np.abs(stem))), 1e-8))
                            write_wav(stem_path, stem*storage_gain, sr)
                            stem_records.append(dict(category=category, seed=stem_seed, start_s=start, end_s=end,
                                level_gain=level_gain, storage_gain=storage_gain, raw_rms=rms,
                                file=str(stem_path.relative_to(args.out)), sha256=digest(stem_path)))
                            sound[round(start*sr):round(end*sr)] += stem
                        counts = dict(independent_generations=len(scene['cues']), denoiser_passes=steps*len(scene['cues']))
                    else:
                        local = None if mode == 'joint' else [dict(v, strength=int(mode=='window_lora' and v['category']=='dog')) for v in views]
                        sound, counts = generate(joint_prompt if mode == 'joint' else background_prompt, duration, seed, local)
                    torch.cuda.synchronize()
                    raw_rms = float(np.sqrt(np.mean(sound.astype(float)**2)))
                    raw_peak = float(np.max(np.abs(sound)))
                    raw_path = args.out/'raw'/f'{scene["id"]}-{seed}-{mode}.wav'
                    raw_gain = min(1., .89/max(raw_peak, 1e-8))
                    write_wav(raw_path, sound*raw_gain, sr)
                    rec = dict(scene=scene['id'], seed=seed, mode=mode, elapsed_s=time.perf_counter()-tick,
                               peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                               raw_rms=raw_rms, raw_peak=raw_peak, raw_storage_gain=raw_gain,
                               near_silent=raw_rms<1e-5, calls=counts, raw_file=str(raw_path.relative_to(args.out)), sha256=digest(raw_path))
                    if mode == 'stems': rec['stems'] = stem_records
                    rec['global_prompt'] = joint_prompt if mode == 'joint' else background_prompt if mode.startswith('window') else None
                    records.append(rec)
                    batch_records.append(rec)
                    outputs[mode] = sound
                    (args.out/'experiment.json').write_text(json.dumps(report, indent=2, default=str))
                # Exact group RMS match followed by ONE common headroom gain.
                # Raw files preserve relative level; near-silent outputs stay failures.
                scaled = {m:a*(.05/max(float(np.sqrt(np.mean(a.astype(float)**2))),1e-8)) for m,a in outputs.items()}
                common = min(1., .89/max(float(np.max(np.abs(a))) for a in scaled.values()))
                order = list(MODES)
                random.Random(cue_seed(seed, scene['id'])).shuffle(order)
                for index, mode in enumerate(order):
                    name = f'{scene["id"]}-{seed}-{chr(65+index)}'
                    listen_path = args.out/'listen'/f'{name}.wav'
                    write_wav(listen_path, scaled[mode]*common, sr)
                    rec = next(r for r in batch_records if r['mode']==mode)
                    rec.update(listen_file=str(listen_path.relative_to(args.out)), listen_sha256=digest(listen_path),
                               target_matched_rms=.05*common, actual_listen_rms=float(np.sqrt(np.mean((scaled[mode]*common).astype(float)**2))))
                    key.append(dict(sample=name, scene=scene['id'], seed=seed, mode=mode))
                    for cue_index, (category, start, end) in enumerate(scene['cues']):
                        ratings.append(dict(sample=name, cue_index=cue_index, category=category, target_start_s=start,
                                            target_end_s=end, present='', first_onset_s='', last_offset_s='',
                                            outside_window='', fidelity_1to5='', cohesion_1to5='', artifacts_1to5='', notes=''))
                (args.out/'experiment.json').write_text(json.dumps(report, indent=2, default=str))
        for filename, rows in [('blind-key.csv', key), ('ratings.csv', ratings)]:
            with (args.out/filename).open('w', newline='') as f:
                writer=csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
        report['elapsed_including_load_s'] = time.perf_counter()-started
        (args.out/'experiment.json').write_text(json.dumps(report, indent=2, default=str))
        status.write_text(json.dumps(dict(state='completed', smoke_test_only=args.smoke)))
        print('COMPOSITION STUDY COMPLETE', flush=True)
    except Exception as exc:
        status.write_text(json.dumps(dict(state='failed', error=str(exc))))
        raise


if __name__ == '__main__':
    main()
