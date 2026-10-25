"""Stable Audio 1.0 dog-LoRA feasibility trainer for the verified V100 stack.

Not a reproduction of full-model pretraining. Read docs/LORA_PILOT.md for assumptions.
"""
import argparse
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
import torch
from diffusers import StableAudioPipeline
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from bgm.lora import inject_lora, adapter_parameters, adapter_state, load_adapter_state, velocity_example
from bgm.training import reviewed_records, digest, cache_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("checkpoints/stable-audio-open-1.0"))
    parser.add_argument("--manifest", type=Path, default=Path("data/esc50-dog/manifest.jsonl"))
    parser.add_argument("--review", type=Path, default=Path("docs/DOG_DATA_REVIEW.json"))
    parser.add_argument("--cache", type=Path, default=Path("data/esc50-dog/cache-v1.pt"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=7026)
    args = parser.parse_args()
    if not 1 <= args.steps <= 2000:
        raise ValueError("steps must be 1..2000")
    if args.smoke:
        args.steps = 2
    args.out.mkdir(parents=True, exist_ok=False)
    status = args.out / "status.json"
    status.write_text(json.dumps({"state": "running"}))
    started = time.perf_counter()
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Training requires one allocated CUDA GPU")
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
        rows = reviewed_records(args.manifest, args.review)
        model_identity = json.loads((args.model/"download_manifest.json").read_text())
        identity = {"cache_version": 1, "model": model_identity,
                    "manifest_sha256": digest(args.manifest), "review_sha256": digest(args.review),
                    "policy": "FP32-posterior-mode-full-waveform-padding"}
        print("Loading frozen backbone and caching train/validation features; test clips excluded...", flush=True)
        pipe = StableAudioPipeline.from_pretrained(str(args.model), torch_dtype=torch.float32, local_files_only=True)
        if pipe.scheduler.config.prediction_type != "v_prediction" or float(pipe.scheduler.config.sigma_data) != 1.0:
            raise ValueError("Trainer requires v_prediction with sigma_data=1")
        if type(pipe.scheduler).__name__ != "CosineDPMSolverMultistepScheduler":
            raise ValueError("Unexpected scheduler; inspect its training parameterization first")
        for component in (pipe.vae, pipe.text_encoder, pipe.projection_model, pipe.transformer):
            component.requires_grad_(False).eval()
        cache = cache_features(pipe, rows, args.manifest, args.cache, identity)
        model = pipe.transformer.to(device="cuda", dtype=torch.float16)
        # Training uses no decoding; frozen encoders remain on CPU after caching.
        train = [item for item in cache["clips"] if item["split"] == "train"]
        validation = [item for item in cache["clips"] if item["split"] == "validation"]
        if len(train) != 24 or len(validation) != 8:
            raise ValueError("Invalid cached split sizes")
        if args.smoke:
            validation = validation[:2]
        conditions = {key: {k: v.to(device="cuda", dtype=torch.float16) for k, v in value.items()}
                      for key, value in cache["conditions"].items()}
        rotary = get_1d_rotary_pos_embed(pipe.rotary_embed_dim, cache["frames"]+1,
                                       use_real=True, repeat_interleave_real=False)

        def prediction(x, t, caption, drop=False):
            cond = conditions[caption]
            cross = torch.zeros_like(cond["cross"]) if drop else cond["cross"]
            with torch.autocast("cuda", dtype=torch.float16):
                return model(x.to(torch.float16), t, encoder_hidden_states=cross,
                             global_hidden_states=cond["global"], rotary_embedding=rotary,
                             return_dict=False)[0]

        def loss_for(item, generator, fixed_time=None, drop=False):
            clean = item["latent"].to(device="cuda", dtype=torch.float32)
            noise = torch.randn(clean.shape, generator=generator, device="cuda")
            t = (torch.rand((1,), generator=generator, device="cuda") * 0.9998 + 0.0001
                 if fixed_time is None else torch.tensor([fixed_time], device="cuda"))
            noisy, target = velocity_example(clean, noise, t)
            output = prediction(noisy, t, item["caption"], drop)
            weight = torch.full((1, 1, clean.shape[-1]), 0.1, device="cuda")
            weight[..., :item["active_frames"]] = 1.0
            # Keep a padding penalty while preventing 42 seconds of silence dominating 5 seconds of bark.
            return ((output.float()-target).square()*weight).sum() / (weight.sum()*clean.shape[1])

        # Compare exactly the same tensor before and after zero-initialized adapter injection.
        probe = train[0]["latent"].to("cuda") * 0.5
        probe_t = torch.tensor([0.5], device="cuda")
        with torch.no_grad():
            before = prediction(probe, probe_t, train[0]["caption"]).detach()
        targets = inject_lora(model, rank=8, alpha=8)
        with torch.no_grad():
            after = prediction(probe, probe_t, train[0]["caption"]).detach()
        identity_error = float((before-after).abs().max())
        if identity_error != 0.0:
            raise RuntimeError(f"Zero-initialized LoRA changed the base output: {identity_error}")
        del before, after
        params = adapter_parameters(model)
        if not params or any(p.dtype != torch.float32 for p in params):
            raise RuntimeError("Expected FP32 trainable adapter parameters")
        if {id(p) for p in params} != {id(p) for p in model.parameters() if p.requires_grad}:
            raise RuntimeError("Unexpected trainable base parameters")
        model.enable_gradient_checkpointing()
        config = {"format": "cmolo-linear-lora-v1", "model": model_identity, "rank": 8, "alpha": 8,
                  "targets": targets, "training_seed": args.seed, "steps": args.steps,
                  "smoke_test_only": args.smoke, "learning_rate": 1e-4, "batch_size": 1,
                  "gradient_accumulation": 1 if args.smoke else 4,
                  "gradient_clipping": 1.0, "caption_dropout_probability": 0.1,
                  "objective": "cosine-v-MSE-uniform-t-in-(0,1)", "padding_loss_weight": 0.1,
                  "trainable_parameters": sum(p.numel() for p in params), "identity_error": identity_error,
                  "cache_identity": identity, "review": json.loads(args.review.read_text()),
                  "train_ids": [x["id"] for x in train], "validation_ids": [x["id"] for x in validation],
                  "test_ids_used": [], "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0),
                  "codec_policy": cache["encoder_policy"], "prediction_type": "v_prediction"}
        (args.out/"config.json").write_text(json.dumps(config, indent=2))
        (args.out/"packages.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
        print(f"Zero-adapter identity PASS; {len(targets)} projections; {config['trainable_parameters']:,} trainable parameters", flush=True)

        def save(name, step, validation_loss):
            bundle = {"metadata": {**config, "step": step, "validation_loss": validation_loss},
                      "adapter": adapter_state(model)}
            temp = args.out / (name + ".tmp")
            torch.save(bundle, temp)
            temp.replace(args.out/name)

        @torch.no_grad()
        def evaluate():
            model.eval()
            generator = torch.Generator("cuda").manual_seed(112026)
            losses = []
            for item in validation:
                for t in ((0.5,) if args.smoke else (0.2, 0.5, 0.8)):
                    value = float(loss_for(item, generator, fixed_time=t))
                    if not math.isfinite(value):
                        raise RuntimeError("Non-finite validation loss")
                    losses.append(value)
            return sum(losses)/len(losses)

        metrics = args.out/"metrics.jsonl"
        def log(row):
            row["elapsed_s"] = time.perf_counter()-started
            with metrics.open("a") as handle:
                handle.write(json.dumps(row)+"\n")
            print(json.dumps(row), flush=True)

        initial = evaluate()
        best, best_step = initial, 0
        save("best-adapter.pt", 0, best)
        log({"step": 0, "validation_loss": initial, "phase": "untrained_base"})
        optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)
        scaler = torch.amp.GradScaler("cuda", init_scale=128.)
        generator = torch.Generator("cuda").manual_seed(args.seed)
        selection = random.Random(args.seed)
        last_validation = initial
        for step in range(1, args.steps+1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total = 0.
            for _ in range(config["gradient_accumulation"]):
                item = train[selection.randrange(len(train))]
                loss = loss_for(item, generator, drop=selection.random() < 0.1)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                total += float(loss.detach()) / config["gradient_accumulation"]
                scaler.scale(loss/config["gradient_accumulation"]).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            if float(norm) == 0:
                raise RuntimeError("Adapter received zero gradients")
            scaler.step(optimizer)
            scaler.update()
            if step == 1 or step % 10 == 0 or step == args.steps:
                log({"step": step, "train_loss": total, "gradient_norm_before_clip": float(norm)})
            if step % 50 == 0 or step == args.steps:
                last_validation = evaluate()
                log({"step": step, "validation_loss": last_validation})
                save(f"step-{step:04d}.pt", step, last_validation)
                if last_validation < best:
                    best, best_step = last_validation, step
                    save("best-adapter.pt", step, best)
        save("last-adapter.pt", args.steps, last_validation)
        # Prove the trained weights survive serialization, not just that a file exists.
        model.eval()
        with torch.no_grad():
            expected = prediction(probe, probe_t, train[0]["caption"]).detach()
            state = torch.load(args.out/"last-adapter.pt", map_location="cpu", weights_only=True)
            changed = sum(float(v.abs().sum()) for k, v in state["adapter"].items() if k.endswith(".lora_b"))
            if changed == 0:
                raise RuntimeError("No adapter updates were saved")
            for parameter in params:
                parameter.zero_()
            load_adapter_state(model, state["adapter"])
            actual = prediction(probe, probe_t, train[0]["caption"])
            reload_error = float((expected-actual).abs().max())
            if reload_error > 1e-4:
                raise RuntimeError(f"Adapter reload changed prediction: {reload_error}")
        report = {"state": "completed", "smoke_test_only": args.smoke, "steps": args.steps,
                  "initial_validation_loss": initial, "best_validation_loss": best, "best_step": best_step,
                  "last_validation_loss": last_validation, "reload_max_abs_error": reload_error,
                  "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
                  "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30,
                  "elapsed_s": time.perf_counter()-started,
                  "note": "Denoising loss is not proof of improved audible quality; compare generated samples."}
        (args.out/"report.json").write_text(json.dumps(report, indent=2))
        status.write_text(json.dumps(report, indent=2))
        print("LORA SMOKE PASSED" if args.smoke else "LORA PILOT COMPLETE", flush=True)
        print(json.dumps(report, indent=2), flush=True)
    except Exception as exc:
        status.write_text(json.dumps({"state": "failed", "error": str(exc)}, indent=2))
        raise


if __name__ == "__main__":
    main()
