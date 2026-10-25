"""Optional torch tests; skipped in the lightweight CPU environment."""
import io
import importlib.util
import math
import unittest
try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    from bgm.lora import LoRALinear, inject_lora, adapter_parameters, adapter_state, load_adapter_state, velocity_example


def tiny_model():
    model = torch.nn.Module()
    model.transformer_blocks = torch.nn.ModuleList([torch.nn.Module()])
    block = model.transformer_blocks[0]
    for name in ("attn1", "attn2"):
        attn = torch.nn.Module()
        attn.to_q = torch.nn.Linear(4, 4)
        attn.to_k = torch.nn.Linear(4, 4)
        attn.to_v = torch.nn.Linear(4, 4)
        setattr(block, name, attn)
    return model


@unittest.skipIf(torch is None, "Install GPU environment to exercise torch LoRA tests")
class LoRATests(unittest.TestCase):
    def test_zero_adapter_and_frozen_base_after_learning(self):
        torch.manual_seed(12)
        base = torch.nn.Linear(4, 3)
        weight = base.weight.detach().clone()
        bias = base.bias.detach().clone()
        x = torch.randn(8, 4)
        expected = base(x).detach()
        wrapped = LoRALinear(base, rank=2, alpha=2)
        torch.testing.assert_close(wrapped(x), expected, rtol=0, atol=0)
        optimizer = torch.optim.AdamW([wrapped.lora_a, wrapped.lora_b], lr=0.01)
        for _ in range(3):
            optimizer.zero_grad()
            loss = (wrapped(x)-1).square().mean()
            loss.backward()
            optimizer.step()
        self.assertGreater(float(wrapped.lora_b.abs().sum()), 0)
        self.assertIsNone(base.weight.grad)
        torch.testing.assert_close(base.weight, weight, rtol=0, atol=0)
        torch.testing.assert_close(base.bias, bias, rtol=0, atol=0)
        wrapped.strength = 0
        torch.testing.assert_close(wrapped(x), expected, rtol=0, atol=0)

    def test_target_selection_and_checkpoint_reload(self):
        model = tiny_model()
        targets = inject_lora(model, rank=2, alpha=2)
        self.assertEqual(len(targets), 4)
        self.assertIsInstance(model.transformer_blocks[0].attn1.to_k, torch.nn.Linear)
        for p in adapter_parameters(model):
            with torch.no_grad():
                p.add_(0.1)
        x = torch.randn(2, 4)
        module = model.transformer_blocks[0].attn1.to_q
        expected = module(x).detach()
        buffer = io.BytesIO()
        torch.save(adapter_state(model), buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        self.assertTrue(all("lora_" in key for key in saved))
        for p in adapter_parameters(model):
            with torch.no_grad():
                p.zero_()
        load_adapter_state(model, saved)
        torch.testing.assert_close(module(x), expected, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            inject_lora(model)
        with self.assertRaises(ValueError):
            load_adapter_state(model, {})

    def test_velocity_matches_scheduler_preconditioning(self):
        torch.manual_seed(3)
        clean, noise = torch.randn(3, 2, 5), torch.randn(3, 2, 5)
        t = torch.tensor([0.1, 0.5, 0.95])
        noisy, target = velocity_example(clean, noise, t)
        sigma = torch.tan(t[:, None, None]*math.pi/2)
        unscaled = clean + sigma*noise
        torch.testing.assert_close(noisy, unscaled / torch.sqrt(1+sigma.square()))
        # Exact cosine scheduler c_skip and negative c_out for v_prediction.
        recovered = unscaled/(1+sigma.square()) - sigma/torch.sqrt(1+sigma.square())*target
        torch.testing.assert_close(recovered, clean, atol=1e-5, rtol=1e-5)

    @unittest.skipUnless(importlib.util.find_spec("diffusers"), "Diffusers required for miniature model integration")
    def test_real_transformer_checkpointed_backward(self):
        from diffusers import StableAudioDiTModel
        from diffusers.models.embeddings import get_1d_rotary_pos_embed
        torch.manual_seed(18)
        model = StableAudioDiTModel(sample_size=16, in_channels=4, out_channels=4,
                                   num_layers=1, attention_head_dim=8, num_attention_heads=2,
                                   num_key_value_attention_heads=1, cross_attention_dim=8,
                                   time_proj_dim=8, global_states_input_dim=16,
                                   cross_attention_input_dim=8)
        x = torch.randn(1, 4, 16)
        t = torch.tensor([0.4])
        kwargs = dict(encoder_hidden_states=torch.randn(1, 5, 8),
                      global_hidden_states=torch.randn(1, 1, 16),
                      rotary_embedding=get_1d_rotary_pos_embed(4, 17, use_real=True,
                                                              repeat_interleave_real=False))
        model.eval()
        with torch.no_grad():
            original = model(x, t, **kwargs).sample
        inject_lora(model, rank=2, alpha=2)
        with torch.no_grad():
            torch.testing.assert_close(model(x, t, **kwargs).sample, original, rtol=0, atol=0)
        model.enable_gradient_checkpointing()
        model.train()
        model(x, t, **kwargs).sample.square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapter_parameters(model)))
        self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))


if __name__ == "__main__":
    unittest.main()
