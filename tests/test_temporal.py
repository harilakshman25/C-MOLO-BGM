import unittest
try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    from bgm.lora import LoRALinear
    from bgm.temporal import masked_adapter


@unittest.skipIf(torch is None, "Requires optional torch environment")
class TemporalTests(unittest.TestCase):
    def model(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = LoRALinear(torch.nn.Linear(2, 2), rank=1)
                self.layer.lora_b.data.fill_(0.5)
                self.seen = []

            def forward(self, x, timestep, return_dict=False):
                self.seen.append((x.data_ptr(), timestep))
                return (self.layer(x.transpose(1, 2)).transpose(1, 2),)
        return Model().eval()

    def test_shared_inputs_blend_and_restore(self):
        model = self.model()
        x = torch.randn(2, 2, 4)  # CFG batch has two branches.
        model.layer.strength = 0
        base = model(x, 0.5)[0]
        model.layer.strength = 1
        full = model(x, 0.5)[0]
        mask = torch.tensor([0., 0.25, 0.75, 1.])
        model.seen.clear()
        with masked_adapter(model, mask) as stats:
            actual = model(x, 0.5, return_dict=False)[0]
        torch.testing.assert_close(actual, base+mask*(full-base))
        self.assertEqual(model.seen, [(x.data_ptr(), 0.5)]*2)
        self.assertEqual(stats, {"steps": 1, "base_passes": 1, "expert_passes": 1})
        self.assertNotIn("forward", model.__dict__)
        self.assertEqual(model.layer.strength, 1)
        # Blending before linear CFG must equal blending guided predictions.
        def cfg(value):
            uncond, cond = value.chunk(2)
            return uncond+7*(cond-uncond)
        torch.testing.assert_close(cfg(actual), cfg(base)+mask*(cfg(full)-cfg(base)))

    def test_endpoints_and_failure_cleanup(self):
        model = self.model()
        x = torch.randn(2, 2, 4)
        for strength in (0, 1):
            model.layer.strength = strength
            expected = model(x, 0.5)[0]
            with masked_adapter(model, torch.full((4,), float(strength))):
                actual = model(x, 0.5, return_dict=False)[0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        model.layer.strength = 0.3
        with self.assertRaises(ValueError):
            with masked_adapter(model, torch.ones(3)):
                model(x, 0.5, return_dict=False)
        self.assertEqual(model.layer.strength, 0.3)
        self.assertNotIn("forward", model.__dict__)
        with self.assertRaises(ValueError):
            with masked_adapter(model, torch.tensor([float('nan')])):
                pass
