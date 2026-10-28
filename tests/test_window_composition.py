import unittest
try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None, 'Optional torch dependency')
class WindowTests(unittest.TestCase):
    def test_overlap_normalization_shared_latent_and_restore(self):
        from bgm.window_composition import window_consensus
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.seen=[]
            def forward(self,x,t,encoder_hidden_states=None,**kwargs):
                self.seen.append((x.clone(),t))
                return (x+encoder_hidden_states.mean(),)
        m=Model()
        x=torch.arange(6.).reshape(1,1,6)
        views=[dict(begin=1,end=4,cross=torch.tensor([2.]),fade_frames=0,**{'global':None}),
               dict(begin=3,end=5,cross=torch.tensor([4.]),fade_frames=0,**{'global':None})]
        with window_consensus(m,views,1) as stats:
            got=m(x,.5,encoder_hidden_states=torch.tensor([0.]),return_dict=False)[0]
        torch.testing.assert_close(got,x+torch.tensor([0.,1.,1.,2.,2.,0.]))
        torch.testing.assert_close(m.seen[1][0],x[...,1:4])
        torch.testing.assert_close(m.seen[2][0],x[...,3:5])
        self.assertEqual(stats,dict(steps=1,global_passes=1,local_passes=2))
        self.assertNotIn('forward',m.__dict__)
        with self.assertRaises(ValueError):
            with window_consensus(m,[dict(begin=-1,end=4)]):
                m(x,.5,encoder_hidden_states=torch.tensor([0.]),return_dict=False)
        self.assertNotIn('forward',m.__dict__)

    def test_real_transformer_crop_rotary_and_full_view_identity(self):
        from diffusers import StableAudioDiTModel
        from diffusers.models.embeddings import get_1d_rotary_pos_embed
        from bgm.window_composition import window_consensus
        from bgm.lora import inject_lora, set_strength
        torch.manual_seed(91)
        m=StableAudioDiTModel(sample_size=16,in_channels=4,out_channels=4,num_layers=1,
            attention_head_dim=8,num_attention_heads=2,num_key_value_attention_heads=1,
            cross_attention_dim=8,time_proj_dim=8,global_states_input_dim=16,cross_attention_input_dim=8).eval()
        inject_lora(m,rank=2,alpha=2)
        set_strength(m,0)
        x=torch.randn(2,4,16)
        cross=torch.randn(2,5,8); glob=torch.randn(2,1,16)
        kwargs=dict(encoder_hidden_states=cross,global_hidden_states=glob,return_dict=False,
                    rotary_embedding=get_1d_rotary_pos_embed(4,17,use_real=True,repeat_interleave_real=False))
        view=dict(begin=0,end=16,cross=cross,**{'global':glob})
        with torch.inference_mode():
            expected=m(x,torch.tensor([.5]),**kwargs)[0]
            with window_consensus(m,[view]):
                actual=m(x,torch.tensor([.5]),**kwargs)[0]
            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
            view.update(begin=3,end=11,strength=1)
            with window_consensus(m,[view]):
                crop=m(x,torch.tensor([.5]),**kwargs)[0]
            self.assertEqual(crop.shape,x.shape)
            self.assertTrue(torch.isfinite(crop).all())
            torch.testing.assert_close(crop[...,:3],expected[...,:3],atol=0,rtol=0)
