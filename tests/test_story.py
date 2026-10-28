import json
from pathlib import Path
import unittest
import numpy as np
from bgm.story import build_plan, fit_stem, level_audio
from bgm.audio import mix


class StoryTests(unittest.TestCase):
    def test_timing_categories_and_mix(self):
        story=json.loads(Path('examples/forest_cabin.json').read_text())
        library=json.loads(Path('examples/sound_library.json').read_text())
        timings=[dict(start_s=1+i*8,end_s=6+i*8) for i in range(8)]
        plan=build_plan(story,timings,library,16000)
        self.assertEqual(set(library),{x['category'] for x in plan['cues']})
        stems={x['id']:np.ones((round(x['end_s']*16000)-round(x['start_s']*16000),2))*.02 for x in plan['cues']}
        voice=np.ones((round(timings[-1]['end_s']*16000),2))*.05
        audio,metrics=mix(plan,stems,voice)
        self.assertTrue(metrics['narration_present'])
        self.assertEqual(len(audio),round(plan['duration_s']*16000))
        self.assertTrue(np.isfinite(audio).all())

    def test_loop_pad_and_level(self):
        sound=np.ones((10,2))*.1
        np.testing.assert_allclose(fit_stem(sound,100,100,True),.1)
        padded=fit_stem(sound,100,100,False)
        np.testing.assert_array_equal(padded[10:],0)
        self.assertEqual(fit_stem(sound,4,100).shape,(4,2))
        leveled,gain=level_audio(sound)
        self.assertLessEqual(float(np.max(np.abs(leveled))),.85)
        with self.assertRaises(ValueError): level_audio(np.zeros((10,2)))
