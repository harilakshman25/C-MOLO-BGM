import unittest
from bgm.alignment import align_cues


class AlignmentTests(unittest.TestCase):
    def fixture(self):
        plan=dict(schema_version=1,story='A dog barked.',duration_s=10,sample_rate=44100,seed=42,
            cues=[dict(id='dog',kind='sfx',category='dog',prompt='Dog barking',start_s=1,end_s=5,fade_s=.1)])
        segments=[dict(text='A dog barked.',start_s=1,end_s=4)]
        transcripts=[dict(text='A dog barked.',chunks=[dict(text=' barked.',timestamp=[2,2.5])])]
        return plan,segments,transcripts,dict(dog=dict(segment=0,word='barked'))

    def test_trigger_end_and_preserve_source(self):
        plan,segments,transcripts,anchors=self.fixture()
        result,changes=align_cues(plan,segments,transcripts,anchors)
        self.assertAlmostEqual(result['cues'][0]['start_s'],3.58)
        self.assertAlmostEqual(result['cues'][0]['end_s'],7.58)
        self.assertEqual(plan['cues'][0]['start_s'],1)
        self.assertEqual(len(changes),1)

    def test_missing_duplicate_and_invalid_timestamps_rejected(self):
        for chunks in ([],[dict(text='barked',timestamp=[0,1])]*2,
                       [dict(text='barked',timestamp=[None,1])],
                       [dict(text='barked',timestamp=[1,9])]):
            plan,segments,transcripts,anchors=self.fixture()
            transcripts[0]['chunks']=chunks
            with self.assertRaises(ValueError): align_cues(plan,segments,transcripts,anchors)

    def test_mismatched_transcript_rejected(self):
        plan,segments,transcripts,anchors=self.fixture()
        transcripts[0]['text']='Completely different speech.'
        with self.assertRaises(ValueError): align_cues(plan,segments,transcripts,anchors)
