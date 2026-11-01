import math
import unittest

from workbench.voice_linking import link_acoustic_measurements, link_voice_prototypes


class VoiceLinkingTests(unittest.TestCase):
    def test_local_over_splitting_can_merge_but_simultaneous_voices_cannot(self):
        def sample(overlap):
            return [{"index":0,"prototypes":[
                {"key":"0000:A","chunk":0,"first":0,"embedding":[1,0]},
                {"key":"0000:B","chunk":0,"first":1,"embedding":[.999,.001]}],
                "turns":[{"speaker":"A","start":0,"end":2},
                         {"speaker":"B","start":1 if overlap else 3,"end":4}]}]
        disjoint = link_acoustic_measurements(sample(False))
        self.assertEqual(disjoint["global_linking"]["global_clusters"],1)
        overlap = link_acoustic_measurements(sample(True))
        self.assertEqual(overlap["global_linking"]["global_clusters"],2)
        self.assertEqual(overlap["global_linking"]["cannot_link_pairs"],1)
        self.assertEqual([r["source_acoustic_key"] for r in disjoint["turns"]],["0000:A","0000:B"])

    def test_same_voice_across_windows_links_and_same_window_voices_stay_distinct(self):
        nodes = [{"key":"a1", "chunk":0, "first":1, "embedding":[1,0]},
                 {"key":"b1", "chunk":0, "first":3, "embedding":[0,1]},
                 {"key":"a2", "chunk":1, "first":10, "embedding":[.99,.01]},
                 {"key":"b2", "chunk":1, "first":12, "embedding":[.01,.99]}]
        mapping, audit = link_voice_prototypes(nodes)
        self.assertEqual(mapping["a1"], mapping["a2"])
        self.assertEqual(mapping["b1"], mapping["b2"])
        self.assertNotEqual(mapping["a1"], mapping["b1"])
        self.assertEqual(audit["global_clusters"], 2)
        # Even acoustically identical distinct local clusters are not merged.
        mapping, _ = link_voice_prototypes([{**n,"embedding":[1,0]} for n in nodes[:2]])
        self.assertNotEqual(mapping["a1"], mapping["b1"])

    def test_complete_link_does_not_chain_two_dissimilar_endpoints(self):
        nodes = [{"key":str(i),"chunk":i,"first":i,"embedding":[math.cos(a),math.sin(a)]}
                 for i,a in enumerate([0,.4,.8])]
        mapping, _ = link_voice_prototypes(nodes, threshold=.85)
        self.assertNotEqual(mapping["0"], mapping["2"])

    def test_missing_embedding_cannot_guess_identity_or_target_count(self):
        nodes = [{"key":str(i),"chunk":i,"first":i,"embedding":[]} for i in range(3)]
        mapping, audit = link_voice_prototypes(nodes)
        self.assertEqual(len(set(mapping.values())), 3)
        self.assertTrue(all(r["insufficient_embedding"] for r in audit["groups"]))
