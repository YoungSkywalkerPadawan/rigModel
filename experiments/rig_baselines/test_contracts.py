import unittest
import numpy as np
from shared import WORLD_TO_MODEL, normalize_world, restore, validate_graph


class GeometryContracts(unittest.TestCase):
    def test_original_postprocessing_runs_without_a_renderer(self):
        from postprocess import puppeteer_postprocess
        post = puppeteer_postprocess()
        raw = np.array([[[0., 0., 0.], [0., .1, 0.]], [[0., .2, 0.], [0., .3, 0.]]])
        joints, bones = post.pred_joints_and_bones(raw)
        joints, bones = post.merge_duplicate_joints_and_fix_bones(joints, bones)
        validate_graph(joints, bones)
        self.assertEqual(len(post.find_connected_components(joints, bones)), 1)
        self.assertEqual((len(joints), len(bones)), (4, 3))

    def test_world_z_becomes_model_y_without_reflection(self):
        np.testing.assert_array_equal(np.array([0., 0., 1.]) @ WORLD_TO_MODEL, [0., 1., 0.])
        self.assertAlmostEqual(np.linalg.det(WORLD_TO_MODEL), 1.)
        frame = dict(scale_m=.21, center_world=[4., -2., 1.], model_to_world_rotation=WORLD_TO_MODEL.T.tolist())
        points = np.array([[.13, -.7, .8], [-.1, .3, .9]])
        np.testing.assert_allclose(normalize_world(restore(points, frame), frame), points)

    def test_puppeteer_inverse_includes_both_normalizations(self):
        center = np.array([3., -1., .7])
        multiplier = 1.23
        pc_center = np.array([.002, -.004, .001])
        pc_scale = 1.999
        frame = dict(scale_m=pc_scale/multiplier,
                     center_world=((pc_center/multiplier+center) @ WORLD_TO_MODEL.T).tolist(),
                     model_to_world_rotation=WORLD_TO_MODEL.T.tolist())
        predicted = np.array([[.12, -.3, .42], [-.4, .2, -.1]])
        official = ((predicted*pc_scale+pc_center)/multiplier+center) @ WORLD_TO_MODEL.T
        np.testing.assert_allclose(restore(predicted, frame), official, atol=1e-12)

    def test_graph_validation_preserves_arbitrary_node_order(self):
        joints = np.array([[0., 0., 0.], [0., 1., 0.], [1., 0., 0.]])
        validate_graph(joints, np.array([[2, 0], [0, 1]]))
        for bones in (np.array([[0, 3]]), np.array([[0, 0]]), np.array([[0., 1.]])):
            with self.assertRaises(ValueError):
                validate_graph(joints, bones)
        with self.assertRaises(ValueError):
            validate_graph(joints*np.nan, np.array([[0, 1]]))


if __name__ == '__main__':
    unittest.main()
