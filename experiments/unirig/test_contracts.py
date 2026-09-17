import contextlib
import io
from pathlib import Path
import tempfile
import unittest
import numpy as np
from common import point_axis_errors, restore, validate_skeleton, write, sha, read
from score import score


class Contracts(unittest.TestCase):
    def test_normalization_and_axis_gauge(self):
        frame = dict(scale_m=0.3, center_world=[4, -2, 10])
        p = np.array([[1, -1, 0], [0.2, 0.4, -0.7]])
        np.testing.assert_allclose((restore(p, frame)-frame['center_world'])/frame['scale_m'], p)
        errors = point_axis_errors([[0.02, 100, 0], [0.02, -40, 0]], [0, 5, 0], [0, 1, 0])
        np.testing.assert_allclose(errors, [0.02, 0.02])

    def test_bad_skeleton_is_rejected(self):
        for parents in [[None, 2], [0, 0], [None, None]]:
            with self.assertRaises(ValueError):
                validate_skeleton(np.zeros((2, 3)), parents)
        validate_skeleton(np.zeros((2, 3)), [None, 0])

    def test_failed_and_missing_predictions_stay_in_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, prediction, output = root/'data', root/'pred', root/'score'
            refs = [dict(sample_id=str(i), asset_key='user/a'+str(i), joint_name='j',
                         center_world=[0.1, 0, 0], origin_world=[0, 0, 0], direction_world=[0, 1, 0], child_scale_m=1.) for i in range(3)]
            write(data/'reference_axes.json', refs)
            write(data/'READY.json', dict(files={'reference_axes.json': sha(data/'reference_axes.json')}))
            for i, state in [(0, 'success'), (1, 'failed')]:
                write(prediction/f'a{i}/candidates.json', dict(joints_world=[[0, 99, 0]]))
                write(prediction/f'a{i}/status.json', dict(status=state))
            with contextlib.redirect_stdout(io.StringIO()):
                score(data, prediction, output)
            result = read(output/'summary.json')
            self.assertEqual((result['planned'], result['scored'], result['oracle_pass_1pct']), (3, 1, 1))


if __name__ == '__main__':
    unittest.main()
