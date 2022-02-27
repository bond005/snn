import os
import sys
import unittest

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error

try:
    from snn.snn import SNNRegressor
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from snn.snn import SNNRegressor


class TestRegressor(unittest.TestCase):
    def setUp(self) -> None:
        self.X_, self.y_ = load_diabetes(return_X_y=True)
        self.X_ = self.X_.astype(np.float32)
        self.y_ = self.y_.astype(np.float32)

    def tearDown(self) -> None:
        if hasattr(self, 'X_'):
            del self.X_
        if hasattr(self, 'y_'):
            del self.y_

    def test_fit_predict(self):
        snn_regressor = SNNRegressor(ensemble_size=5, minibatch_size=16,
                                     verbose=True)
        snn_regressor.fit(self.X_, self.y_)
        y_pred = snn_regressor.predict(self.X_)
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertEqual((self.y_.shape[0], 2), y_pred.shape)
        score = mean_absolute_error(y_true=self.y_, y_pred=y_pred[:, 0])
        self.assertLess(score, 1.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
