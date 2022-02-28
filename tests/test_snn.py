import copy
import os
import pickle
import sys
import tempfile
import unittest

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf

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
        if hasattr(self, 'temp_file_name'):
            if os.path.isfile(self.temp_file_name):
                os.remove(self.temp_file_name)

    def test_fit_predict(self):
        snn_regressor = SNNRegressor(ensemble_size=3, minibatch_size=256,
                                     n_layers=18, hidden_layer_size=128,
                                     validation_fraction=0.1, patience=5,
                                     verbose=True)
        snn_regressor.fit(self.X_, self.y_)
        y_pred = snn_regressor.predict(self.X_)
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertEqual((self.y_.shape[0],), y_pred.shape)
        score = mean_absolute_error(y_true=self.y_, y_pred=y_pred)
        self.assertGreaterEqual(score, 0.0)
        print(f'MAE = {score}')
        print(f'MAPE = {mean_absolute_percentage_error(self.y_, y_pred)}')

    def test_copy(self):
        snn_regressor1 = SNNRegressor(ensemble_size=3, minibatch_size=256,
                                      n_layers=18, hidden_layer_size=128,
                                      validation_fraction=0.1, patience=5,
                                      verbose=True)
        snn_regressor1.fit(self.X_, self.y_)
        y_pred1 = snn_regressor1.predict(self.X_)
        snn_regressor2 = copy.copy(snn_regressor1)
        y_pred2 = snn_regressor2.predict(self.X_)
        self.assertEqual(y_pred1.shape, y_pred2.shape)
        for sample_idx in range(y_pred1.shape[0]):
            err_msg = f'[{sample_idx}]: {y_pred1[sample_idx]} != ' \
                      f'{y_pred2[sample_idx]}'
            self.assertAlmostEqual(y_pred1[sample_idx], y_pred2[sample_idx],
                                   msg=err_msg)

    def test_deepcopy(self):
        snn_regressor1 = SNNRegressor(ensemble_size=3, minibatch_size=256,
                                      n_layers=18, hidden_layer_size=128,
                                      validation_fraction=0.1, patience=5,
                                      verbose=True)
        snn_regressor1.fit(self.X_, self.y_)
        y_pred1 = snn_regressor1.predict(self.X_)
        snn_regressor2 = copy.deepcopy(snn_regressor1)
        y_pred2 = snn_regressor2.predict(self.X_)
        self.assertEqual(y_pred1.shape, y_pred2.shape)
        for sample_idx in range(y_pred1.shape[0]):
            err_msg = f'[{sample_idx}]: {y_pred1[sample_idx]} != ' \
                      f'{y_pred2[sample_idx]}'
            self.assertAlmostEqual(y_pred1[sample_idx], y_pred2[sample_idx],
                                   msg=err_msg)

    def test_serialization(self):
        snn_regressor1 = SNNRegressor(ensemble_size=3, minibatch_size=256,
                                      n_layers=18, hidden_layer_size=128,
                                      validation_fraction=0.1, patience=5,
                                      verbose=True)
        snn_regressor1.fit(self.X_, self.y_)
        y_pred1 = snn_regressor1.predict(self.X_)
        with tempfile.NamedTemporaryFile(mode='w', delete=True) as fp:
            self.temp_file_name = fp.name
        with open(self.temp_file_name, mode='wb') as fp:
            pickle.dump(snn_regressor1, fp)
        del snn_regressor1
        tf.keras.backend.clear_session()
        with open(self.temp_file_name, mode='rb') as fp:
            snn_regressor2 = pickle.load(fp)
        y_pred2 = snn_regressor2.predict(self.X_)
        self.assertEqual(y_pred1.shape, y_pred2.shape)
        for sample_idx in range(y_pred1.shape[0]):
            err_msg = f'[{sample_idx}]: {y_pred1[sample_idx]} != ' \
                      f'{y_pred2[sample_idx]}'
            self.assertAlmostEqual(y_pred1[sample_idx], y_pred2[sample_idx],
                                   msg=err_msg)


if __name__ == '__main__':
    unittest.main(verbosity=2)
