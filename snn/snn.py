import gc
from typing import List

import numpy as np
import six
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import median_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import Binarizer, KBinsDiscretizer
from sklearn.utils import check_X_y
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras.utils import losses_utils, tf_utils
from tensorflow.python.ops.losses import util as tf_losses_util


class LossFunctionWrapper(tf.keras.losses.Loss):
    def __init__(self,
                 fn,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name=None,
                 **kwargs):
        super(LossFunctionWrapper, self).__init__(reduction=reduction,
                                                  name=name)
        self.fn = fn
        self._fn_kwargs = kwargs

    def call(self, y_true, y_pred):
        if tensor_util.is_tensor(y_pred) and tensor_util.is_tensor(y_true):
            y_pred, y_true = tf_losses_util.squeeze_or_expand_dimensions(
                y_pred, y_true
            )
        return self.fn(y_true, y_pred, **self._fn_kwargs)

    def get_config(self):
        config = {}
        for k, v in six.iteritems(self._fn_kwargs):
            config[k] = tf.keras.backend.eval(v) \
                if tf_utils.is_tensor_or_variable(v) \
                else v
        base_config = super(LossFunctionWrapper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def npairs_loss(labels, feature_vectors):
    feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
    logits = tf.divide(
        tf.matmul(
            feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
        ),
        0.5  # temperature
    )
    return tfa.losses.npairs_loss(tf.squeeze(labels), logits)


class NPairsLoss(LossFunctionWrapper):
    def __init__(self, reduction=losses_utils.ReductionV2.AUTO,
                 name='m_pairs_loss'):
        super(NPairsLoss, self).__init__(npairs_loss, name=name,
                                         reduction=reduction)


def build_preprocessor(X: np.ndarray, colnames: List[str]) -> Pipeline:
    X_ = Pipeline(steps=[
        (
            'imputer', SimpleImputer(
                missing_values=np.nan, strategy='constant',
                fill_value=-1.0
            )
        ),
        (
            'scaler',
            MinMaxScaler()
        )
    ]).fit_transform(X)
    X_ = np.rint(X_ * 100000.0).astype(np.int32)
    binary_features = dict()
    categorical_features = dict()
    removed_features = []
    for col_idx in range(X.shape[1]):
        values = set(X_[:, col_idx].tolist())
        print(f'Column {col_idx} "{colnames[col_idx]}" has '
              f'{len(values)} unique values.')
        if len(values) > 1:
            if len(values) < 3:
                binary_features[col_idx] = np.min(X[:, col_idx])
            else:
                categorical_features[col_idx] = len(values)
        else:
            removed_features.append(col_idx)
        del values
    del X_
    all_features = set(range(X.shape[1]))
    useful_features = sorted(list(all_features - set(removed_features)))
    if len(useful_features) == 0:
        raise ValueError('Training inputs are bad. All features are removed.')
    print(f'There are {X.shape[1]} features.')
    if len(removed_features) > 0:
        print(f'These features will be removed: '
              f'{[colnames[col_idx] for col_idx in removed_features]}.')
    transformers = []
    if (len(categorical_features) > 0) and (len(binary_features) > 0):
        print(f'There are {len(categorical_features)} categorical '
              f'features and {len(binary_features)} binary features.')
    elif len(categorical_features) > 0:
        print(f'There are {len(categorical_features)} categorical features.')
    else:
        print(f'There are {len(binary_features)} binary features.')
    for col_idx in categorical_features:
        n_unique_values = categorical_features[col_idx]
        transformers.append(
            (
                colnames[col_idx],
                KBinsDiscretizer(
                    n_bins=min(max(n_unique_values // 3, 3), 256),
                    encode='ordinal',
                    strategy=('quantile' if n_unique_values > 50 else 'kmeans')
                ),
                (col_idx,)
            )
        )
    for col_idx in binary_features:
        transformers.append(
            (
                colnames[col_idx],
                Binarizer(threshold=0.0),
                (col_idx,)
            )
        )
    preprocessor = Pipeline(steps=[
        (
            'imputer', SimpleImputer(
                missing_values=np.nan, strategy='constant',
                fill_value=-1.0
            )
        ),
        (
            'minmax_scaler',
            MinMaxScaler()
        ),
        (
            'composite_transformer', ColumnTransformer(
                transformers=transformers,
                sparse_threshold=0.0,
                n_jobs=1
            )
        ),
        (
            'selector',
            VarianceThreshold()
        ),
        (
            'standard_scaler',
            StandardScaler(with_mean=True, with_std=True)
        ),
        (
            'pca',
            PCA(random_state=42)
        )
    ])
    return preprocessor.fit(X)


class SNNRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, ensemble_size: int = 20,
                 hidden_layer_size: int = 512, n_layers: int = 18,
                 dropout_rate: float = 3e-4,
                 max_epochs: int = 1000, patience: int = 15,
                 minibatch_size: int = 4096, validation_fraction: float = 0.1,
                 verbose: bool = False, clear_session: bool = True):
        super(SNNRegressor, self).__init__()
        self.ensemble_size = ensemble_size
        self.hidden_layer_size = hidden_layer_size
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.minibatch_size = minibatch_size
        self.verbose = verbose
        self.validation_fraction = validation_fraction
        self.clear_session = clear_session

    def fit(self, X, y, **kwargs):
        self.check_params(
            ensemble_size=self.ensemble_size,
            hidden_layer_size=self.hidden_layer_size,
            n_layers=self.n_layers,
            minibatch_size=self.minibatch_size,
            dropout_rate=self.dropout_rate,
            max_epochs=self.max_epochs,
            patience=self.patience,
            validation_fraction=self.validation_fraction,
            verbose=self.verbose,
            clear_session=self.clear_session
        )
        X_, y_ = check_X_y(X, y,
                           force_all_finite='allow-nan',
                           ensure_min_samples=max(self.ensemble_size * 3, 100),
                           multi_output=False, y_numeric=True,
                           estimator='SNNRegressor')
        if 'feature_names' in kwargs:
            feature_names = kwargs['feature_names']
            if (not isinstance(feature_names, list)) and \
                    (not isinstance(feature_names, tuple)):
                err_msg = f'`{feature_names}` is wrong! ' \
                          f'Expected `{type([1, 2])}`, ' \
                          f'got `{type(kwargs["feature_names"])}`.'
                raise ValueError(err_msg)
        else:
            max_number_width = len(str(X_.shape[1]))
            feature_names = ['x{0:>0{1}}'.format(col_idx, max_number_width)
                             for col_idx in range(X_.shape[1])]
        if hasattr(self, 'nn_'):
            del self.nn_
        if hasattr(self, 'preprocessor_'):
            del self.preprocessor_
        if hasattr(self, 'postprocessors_'):
            del self.postprocessors_
        gc.collect()
        if self.clear_session:
            tf.keras.backend.clear_session()
        self.preprocessor_ = build_preprocessor(X_, feature_names)
        pass

    @staticmethod
    def check_integer_param(param_name: str, **kwargs):
        if param_name not in kwargs:
            raise ValueError(f'`{param_name}` is not specified!')
        if (not isinstance(kwargs[param_name], int)) and \
                (not isinstance(kwargs[param_name], np.int64)) and \
                (not isinstance(kwargs[param_name], np.uint64)) and \
                (not isinstance(kwargs[param_name], np.int16)) and \
                (not isinstance(kwargs[param_name], np.uint16)) and \
                (not isinstance(kwargs[param_name], np.int8)) and \
                (not isinstance(kwargs[param_name], np.uint8)) and \
                (not isinstance(kwargs[param_name], np.int32)) and \
                (not isinstance(kwargs[param_name], np.uint32)):
            err_msg = f'`{param_name}` is wrong! Expected `{type(3)}`, ' \
                      f'got `{type(kwargs[param_name])}`.'
            raise ValueError(err_msg)
        if kwargs[param_name] < 1:
            err_msg = f'`{param_name}` is wrong! Expected a positive ' \
                      f'integer value, but {kwargs[param_name]} is ' \
                      f'not positive.'
            raise ValueError(err_msg)

    @staticmethod
    def check_boolean_param(param_name: str, **kwargs):
        if param_name not in kwargs:
            raise ValueError(f'`{param_name}` is not specified!')
        if (not isinstance(kwargs[param_name], int)) and \
                (not isinstance(kwargs[param_name], bool)) and \
                (not isinstance(kwargs[param_name], np.bool)) and \
                (not isinstance(kwargs[param_name], np.int64)) and \
                (not isinstance(kwargs[param_name], np.uint64)) and \
                (not isinstance(kwargs[param_name], np.int16)) and \
                (not isinstance(kwargs[param_name], np.uint16)) and \
                (not isinstance(kwargs[param_name], np.int8)) and \
                (not isinstance(kwargs[param_name], np.uint8)) and \
                (not isinstance(kwargs[param_name], np.int32)) and \
                (not isinstance(kwargs[param_name], np.uint32)):
            err_msg = f'`{param_name}` is wrong! Expected `{type(True)}`, ' \
                      f'got `{type(kwargs[param_name])}`.'
            raise ValueError(err_msg)

    @staticmethod
    def check_float_param(param_name: str, **kwargs):
        if param_name not in kwargs:
            raise ValueError(f'`{param_name}` is not specified!')
        if (not isinstance(kwargs[param_name], float)) and \
                (not isinstance(kwargs[param_name], np.float)) and \
                (not isinstance(kwargs[param_name], np.float16)) and \
                (not isinstance(kwargs[param_name], np.float8)) and \
                (not isinstance(kwargs[param_name], np.float32)):
            err_msg = f'`{param_name}` is wrong! Expected `{type(2.3)}`, ' \
                      f'got `{type(kwargs[param_name])}`.'
            raise ValueError(err_msg)

    @staticmethod
    def check_params(**kwargs):
        SNNRegressor.check_integer_param('minibatch_size', **kwargs)
        SNNRegressor.check_integer_param('patience', **kwargs)
        SNNRegressor.check_integer_param('ensemble_size', **kwargs)
        SNNRegressor.check_integer_param('hidden_layer_size', **kwargs)
        SNNRegressor.check_integer_param('n_layers', **kwargs)
        SNNRegressor.check_integer_param('max_epochs', **kwargs)
        SNNRegressor.check_boolean_param('verbose', **kwargs)
        SNNRegressor.check_boolean_param('clear_session', **kwargs)
        SNNRegressor.check_float_param('dropout_rate', **kwargs)
        SNNRegressor.check_float_param('validation_fraction', **kwargs)
        if kwargs['dropout_rate'] < 0.0:
            err_msg = f'`dropout_rate` is wrong! Expected a non-negative ' \
                      f'value, but {kwargs["dropout_rate"]} is negative!'
            raise ValueError(err_msg)
        if kwargs['validation_fraction'] <= 0.0:
            err_msg = f'`validation_fraction` is wrong! Expected a ' \
                      f'floating-point value in (0.0, 1.0), ' \
                      f'but {kwargs["validation_fraction"]} is inadmissible!'
            raise ValueError(err_msg)
        if kwargs['validation_fraction'] >= 1.0:
            err_msg = f'`validation_fraction` is wrong! Expected a ' \
                      f'floating-point value in (0.0, 1.0), ' \
                      f'but {kwargs["validation_fraction"]} is inadmissible!'
            raise ValueError(err_msg)
