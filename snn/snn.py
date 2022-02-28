import copy
import gc
import random
import re
from typing import List, Tuple, Union
import uuid

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import Binarizer, KBinsDiscretizer
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp


def build_preprocessor(X: np.ndarray, colnames: List[str],
                       verbose: bool, seed: int) -> Pipeline:
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
        if verbose:
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
    if verbose:
        print(f'There are {X.shape[1]} features.')
        if len(removed_features) > 0:
            print(f'These features will be removed: '
                  f'{[colnames[col_idx] for col_idx in removed_features]}.')
    transformers = []
    if verbose:
        if (len(categorical_features) > 0) and (len(binary_features) > 0):
            print(f'There are {len(categorical_features)} categorical '
                  f'features and {len(binary_features)} binary features.')
        elif len(categorical_features) > 0:
            print(f'There are {len(categorical_features)} categorical '
                  f'features.')
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
            PCA(random_state=seed)
        )
    ])
    preprocessor.fit(X)
    return preprocessor


def build_neural_network(input_size: int, layer_size: int, n_classes: int,
                         n_layers: int, dropout_rate: float, scale_coeff: float,
                         nn_name: str) -> tf.keras.Model:
    feature_vector = tf.keras.layers.Input(
        shape=(input_size,), dtype=tf.float32,
        name=f'{nn_name}_feature_vector'
    )
    hidden_layer = tf.keras.layers.AlphaDropout(
        rate=dropout_rate,
        seed=random.randint(0, 2147483647),
        name=f'{nn_name}_dropout1'
    )(feature_vector)
    for layer_idx in range(1, (2 * n_layers) // 3 + 1):
        try:
            kernel_initializer = tf.keras.initializers.LecunNormal(
                seed=random.randint(0, 2147483647)
            )
        except:
            kernel_initializer = tf.compat.v1.keras.initializers.lecun_normal(
                seed=random.randint(0, 2147483647)
            )
        hidden_layer = tf.keras.layers.Dense(
            units=layer_size,
            activation='selu',
            kernel_initializer=kernel_initializer,
            bias_initializer='zeros',
            name=f'{nn_name}_dense{layer_idx}'
        )(hidden_layer)
        hidden_layer = tf.keras.layers.AlphaDropout(
            rate=dropout_rate,
            seed=random.randint(0, 2147483647),
            name=f'{nn_name}_dropout{layer_idx + 1}'
        )(hidden_layer)
    try:
        kernel_initializer = tf.keras.initializers.LecunNormal(
            seed=random.randint(0, 2147483647)
        )
    except:
        kernel_initializer = tf.compat.v1.keras.initializers.lecun_normal(
            seed=random.randint(0, 2147483647)
        )
    cls_layer = tf.keras.layers.Dense(
        units=n_classes,
        activation=None,
        use_bias=True,
        kernel_initializer=kernel_initializer,
        name=f'{nn_name}_classification'
    )(hidden_layer)
    for layer_idx in range((2 * n_layers) // 3 + 1, n_layers + 1):
        try:
            kernel_initializer = tf.keras.initializers.LecunNormal(
                seed=random.randint(0, 2147483647)
            )
        except:
            kernel_initializer = tf.compat.v1.keras.initializers.lecun_normal(
                seed=random.randint(0, 2147483647)
            )
        hidden_layer = tf.keras.layers.Dense(
            units=layer_size,
            activation='selu',
            kernel_initializer=kernel_initializer,
            bias_initializer='zeros',
            name=f'{nn_name}_dense{layer_idx}'
        )(hidden_layer)
        hidden_layer = tf.keras.layers.AlphaDropout(
            rate=dropout_rate,
            seed=random.randint(0, 2147483647),
            name=f'{nn_name}_dropout{layer_idx + 1}'
        )(hidden_layer)
    try:
        kernel_initializer = tf.keras.initializers.LecunNormal(
            seed=random.randint(0, 2147483647)
        )
    except:
        kernel_initializer = tf.compat.v1.keras.initializers.lecun_normal(
            seed=random.randint(0, 2147483647)
        )
    output_layer = tf.keras.layers.Dense(
        units=2,
        activation=None,
        use_bias=False,
        kernel_initializer=kernel_initializer,
        name=f'{nn_name}_output'
    )(hidden_layer)
    bayesian_layer = tfp.layers.DistributionLambda(
        lambda t: tfp.distributions.Normal(
            loc=t[..., :1],
            scale=1e-6 + tf.math.softplus((1.0 / scale_coeff) * t[..., 1:])
        ),
        name=f'{nn_name}_distribution'
    )(output_layer)
    neural_network = tf.keras.Model(
        inputs=feature_vector,
        outputs=[bayesian_layer, cls_layer],
        name=nn_name
    )
    negloglik = lambda y, rv_y: -rv_y.log_prob(y)
    radam = tfa.optimizers.RectifiedAdam(learning_rate=3e-4)
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
    losses = {
        f'{nn_name}_distribution': negloglik,
        f'{nn_name}_classification': tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=0.05,
            from_logits=True
        )
    }
    loss_weights = {
        f'{nn_name}_distribution': 1.0,
        f'{nn_name}_classification': 0.5
    }
    metrics = {
        f'{nn_name}_distribution': [
            tf.keras.metrics.MeanAbsoluteError()
        ]
    }
    neural_network.compile(
        optimizer=ranger,
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics
    )
    return neural_network


def predict_with_single_nn(input_data: np.ndarray,
                           model_for_prediction: tf.keras.Model,
                           batch_size: int, output_scaler: StandardScaler) \
        -> Tuple[np.ndarray, np.ndarray]:
    if len(input_data.shape) != 2:
        err_msg = f'The `input_data` argument is wrong! Expected 2-D array, ' \
                  f'got {len(input_data.shape)}-D one!'
        raise ValueError(err_msg)
    n_batches = int(np.ceil(input_data.shape[0] / float(batch_size)))
    pred_mean = []
    pred_std = []
    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(input_data.shape[0], batch_start + batch_size)
        instant_predictions = model_for_prediction(
            input_data[batch_start:batch_end]
        )[0]
        if not isinstance(instant_predictions, tfp.distributions.Distribution):
            err_msg = f'Minibatch {batch_idx}: predictions are wrong! ' \
                      f'Expected tfp.distributions.Distribution, ' \
                      f'got {type(instant_predictions)}.'
            raise ValueError(err_msg)
        instant_mean = instant_predictions.mean()
        instant_std = instant_predictions.stddev()
        del instant_predictions
        if not isinstance(instant_mean, np.ndarray):
            instant_mean = instant_mean.numpy()
        if not isinstance(instant_std, np.ndarray):
            instant_std = instant_std.numpy()
        instant_mean = instant_mean.astype(np.float64).flatten()
        instant_std = instant_std.astype(np.float64).flatten()
        pred_mean.append(instant_mean)
        pred_std.append(instant_std)
        del instant_mean, instant_std
    pred_mean = np.concatenate(pred_mean)
    pred_std = np.concatenate(pred_std)
    pred_mean = output_scaler.inverse_transform(
        pred_mean.reshape((input_data.shape[0], 1))
    ).flatten()
    pred_std *= output_scaler.scale_[0]
    return pred_mean, pred_std * pred_std


def predict_by_ensemble(input_data: np.ndarray,
                        preprocessing: Pipeline,
                        ensemble: List[tf.keras.Model],
                        postprocessing: StandardScaler,
                        minibatch: int) -> np.ndarray:
    num_samples = input_data.shape[0]
    ensemble_size = len(ensemble)
    predictions_of_ensemble = np.empty((ensemble_size, num_samples, 2),
                                       dtype=np.float64)
    X = preprocessing.transform(input_data).astype(np.float32)
    for model_idx, cur_model in enumerate(ensemble):
        y_mean, y_var = predict_with_single_nn(
            input_data=X,
            model_for_prediction=cur_model,
            output_scaler=postprocessing,
            batch_size=minibatch
        )
        predictions_of_ensemble[model_idx, :, 0] = y_mean
        predictions_of_ensemble[model_idx, :, 1] = y_var
    return np.mean(predictions_of_ensemble, axis=0)


class SNNRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, ensemble_size: int = 20,
                 hidden_layer_size: int = 512, n_layers: int = 18,
                 dropout_rate: float = 3e-4,
                 max_epochs: int = 1000, patience: int = 15,
                 minibatch_size: int = 4096, validation_fraction: float = 0.1,
                 verbose: bool = False, clear_session: bool = True,
                 random_seed: Union[int, None] = None):
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
        self.random_seed = random_seed

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
        if hasattr(self, 'deep_ensemble_'):
            del self.deep_ensemble_
        if hasattr(self, 'names_of_deep_ensemble_'):
            del self.names_of_deep_ensemble_
        if hasattr(self, 'preprocessor_'):
            del self.preprocessor_
        if hasattr(self, 'postprocessor_'):
            del self.postprocessor_
        if hasattr(self, 'random_gen_'):
            del self.random_gen_
        gc.collect()
        if self.clear_session:
            tf.keras.backend.clear_session()
        self.random_gen_ = np.random.default_rng(
            seed=(self.random_seed if hasattr(self, 'random_seed') else None)
        )
        self.preprocessor_ = build_preprocessor(
            X_, feature_names,
            verbose=self.verbose,
            seed=self.random_gen_.integers(0, 2147483647)
        )
        if self.verbose:
            print('')
        self.postprocessor_ = StandardScaler(with_mean=True, with_std=True)
        self.postprocessor_.fit(y_.reshape((y_.shape[0], 1)))
        X_ = self.preprocessor_.transform(X_)
        y_ = self.postprocessor_.transform(
            y_.reshape((y_.shape[0], 1))
        ).reshape((y_.shape[0],))
        all_indices = np.array(list(range(X_.shape[0])), dtype=np.int32)
        self.random_gen_.shuffle(all_indices)
        X_ = X_[all_indices]
        y_ = y_[all_indices]
        del all_indices
        gc.collect()
        self.n_classes_ = max(3, min(100, int(round(np.sqrt(y_.shape[0])))))
        y_class_ = KBinsDiscretizer(
            n_bins=self.n_classes_,
            encode='ordinal',
            strategy='kmeans'
        ).fit_transform(y_.reshape((y_.shape[0], 1))).reshape((y_.shape[0],))
        y_class_ = y_class_.astype(np.int32)
        y_class__ = np.zeros((y_class_.shape[0], self.n_classes_),
                             dtype=np.float32)
        for sample_idx, class_idx in enumerate(y_class_):
            y_class__[sample_idx, class_idx] = 1.0
        if self.verbose:
            class_freq = np.zeros((self.n_classes_,), dtype=np.int32)
            for class_idx in y_class_:
                class_freq[class_idx] += 1
            max_num_width = max(map(lambda it: len(str(it)), class_freq))
            for class_idx, freq in enumerate(class_freq):
                print('Class {0:>02}: {1:>{2}} samples'.format(class_idx, freq,
                                                               max_num_width))
            print('')
        splitting = list(StratifiedKFold(
            n_splits=self.ensemble_size,
            shuffle=True,
            random_state=self.random_gen_.integers(0, 2147483647)
        ).split(X_, y_class_))
        self.feature_vector_size_ = X_.shape[1]
        self.deep_ensemble_ = []
        self.names_of_deep_ensemble_ = []
        max_epochs = 1000
        patience = 5
        for alg_id in range(self.ensemble_size):
            model_uuid = str(uuid.uuid1()).split('-')[0]
            model_name = f'snn_regressor_{alg_id + 1}_{model_uuid}'
            regression_output_name = f'{model_name}_distribution'
            self.names_of_deep_ensemble_.append(model_name)
            train_index, test_index = splitting[alg_id]
            steps_per_epoch = len(train_index) // self.minibatch_size
            train_dataset = tf.data.Dataset.from_tensor_slices(
                (
                    X_[train_index],
                    (
                        y_[train_index].flatten(),
                        y_class__[train_index]
                    )
                )
            ).repeat().shuffle(len(train_index)).batch(self.minibatch_size)
            val_dataset = tf.data.Dataset.from_tensor_slices(
                (
                    X_[test_index],
                    (
                        y_[test_index].flatten(),
                        y_class__[test_index]
                    )
                )
            ).batch(self.minibatch_size)
            new_model = build_neural_network(
                input_size=self.feature_vector_size_,
                layer_size=self.hidden_layer_size,
                n_classes=self.n_classes_,
                n_layers=self.n_layers,
                dropout_rate=self.dropout_rate,
                scale_coeff=self.postprocessor_.scale_[0],
                nn_name=model_name
            )
            if self.verbose:
                print('====================')
                print(f'NEURAL NETWORK {alg_id + 1}')
                print('====================')
                print('')
                new_model.summary()
                print('')
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor=f'val_{regression_output_name}_mean_absolute_error',
                    restore_best_weights=True, mode='min',
                    patience=patience, verbose=self.verbose
                )
            ]
            new_model.fit(
                train_dataset, epochs=max_epochs,
                steps_per_epoch=steps_per_epoch,
                callbacks=callbacks, validation_data=val_dataset,
                verbose=(2 if self.verbose else 0)
            )
            del callbacks, train_dataset, val_dataset, train_index, test_index
            self.deep_ensemble_.append(new_model)
            del new_model
            gc.collect()
        return self

    def predict(self, X):
        check_is_fitted(self, ['deep_ensemble_', 'names_of_deep_ensemble_',
                               'preprocessor_', 'postprocessor_',
                               'random_gen_', 'feature_vector_size_',
                               'n_classes_'])
        X_ = check_array(X, force_all_finite='allow-nan',
                         estimator='SNNRegressor')
        X_ = self.preprocessor_.transform(X_)
        y = predict_by_ensemble(
            input_data=X_,
            preprocessing=self.preprocessor_,
            postprocessing=self.postprocessor_,
            ensemble=self.deep_ensemble_,
            minibatch=self.minibatch_size
        )
        return y

    def fit_predict(self, X, y, **kwargs):
        return self.fit(X, y).predict(X)

    def get_params(self, deep=True) -> dict:
        return {
            'ensemble_size': self.ensemble_size,
            'hidden_layer_size': self.hidden_layer_size,
            'n_layers': self.n_layers,
            'dropout_rate': self.dropout_rate,
            'max_epochs': self.max_epochs,
            'patience': self.patience,
            'minibatch_size': self.minibatch_size,
            'verbose': self.verbose,
            'validation_fraction': self.validation_fraction,
            'clear_session': self.clear_session,
            'random_seed': (self.random_seed if hasattr(self, 'random_seed')
                            else None)
        }

    def set_params(self, **params):
        for parameter, value in params.items():
            self.__setattr__(parameter, value)
        return self

    def _copy_deep_ensemble(self) -> List[tf.keras.Model]:
        new_deep_ensemble = []
        re_for_hidden_layer = re.compile(r'_dense\d+$')
        for alg_id in range(self.ensemble_size):
            model_uuid = str(uuid.uuid1()).split('-')[0]
            model_name = f'snn_regressor_{alg_id + 1}_{model_uuid}'
            new_model = build_neural_network(
                input_size=self.feature_vector_size_,
                layer_size=self.hidden_layer_size,
                n_classes=self.n_classes_,
                n_layers=self.n_layers,
                dropout_rate=self.dropout_rate,
                scale_coeff=self.postprocessor_.scale_[0],
                nn_name=model_name
            )
            new_model.build(input_shape=(None, self.feature_vector_size_))
            old_model = self.deep_ensemble_[alg_id]
            n_copied = 0
            for old_layer, new_layer in zip(old_model.layers, new_model.layers):
                if old_layer.name.endswith('_classification'):
                    copy_weight = True
                elif old_layer.name.endswith('_output'):
                    copy_weight = True
                else:
                    if re_for_hidden_layer.search(old_layer.name) is None:
                        copy_weight = False
                    else:
                        copy_weight = True
                if copy_weight:
                    n_copied += 1
                    new_layer.set_weights(old_layer.get_weights())
            if n_copied < 1:
                err_msg = f'The neural network {alg_id + 1} cannot be copied!'
                raise ValueError(err_msg)
            new_deep_ensemble.append(new_model)
        return new_deep_ensemble

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.set_params(
            ensemble_size=self.ensemble_size,
            hidden_layer_size=self.hidden_layer_size,
            n_layers=self.n_layers,
            dropout_rate=self.dropout_rate,
            max_epochs=self.max_epochs,
            patience=self.patience,
            minibatch_size=self.minibatch_size,
            verbose=self.verbose,
            validation_fraction=self.validation_fraction,
            clear_session=self.clear_session,
            random_seed=self.random_seed
        )
        try:
            check_is_fitted(self, ['deep_ensemble_', 'names_of_deep_ensemble_',
                                   'preprocessor_', 'postprocessor_',
                                   'random_gen_', 'feature_vector_size_',
                                   'n_classes_'])
            is_fitted = True
        except:
            is_fitted = False
        if is_fitted:
            result.deep_ensemble_ = self.deep_ensemble_
            result.names_of_deep_ensemble_ = self.names_of_deep_ensemble_
            result.preprocessor_ = self.preprocessor_
            result.postprocessor_ = self.postprocessor_
            result.random_gen_ = self.random_gen_
            result.feature_vector_size_ = self.feature_vector_size_
            result.n_classes_ = self.n_classes_
        return result

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        result.set_params(
            ensemble_size=self.ensemble_size,
            hidden_layer_size=self.hidden_layer_size,
            n_layers=self.n_layers,
            dropout_rate=self.dropout_rate,
            max_epochs=self.max_epochs,
            patience=self.patience,
            minibatch_size=self.minibatch_size,
            verbose=self.verbose,
            validation_fraction=self.validation_fraction,
            clear_session=self.clear_session,
            random_seed=self.random_seed
        )
        try:
            check_is_fitted(self, ['deep_ensemble_', 'names_of_deep_ensemble_',
                                   'preprocessor_', 'postprocessor_',
                                   'random_gen_', 'feature_vector_size_',
                                   'n_classes_'])
            is_fitted = True
        except:
            is_fitted = False
        if is_fitted:
            result.feature_vector_size_ = self.feature_vector_size_
            result.n_classes_ = self.n_classes_
            result.names_of_deep_ensemble_ = copy.deepcopy(
                self.names_of_deep_ensemble_,
                memo=memodict
            )
            result.preprocessor_ = copy.deepcopy(self.preprocessor_,
                                                 memo=memodict)
            result.postprocessor_ = copy.deepcopy(self.postprocessor_,
                                                  memo=memodict)
            if self.random_gen_ is None:
                result.random_gen_ = None
            else:
                result.random_gen_ = copy.deepcopy(self.random_gen_,
                                                   memo=memodict)
            result.deep_ensemble_ = self._copy_deep_ensemble()
        return result

    def __getstate__(self):
        params = self.get_params(True)
        try:
            check_is_fitted(self, ['deep_ensemble_', 'names_of_deep_ensemble_',
                                   'preprocessor_', 'postprocessor_',
                                   'random_gen_', 'feature_vector_size_',
                                   'n_classes_'])
            is_fitted = True
        except:
            is_fitted = False
        if is_fitted:
            params['names_of_deep_ensemble_'] = copy.deepcopy(
                self.names_of_deep_ensemble_
            )
            params['feature_vector_size_'] = self.feature_vector_size_
            params['n_classes_'] = self.n_classes_
            params['preprocessor_'] = copy.deepcopy(self.preprocessor_)
            params['postprocessor_'] = copy.deepcopy(self.postprocessor_)
            if self.random_gen_ is None:
                params['random_gen_'] = None
            else:
                params['random_gen_'] = copy.deepcopy(self.random_gen_)
            params['deep_ensemble_'] = []
            for alg_id in range(self.ensemble_size):
                params['deep_ensemble_'].append(
                    self.deep_ensemble_[alg_id].get_weights()
                )
        return params

    def __setstate__(self, state: dict):
        if not isinstance(state, dict):
            err_msg = f'The `state` is wrong! Expected `{type({"a": 1})}`, ' \
                      f'got `{type(state)}`.'
            raise ValueError(err_msg)
        self.check_params(**state)
        is_fitted = ('deep_ensemble_' in state)
        if is_fitted:
            if 'names_of_deep_ensemble_' not in state:
                err_msg = 'The `names_of_deep_ensemble_` is not found ' \
                          'in the `state`!'
                raise ValueError(err_msg)
            if 'feature_vector_size_' not in state:
                err_msg = 'The `feature_vector_size_` is not found ' \
                          'in the `state`!'
                raise ValueError(err_msg)
            if 'preprocessor_' not in state:
                err_msg = 'The `preprocessor_` is not found in the `state`!'
                raise ValueError(err_msg)
            if 'postprocessor_' not in state:
                err_msg = 'The `postprocessor_` is not found in the `state`!'
                raise ValueError(err_msg)
        self.set_params(**state)
        if hasattr(self, 'deep_ensemble_'):
            del self.deep_ensemble_
        if hasattr(self, 'names_of_deep_ensemble_'):
            del self.names_of_deep_ensemble_
        if hasattr(self, 'preprocessor_'):
            del self.preprocessor_
        if hasattr(self, 'postprocessor_'):
            del self.postprocessor_
        if hasattr(self, 'random_gen_'):
            del self.random_gen_
        gc.collect()
        if self.clear_session:
            tf.keras.backend.clear_session()
        if is_fitted:
            self.feature_vector_size_ = state['feature_vector_size_']
            self.n_classes_ = state['n_classes_']
            self.names_of_deep_ensemble_ = copy.deepcopy(
                state['names_of_deep_ensemble_']
            )
            self.preprocessor_ = copy.deepcopy(state['preprocessor_'])
            self.postprocessor_ = copy.deepcopy(state['postprocessor_'])
            if 'random_gen_' in state:
                self.random_gen_ = copy.deepcopy(state['random_gen_'])
            else:
                self.random_gen_ = np.random.default_rng(
                    seed=(self.random_seed if hasattr(self, 'random_seed')
                          else None)
                )
            self.deep_ensemble_ = []
            for alg_id in range(self.ensemble_size):
                model_name = self.names_of_deep_ensemble_[alg_id]
                new_model = build_neural_network(
                    input_size=self.feature_vector_size_,
                    layer_size=self.hidden_layer_size,
                    n_classes=self.n_classes_,
                    n_layers=self.n_layers,
                    dropout_rate=self.dropout_rate,
                    scale_coeff=self.postprocessor_.scale_[0],
                    nn_name=model_name
                )
                new_model.build(input_shape=(None, self.feature_vector_size_))
                new_model.set_weights(state['deep_ensemble_'][alg_id])
                self.deep_ensemble_.append(new_model)
        return self

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
        if 'random_seed' in kwargs:
            if kwargs['random_seed'] is not None:
                SNNRegressor.check_integer_param('random_seed', **kwargs)
