"""
ComBat for correcting batch effects in neuroimaging data
"""
import numpy as np
import numpy.linalg as la
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import (check_is_fitted, check_random_state, FLOAT_DTYPES)


__all__ = [
    'CombatModel',
]

#  From https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/preprocessing/label.py
def _encode_numpy(values, uniques=None, encode=False):
    # only used in _encode below, see docstring there for details
    if uniques is None:
        if encode:
            uniques, encoded = np.unique(values, return_inverse=True)
            return uniques, encoded
        else:
            # unique sorts
            return np.unique(values)
    if encode:
        diff = _encode_check_unknown(values, uniques)
        if diff:
            raise ValueError("y contains previously unseen labels: %s"
                             % str(diff))
        encoded = np.searchsorted(uniques, values)
        return uniques, encoded
    else:
        return uniques


def _encode_python(values, uniques=None, encode=False):
    # only used in _encode below, see docstring there for details
    if uniques is None:
        uniques = sorted(set(values))
        uniques = np.array(uniques, dtype=values.dtype)
    if encode:
        table = {val: i for i, val in enumerate(uniques)}
        try:
            encoded = np.array([table[v] for v in values])
        except KeyError as e:
            raise ValueError("y contains previously unseen labels: %s"
                             % str(e))
        return uniques, encoded
    else:
        return uniques


def _encode(values, uniques=None, encode=False):
    """Helper function to factorize (find uniques) and encode values.
    Uses pure python method for object dtype, and numpy method for
    all other dtypes.
    The numpy method has the limitation that the `uniques` need to
    be sorted. Importantly, this is not checked but assumed to already be
    the case. The calling method needs to ensure this for all non-object
    values.
    Parameters
    ----------
    values : array
        Values to factorize or encode.
    uniques : array, optional
        If passed, uniques are not determined from passed values (this
        can be because the user specified categories, or because they
        already have been determined in fit).
    encode : bool, default False
        If True, also encode the values into integer codes based on `uniques`.
    Returns
    -------
    uniques
        If ``encode=False``. The unique values are sorted if the `uniques`
        parameter was None (and thus inferred from the data).
    (uniques, encoded)
        If ``encode=True``.
    """
    if values.dtype == object:
        try:
            res = _encode_python(values, uniques, encode)
        except TypeError:
            raise TypeError("argument must be a string or number")
        return res
    else:
        return _encode_numpy(values, uniques, encode)


def _encode_check_unknown(values, uniques, return_mask=False):
    """
    Helper function to check for unknowns in values to be encoded.
    Uses pure python method for object dtype, and numpy method for
    all other dtypes.
    Parameters
    ----------
    values : array
        Values to check for unknowns.
    uniques : array
        Allowed uniques values.
    return_mask : bool, default False
        If True, return a mask of the same shape as `values` indicating
        the valid values.
    Returns
    -------
    diff : list
        The unique values present in `values` and not in `uniques` (the
        unknown values).
    valid_mask : boolean array
        Additionally returned if ``return_mask=True``.
    """
    if values.dtype == object:
        uniques_set = set(uniques)
        diff = list(set(values) - uniques_set)
        if return_mask:
            if diff:
                valid_mask = np.array([val in uniques_set for val in values])
            else:
                valid_mask = np.ones(len(values), dtype=bool)
            return diff, valid_mask
        else:
            return diff
    else:
        unique_values = np.unique(values)
        diff = list(np.setdiff1d(unique_values, uniques, assume_unique=True))
        if return_mask:
            if diff:
                valid_mask = np.in1d(values, uniques)
            else:
                valid_mask = np.ones(len(values), dtype=bool)
            return diff, valid_mask
        else:
            return diff

# TODO: do not perform the same of LabelEncoder
class CombatModel(BaseEstimator, TransformerMixin):
    """"""
    def __init__(self, copy=True):
        self.copy = copy

    def _reset(self):
        """Reset internal data-dependent state, if necessary.

        __init__ parameters are not touched.
        """

        # Checking one attribute is enough, because they are all set together
        if hasattr(self, 'gamma_star'):
            del self.gamma_star
            del self.delta_star

    def fit(self, X, covars, batch_col, discrete_cols=None, continuous_cols=None):
        """Compute the parameters to perform the harmonization/normalization."""

        # Reset internal state before fitting
        self._reset()

        # Checa de covars eh pandas.
        if not isinstance(covars, pd.DataFrame):
            raise ValueError('covars must be pandas datafrmae -> try: covars = pandas.DataFrame(covars)')

        # Checa de discrete_cols eh lista.
        if not isinstance(discrete_cols, (list, tuple)):
            if discrete_cols is None:
                discrete_cols = []
            else:
                discrete_cols = [discrete_cols]

        # Checa de continuous_cols eh lista.
        if not isinstance(continuous_cols, (list, tuple)):
            if continuous_cols is None:
                continuous_cols = []
            else:
                continuous_cols = [continuous_cols]

        X = check_array(X, copy=self.copy,
                        estimator=self, dtype=FLOAT_DTYPES,
                        force_all_finite="allow-nan")

        # Transforma covars em numpy e float32
        # Armazena covars columns
        covar_labels = np.array(covars.columns)
        covars = np.array(covars, dtype='object')
        for i in range(covars.shape[-1]):
            try:
                covars[:, i] = covars[:, i].astype('float32')
            except:
                pass

        X = X.T  # transpose X to make it (features, samples)... a weird genetics convention..

        ##############################

        # Pega o indice de cada variavel
        # get column indices for relevant variables
        batch_col = np.where(covar_labels == batch_col)[0][0]
        cat_cols = [np.where(covar_labels == c_var)[0][0] for c_var in discrete_cols]
        num_cols = [np.where(covar_labels == n_var)[0][0] for n_var in continuous_cols]

        # convert batch col to integer
        # also return the indices of the unique array
        # TODO: Usar nome diferente
        covars[:, batch_col] = np.unique(covars[:, batch_col], return_inverse=True)[-1]

        # create dictionary that stores batch info
        # Pega lista de unicos e numero de ocorrencias
        # TODO: Verificar se pode por tudo no mesmo return
        batch_levels, sample_per_batch = np.unique(covars[:, batch_col], return_counts=True)

        # TODO: Transformar dict to variables

        batch_levels = batch_levels.astype('int')
        n_batch = len(batch_levels)
        n_sample =  int(covars.shape[0])
        sample_per_batch = sample_per_batch.astype('int')
        # pega lista de lista com indices de cada batch
        batch_info = [list(np.where(covars[:, batch_col] == idx)[0]) for idx in batch_levels]


        # create design matrix
        design = self._make_design_matrix(covars, batch_col, cat_cols, num_cols)

        # standardize X across features
        s_data, s_mean, v_pool = self._standardize_across_features(X, design, n_batch, n_sample, sample_per_batch)

        # fit L/S models and find priors
        LS_dict = self._fit_LS_model_and_find_priors(s_data, design, n_batch, batch_info)

        # find parametric adjustments
        self.gamma_star, self.delta_star = self._find_parametric_adjustments(s_data, LS_dict, batch_info)

        return self

    def transform(self, X):
        """Center and scale the data.
        Parameters
        ----------
        X : {array-like, sparse matrix}
            The data used to scale along the specified axis.
        """
        check_is_fitted(self, 'center_', 'scale_')
        X = check_array(X, accept_sparse=('csr', 'csc'), copy=self.copy,
                        estimator=self, dtype=FLOAT_DTYPES,
                        force_all_finite='allow-nan')

        # TODO:Calcular novo n_sample
        n_sample =  int(covars.shape[0])

        # TODO: Calcular s_data, criar design, stand_mean
        design = None
        stand_mean = np.dot(self.grand_mean.T.reshape((len(self.grand_mean), 1)), np.ones((1, n_sample)))

        s_data = None

        bayes_data = _adjust_data_final(s_data,
                                       design,
                                       self.gamma_star, self.delta_star,
                                       stand_mean, self.var_pooled,
                                       sample_per_batch, n_batch, n_sample, batch_info)

        bayes_data = np.array(bayes_data)

        return X

    def _make_design_matrix(self, Y, batch_col, cat_cols, num_cols):
        """
        Return Matrix containing the following parts:
            - one-hot matrix of batch variable (full)
            - one-hot matrix for each categorical_targts (removing the first column)
            - column for each continuous_cols
        """
        # TODO: USAR O DO sklearn
        def to_categorical(y, nb_classes=None):
            if not nb_classes:
                nb_classes = np.max(y) + 1
            Y = np.zeros((len(y), nb_classes))
            for i in range(len(y)):
                Y[i, y[i]] = 1.
            return Y

        hstack_list = []

        ### batch one-hot ###
        # convert batch column to integer in case it's string
        # TODO: Faz o mesmo que a linha la emcima. Eliminar
        batch = np.unique(Y[:, batch_col], return_inverse=True)[-1]
        batch_onehot = to_categorical(batch, len(np.unique(batch)))
        hstack_list.append(batch_onehot)

        ### categorical one-hots ###
        for cat_col in cat_cols:
            _, cat = np.unique(np.array(Y[:, cat_col]), return_inverse=True)
            # one-hot encoding permitindo tudo zero.
            cat_onehot = to_categorical(cat, len(np.unique(cat)))[:, 1:]
            hstack_list.append(cat_onehot)

        ### numerical vectors ###
        for num_col in num_cols:
            num = np.array(Y[:, num_col], dtype='float32')
            num = num[:, np.newaxis]
            hstack_list.append(num)
        # POE NA HORIZONTAL TODOS AS VARIAVEIS
        design = np.hstack(hstack_list)
        return design

    def _standardize_across_features(self, X, design, n_batch, n_sample, sample_per_batch):


        # https: // github.com / Jfortin1 / ComBatHarmonization / blob / master / Matlab / scripts / combat.m
        # fprintf('[combat] Standardizing Data across features\n')
        # B_hat = inv(design
        # '*design)*design' * dat
        # ';
        # % Standarization
        # Model
        # grand_mean = (n_batches / n_array) * B_hat(1:n_batch,:);
        # var_pooled = ((dat - (design * B_hat)').^2)*repmat(1/n_array,n_array,1);
        # stand_mean = grand_mean'*repmat(1,1,n_array);
        #
        # if not (isempty(design))
        # tmp = design;
        # tmp(:, 1: n_batch) = 0;
        # stand_mean = stand_mean + (tmp * B_hat)
        # ';
        # end
        # s_data = (dat - stand_mean). / (sqrt(var_pooled) * repmat(1, 1, n_array));


        B_hat = np.dot(np.dot(la.inv(np.dot(design.T, design)), design.T), X.T)
        # B.hat < - solve(crossprod(design), tcrossprod(t(design), as.matrix(dat)))

        self.grand_mean = np.dot((sample_per_batch / float(n_sample)).T, B_hat[:n_batch, :])
        # if (! is.null(ref.batch)) {
        # grand.mean < - t(B.hat[ref, ])
        # }
        # else {
        # grand.mean < - crossprod(n.batches / n.array, B.hat[1:n.batch,
        # ])
        # }

        self.var_pooled = np.dot(((X - np.dot(design, B_hat).T) ** 2), np.ones((n_sample, 1)) / float(n_sample))
        # if (! is.null(ref.batch)) {
        # ref.dat < - dat[, batches[[ref]]]
        # var.pooled < - ((ref.dat - t(design[batches[[ref]],
        # ] % * % B.hat)) ^ 2) % * % rep(1 / n.batches[ref], n.batches[ref])
        # }
        # else {
        # var.pooled < - ((dat - t(design % * % B.hat)) ^ 2) % * %
        # rep(1 / n.array, n.array)
        # }

        stand_mean = np.dot(self.grand_mean.T.reshape((len(self.grand_mean), 1)), np.ones((1, n_sample)))
        # stand.mean < - t(grand.mean) % * % t(rep(1, n.array))
        # if (! is.null(dat2)) stand.mean2 < - t(grand.mean) % * % t(rep(1, n.array2))
        tmp = np.array(design.copy())
        tmp[:, :n_batch] = 0
        stand_mean += np.dot(tmp, B_hat).T

        s_data = ((X - stand_mean) / np.dot(np.sqrt(self.var_pooled), np.ones((1, n_sample))))
        # s.data < - (dat - stand.mean) / (sqrt(var.pooled) % * % t(rep(1,
        #                                                               n.array)))
        # if (! is.null(dat2)) {
        # s.data2 < - (dat2 - stand.mean2) / (sqrt(var.pooled) % * % t(rep(1,
        # n.array2)))


        return s_data

    def _aprior(gamma_hat):
        m = np.mean(gamma_hat)
        s2 = np.var(gamma_hat, ddof=1)
        return (2 * s2 + m ** 2) / float(s2)

    def _bprior(gamma_hat):
        m = gamma_hat.mean()
        s2 = np.var(gamma_hat, ddof=1)
        return (m * s2 + m ** 3) / s2

    def _postmean(g_hat, g_bar, n, d_star, t2):
        return (t2 * n * g_hat + d_star * g_bar) / (t2 * n + d_star)

    def _postvar(sum2, n, a, b):
        return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)

    def _fit_LS_model_and_find_priors(s_data, design, n_batch, batch_info):

        batch_design = design[:, :n_batch]
        gamma_hat = np.dot(np.dot(la.inv(np.dot(batch_design.T, batch_design)), batch_design.T), s_data.T)

        delta_hat = []
        for i, batch_idxs in enumerate(batch_info):
            delta_hat.append(np.var(s_data[:, batch_idxs], axis=1, ddof=1))

        gamma_bar = np.mean(gamma_hat, axis=1)
        t2 = np.var(gamma_hat, axis=1, ddof=1)

        a_prior = list(map(_aprior, delta_hat))
        b_prior = list(map(_bprior, delta_hat))

        LS_dict = {}
        LS_dict['gamma_hat'] = gamma_hat
        LS_dict['delta_hat'] = delta_hat
        LS_dict['gamma_bar'] = gamma_bar
        LS_dict['t2'] = t2
        LS_dict['a_prior'] = a_prior
        LS_dict['b_prior'] = b_prior
        return LS_dict

    def _it_sol(sdat, g_hat, d_hat, g_bar, t2, a, b, conv=0.0001):
        n = (1 - np.isnan(sdat)).sum(axis=1)
        g_old = g_hat.copy()
        d_old = d_hat.copy()

        change = 1
        count = 0
        while change > conv:
            g_new = _postmean(g_hat, g_bar, n, d_old, t2)
            sum2 = ((sdat - np.dot(g_new.reshape((g_new.shape[0], 1)), np.ones((1, sdat.shape[1])))) ** 2).sum(
                axis=1)
            d_new = _postvar(sum2, n, a, b)

            change = max((abs(g_new - g_old) / g_old).max(), (abs(d_new - d_old) / d_old).max())
            g_old = g_new  # .copy()
            d_old = d_new  # .copy()
            count = count + 1
        adjust = (g_new, d_new)
        return adjust

    def _find_parametric_adjustments(s_data, LS, batch_info):
        gamma_star, delta_star = [], []
        for i, batch_idxs in enumerate(batch_info):
            temp = _it_sol(s_data[:, batch_idxs], LS['gamma_hat'][i],
                          LS['delta_hat'][i], LS['gamma_bar'][i], LS['t2'][i],
                          LS['a_prior'][i], LS['b_prior'][i])

            gamma_star.append(temp[0])
            delta_star.append(temp[1])

        return np.array(gamma_star), np.array(delta_star)

    def _adjust_data_final(s_data,
                           design,
                           gamma_star, delta_star,
                           stand_mean, var_pooled,
                           sample_per_batch, n_batch, n_sample, batch_info):


        batch_design = design[:, :n_batch]

        bayesdata = s_data
        gamma_star = np.array(gamma_star)
        delta_star = np.array(delta_star)

        for j, batch_idxs in enumerate(batch_info):
            dsq = np.sqrt(delta_star[j, :])
            dsq = dsq.reshape((len(dsq), 1))
            denom = np.dot(dsq, np.ones((1, sample_per_batch[j])))
            numer = np.array(bayesdata[:, batch_idxs] - np.dot(batch_design[batch_idxs, :], gamma_star).T)

            bayesdata[:, batch_idxs] = numer / denom

        vpsq = np.sqrt(var_pooled).reshape((len(var_pooled), 1))
        bayesdata = bayesdata * np.dot(vpsq, np.ones((1, n_sample))) + stand_mean

        return bayesdata

    #
    #     # at fit, convert sparse matrices to csc for optimized computation of
    #     # the quantiles
    #     X = check_array(X, accept_sparse='csc', copy=self.copy, estimator=self,
    #                     dtype=FLOAT_DTYPES, force_all_finite='allow-nan')
    #
    #     q_min, q_max = self.quantile_range
    #     if not 0 <= q_min <= q_max <= 100:
    #         raise ValueError("Invalid quantile range: %s" %
    #                          str(self.quantile_range))
    #
    #     if self.with_centering:
    #         if sparse.issparse(X):
    #             raise ValueError(
    #                 "Cannot center sparse matrices: use `with_centering=False`"
    #                 " instead. See docstring for motivation and alternatives.")
    #         self.center_ = np.nanmedian(X, axis=0)
    #     else:
    #         self.center_ = None
    #
    #     if self.with_scaling:
    #         quantiles = []
    #         for feature_idx in range(X.shape[1]):
    #             if sparse.issparse(X):
    #                 column_nnz_data = X.data[X.indptr[feature_idx]:
    #                                          X.indptr[feature_idx + 1]]
    #                 column_data = np.zeros(shape=X.shape[0], dtype=X.dtype)
    #                 column_data[:len(column_nnz_data)] = column_nnz_data
    #             else:
    #                 column_data = X[:, feature_idx]
    #
    #             quantiles.append(np.nanpercentile(column_data,
    #                                               self.quantile_range))
    #
    #         quantiles = np.transpose(quantiles)
    #
    #         self.scale_ = quantiles[1] - quantiles[0]
    #         self.scale_ = _handle_zeros_in_scale(self.scale_, copy=False)
    #     else:
    #         self.scale_ = None
    #
    #     return self
    #
    #
