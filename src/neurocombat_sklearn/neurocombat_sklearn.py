# Authors: Walter Hugo Lopez Pinaya, Abhishek Roy
# License: MIT
import numpy as np
import numpy.linalg as la
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_array
from sklearn.utils.validation import (
    check_is_fitted,
    check_consistent_length,
    check_X_y,
    FLOAT_DTYPES,
)
import h5py
from typing import Optional, List, Tuple

__all__ = [
    "CombatModel",
]


class CombatModel(BaseEstimator):
    """Harmonize/normalize features using Combat's [1] parametric empirical Bayes framework

    [1] Fortin, Jean-Philippe, et al. "Harmonization of cortical thickness
    measurements across scanners and sites." Neuroimage 167 (2018): 104-120.
    """

    n_sites: int
    sites_names: np.ndarray
    discrete_covariates_used: bool
    continuous_covariates_used: bool
    copy: bool
    site_encoder: OneHotEncoder
    discrete_encoders: list
    beta_hat: np.ndarray
    grand_mean: np.ndarray
    var_pooled: np.ndarray
    gamma_star: np.ndarray
    delta_star: np.ndarray

    def __init__(self, copy: bool = True) -> None:
        self.copy = copy
        self.discrete_covariates_used = False
        self.continuous_covariates_used = False

    def _reset(self) -> None:
        """Reset internal data-dependent state, if necessary.

        __init__ parameters are not touched.
        """

        # Checking one attribute is enough, because they are all set together
        if hasattr(self, "n_sites"):
            del self.n_sites
            del self.sites_names
            del self.site_encoder
            del self.discrete_encoders
            del self.beta_hat
            del self.grand_mean
            del self.var_pooled
            del self.gamma_star
            del self.delta_star

        self.discrete_covariates_used = False
        self.continuous_covariates_used = False

    def fit(
        self,
        data: np.ndarray,
        sites: np.ndarray,
        discrete_covariates: Optional[np.ndarray] = None,
        continuous_covariates: Optional[np.ndarray] = None,
    ) -> "CombatModel":
        """Compute the parameters to perform the harmonization/normalization

        Parameters
        ----------
        data : array-like, shape [n_samples, n_features]
            The data used to compute the per-feature statistics
            used for later harmonization along the acquisition sites.
        sites : array-like, shape [n_samples, 1]
            The target variable for harmonization problems (e.g. acquisition sites or batches).
        discrete_covariates : array-like, shape [n_samples, n_discrete_covariates]
            The covariates which are categorical
            (e.g. schizophrenia patient or healthy control).
        continuous_covariates : array-like, shape [n_samples, n_continuous_covariates]
            The covariates which are continuous
            (e.g. age and clinical scores)
        """
        # Reset internal state before fitting
        self._reset()

        # Validate inputs
        if sites is None:
            raise ValueError("The 'sites' parameter cannot be None.")

        if not isinstance(sites, np.ndarray):
            raise TypeError("The 'sites' parameter must be a numpy array.")

        if discrete_covariates is not None and not isinstance(
            discrete_covariates, np.ndarray
        ):
            raise TypeError(
                "The 'discrete_covariates' parameter must be a numpy array."
            )

        if continuous_covariates is not None and not isinstance(
            continuous_covariates, np.ndarray
        ):
            raise TypeError(
                "The 'continuous_covariates' parameter must be a numpy array."
            )

        # ensure sites are 'S' strings for saving and loading
        if not np.issubdtype(sites.dtype, np.string_):
            try:
                sites = sites.astype("S")
            except Exception as e:
                raise TypeError(
                    e,
                    "The 'sites' parameter must contain byte strings or be convertable to byte strings",
                )

        # Ensure sites is a 1D array for sklearn compatibility
        sites = sites.ravel()

        check_X_y(data, sites, dtype=FLOAT_DTYPES, copy=self.copy)

        if discrete_covariates is not None:
            self.discrete_covariates_used = True
            if not np.issubdtype(discrete_covariates.dtype, np.string_):
                try:
                    discrete_covariates = discrete_covariates.astype("S")
                except Exception as e:
                    raise TypeError(
                        e,
                        "The 'discrete_covariates' parameter must contain byte strings or be convertable to byte strings",
                    )
            discrete_covariates = check_array(
                discrete_covariates, copy=self.copy, dtype=None, estimator=self
            )

        if continuous_covariates is not None:
            if not np.issubdtype(continuous_covariates.dtype, np.number):
                raise TypeError(
                    "The 'continuous_covariates' parameter must contain numeric values."
                )
            self.continuous_covariates_used = True
            continuous_covariates = check_array(
                continuous_covariates,
                copy=self.copy,
                estimator=self,
                dtype=FLOAT_DTYPES,
            )

        # To have a similar code to neuroCombat and Combat original scripts
        # transforms data dims to [n_features, n_samples]
        data = data.T
        sites_names, n_samples_per_site = np.unique(sites, return_counts=True)
        self.sites_names = sites_names
        self.n_sites = len(sites_names)
        n_samples = sites.shape[0]
        idx_per_site = [list(np.where(sites == idx)[0]) for idx in sites_names]
        design = self._make_design_matrix(
            sites, discrete_covariates, continuous_covariates, fitting=True
        )
        standardized_data, _ = self._standardize_across_features(
            data, design, n_samples, n_samples_per_site, fitting=True
        )
        gamma_hat, delta_hat = self._fit_ls_model(
            standardized_data, design, idx_per_site
        )
        gamma_bar, tau_2, a_prior, b_prior = self._find_priors(gamma_hat, delta_hat)
        self.gamma_star, self.delta_star = self._find_parametric_adjustments(
            standardized_data,
            idx_per_site,
            gamma_hat,
            delta_hat,
            gamma_bar,
            tau_2,
            a_prior,
            b_prior,
        )
        return self

    def transform(
        self,
        data: np.ndarray,
        sites: np.ndarray,
        discrete_covariates: Optional[np.ndarray] = None,
        continuous_covariates: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Transform data to harmonized space

        Parameters
        ----------
        data : array-like
            Input data that will be transformed.
        sites : array-like
            Site info of the inputted data
        discrete_covariates : array-like
            The covariates which are categorical
        continuous_covariates : array-like
            The covariates which are continuous
        """

        check_is_fitted(self, "n_sites")

        # Validate inputs
        if sites is None:
            raise ValueError("The 'sites' parameter cannot be None.")

        if not isinstance(sites, np.ndarray):
            raise TypeError("The 'sites' parameter must be a numpy array.")

        if discrete_covariates is not None and not isinstance(
            discrete_covariates, np.ndarray
        ):
            raise TypeError(
                "The 'discrete_covariates' parameter must be a numpy array."
            )

        if continuous_covariates is not None and not isinstance(
            continuous_covariates, np.ndarray
        ):
            raise TypeError(
                "The 'continuous_covariates' parameter must be a numpy array."
            )

        # ensure sites are 'S' strings for saving and loading

        if not np.issubdtype(sites.dtype, np.string_):
            try:
                sites = sites.astype("S")
            except Exception as e:
                raise TypeError(
                    e,
                    "The 'sites' parameter must contain byte strings or be convertable to byte strings",
                )

        # Ensure sites is a 1D array for sklearn compatibility
        sites = sites.ravel()

        check_X_y(data, sites, dtype=FLOAT_DTYPES, copy=self.copy)

        discrete_covariates_used = discrete_covariates is not None
        assert discrete_covariates_used == self.discrete_covariates_used, (
            f"The model has discrete_covariates_used:{self.discrete_covariates_used}\n",
            f"Transform was given discrete_covariates: {discrete_covariates_used}",
            "These must match",
        )

        if discrete_covariates_used:
            if not np.issubdtype(discrete_covariates.dtype, np.string_):
                try:
                    discrete_covariates = discrete_covariates.astype("S")
                except Exception as e:
                    raise TypeError(
                        e,
                        "The 'discrete_covariates' parameter must contain byte strings or be convertable to byte strings",
                    )
            discrete_covariates = check_array(
                discrete_covariates, copy=self.copy, dtype=None, estimator=self
            )

        continuous_covariates_used = continuous_covariates is not None
        assert continuous_covariates_used == self.continuous_covariates_used, (
            f"The model has continuous_covariates_used:{self.continuous_covariates_used}\n",
            f"Transform was given continuous_covariates: {continuous_covariates_used}",
            "These must match",
        )
        if continuous_covariates_used:
            if not np.issubdtype(continuous_covariates.dtype, np.number):
                raise TypeError(
                    "The 'continuous_covariates' parameter must contain numeric values."
                )
            continuous_covariates = check_array(
                continuous_covariates,
                copy=self.copy,
                estimator=self,
                dtype=FLOAT_DTYPES,
            )

        check_consistent_length(data, sites)

        # To have a similar code to neuroCombat and Combat original scripts
        data = data.T

        new_data_sites_name = np.unique(sites)

        # Check all sites from new_data were seen
        if not all(site_name in self.sites_names for site_name in new_data_sites_name):
            raise ValueError(
                "There is a site unseen during the fit method in the data."
            )

        n_samples = sites.shape[0]
        n_samples_per_site = np.array(
            [np.sum(sites == site_name) for site_name in self.sites_names]
        )
        idx_per_site = [
            list(np.where(sites == site_name)[0]) for site_name in self.sites_names
        ]

        design = self._make_design_matrix(
            sites, discrete_covariates, continuous_covariates, fitting=False
        )

        standardized_data, standardized_mean = self._standardize_across_features(
            data, design, n_samples, n_samples_per_site, fitting=False
        )

        bayes_data = self._adjust_data_final(
            standardized_data,
            design,
            standardized_mean,
            n_samples_per_site,
            n_samples,
            idx_per_site,
        )

        return bayes_data.T

    def fit_transform(
        self, data: np.ndarray, sites: np.ndarray, *args: np.ndarray
    ) -> np.ndarray:
        """Fit to data, then transform it"""
        return self.fit(data, sites, *args).transform(data, sites, *args)

    def save_model(self, filepath: str) -> None:
        """Save the model parameters to an HDF5 file."""
        with h5py.File(filepath, "w") as f:
            f.create_dataset("n_sites", data=self.n_sites)
            f.create_dataset("sites_names", data=self.sites_names.astype("S"))
            f.create_dataset(
                "discrete_covariates_used", data=self.discrete_covariates_used
            )
            f.create_dataset(
                "continuous_covariates_used", data=self.continuous_covariates_used
            )
            f.create_dataset("beta_hat", data=self.beta_hat)
            f.create_dataset("grand_mean", data=self.grand_mean)
            f.create_dataset("var_pooled", data=self.var_pooled)
            f.create_dataset("gamma_star", data=self.gamma_star)
            f.create_dataset("delta_star", data=self.delta_star)

            # Save site_encoder attributes
            if hasattr(self, "site_encoder") and self.site_encoder is not None:
                site_encoder_categories = self.site_encoder.categories_[0].astype("S")
                f.create_dataset(
                    "site_encoder_categories", data=site_encoder_categories
                )

            # Save discrete_encoders attributes
            if (
                hasattr(self, "discrete_encoders")
                and self.discrete_encoders is not None
            ):
                for i, encoder in enumerate(self.discrete_encoders):
                    categories = encoder.categories_[0]
                    categories = categories.astype("S")
                    f.create_dataset(
                        f"discrete_encoder_{i}_categories", data=categories
                    )

    @classmethod
    def load_model(cls, filepath: str) -> "CombatModel":
        """Load the model parameters from an HDF5 file."""
        with h5py.File(filepath, "r") as f:
            model = cls()
            model.n_sites = f["n_sites"][()]
            model.sites_names = f["sites_names"][:].astype("S")
            model.discrete_covariates_used = f["discrete_covariates_used"][()]
            model.continuous_covariates_used = f["continuous_covariates_used"][()]
            model.beta_hat = f["beta_hat"][:]
            model.grand_mean = f["grand_mean"][:]
            model.var_pooled = f["var_pooled"][:]
            model.gamma_star = f["gamma_star"][:]
            model.delta_star = f["delta_star"][:]

            # Load site_encoder attributes
            if "site_encoder_categories" in f:
                site_encoder_categories = f["site_encoder_categories"][:].astype("S")
                model.site_encoder = OneHotEncoder(sparse_output=False)
                model.site_encoder.fit(site_encoder_categories.reshape(-1, 1))

            # Load discrete_encoders attributes
            model.discrete_encoders = []
            i = 0
            while f.get(f"discrete_encoder_{i}_categories") is not None:
                categories = f[f"discrete_encoder_{i}_categories"][:].astype("S")
                encoder = OneHotEncoder(sparse_output=False)
                encoder.fit(categories.reshape(-1, 1))
                model.discrete_encoders.append(encoder)
                i += 1

        return model

    def _make_design_matrix(
        self,
        sites: np.ndarray,
        discrete_covariates: Optional[np.ndarray],
        continuous_covariates: Optional[np.ndarray],
        fitting: bool = False,
    ) -> np.ndarray:
        """Method to create a design matrix that contain:

            - One-hot encoding of the sites [n_samples, n_sites]
            - One-hot encoding of each discrete covariates (removing
            the first column) [n_samples, (n_discrete_covariate_names-1) * n_discrete_covariates]
            - Each continuous covariates

        Parameters
        ----------
        sites : array-like
        discrete_covariates : array-like
        continuous_covariates : array-like
        fitting : boolean, default is False

        Returns
        -------
        design : array-like
            The design matrix.
        """
        design_list = []

        # Ensure sites is reshaped to 2D for OneHotEncoder compatibility
        sites = sites.reshape(-1, 1)

        # Sites
        if fitting:
            self.site_encoder = OneHotEncoder(sparse_output=False)
            self.site_encoder.fit(sites)

        sites_design = self.site_encoder.transform(sites)
        design_list.append(sites_design)

        # Discrete covariates
        if discrete_covariates is not None:
            n_discrete_covariates = discrete_covariates.shape[1]

            if fitting:
                self.discrete_encoders = []

                for i in range(n_discrete_covariates):
                    discrete_encoder = OneHotEncoder(sparse_output=False)
                    discrete_encoder.fit(discrete_covariates[:, i][:, np.newaxis])
                    self.discrete_encoders.append(discrete_encoder)

            for i in range(n_discrete_covariates):
                discrete_encoder = self.discrete_encoders[i]
                discrete_covariate_one_hot = discrete_encoder.transform(
                    discrete_covariates[:, i][:, np.newaxis]
                )
                discrete_covariate_design = discrete_covariate_one_hot[:, 1:]
                design_list.append(discrete_covariate_design)

        # Continuous covariates
        if continuous_covariates is not None:
            design_list.append(continuous_covariates)

        design = np.hstack(design_list)
        return design

    def _standardize_across_features(
        self,
        data: np.ndarray,
        design: np.ndarray,
        n_samples: int,
        n_samples_per_site: np.ndarray,
        fitting: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Standardization of the features

        The magnitude of the features could create bias in the empirical Bayes estimates of the prior distribution.
        To avoid this, the features are standardized to all of them have similar overall mean and variance.

        Parameters
        ----------
        data : array-like
        design : array-like
        n_samples : integer
        n_samples_per_site : list of integer
        fitting : boolean, default is False
            Indicates if this method is executed inside the
            fit method (in order to save the parameters to use later).

        Returns
        -------
        standardized_data : array-like
        standardized_mean : array-like
            Standardized mean used during the process
        """
        if fitting:
            self.beta_hat = np.dot(
                np.dot(la.inv(np.dot(design.T, design)), design.T), data.T
            )

            # Standardization Model
            self.grand_mean = np.dot(
                (n_samples_per_site / float(n_samples)).T,
                self.beta_hat[: self.n_sites, :],
            )
            self.var_pooled = np.dot(
                ((data - np.dot(design, self.beta_hat).T) ** 2),
                np.ones((n_samples, 1)) / float(n_samples),
            )

        standardized_mean = np.dot(
            self.grand_mean.T[:, np.newaxis], np.ones((1, n_samples))
        )

        tmp = np.array(design.copy())
        tmp[:, : self.n_sites] = 0
        standardized_mean += np.dot(tmp, self.beta_hat).T

        standardized_data = (data - standardized_mean) / np.dot(
            np.sqrt(self.var_pooled), np.ones((1, n_samples))
        )

        return standardized_data, standardized_mean

    def _fit_ls_model(
        self,
        standardized_data: np.ndarray,
        design: np.ndarray,
        idx_per_site: List[List[int]],
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Location and scale (L/S) adjustments

        Parameters
        ----------
        standardized_data : array-like
        design : array-like
        idx_per_site : list of list of integer
        """
        site_design = design[:, : self.n_sites]
        gamma_hat = np.dot(
            np.dot(la.inv(np.dot(site_design.T, site_design)), site_design.T),
            standardized_data.T,
        )

        delta_hat = []
        for i, site_idxs in enumerate(idx_per_site):
            delta_hat.append(np.var(standardized_data[:, site_idxs], axis=1, ddof=1))

        return gamma_hat, delta_hat

    def _find_priors(
        self, gamma_hat: np.ndarray, delta_hat: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, List[float], List[float]]:
        """Compute a and b priors"""
        gamma_bar = np.mean(gamma_hat, axis=1)
        tau_2 = np.var(gamma_hat, axis=1, ddof=1)

        def aprior_fn(gamma_hat):
            m = np.mean(gamma_hat)
            s2 = np.var(gamma_hat, ddof=1, dtype=np.float32)
            return (2 * s2 + m**2) / s2

        a_prior = list(map(aprior_fn, delta_hat))

        def bprior_fn(gamma_hat):
            m = np.mean(gamma_hat)
            s2 = np.var(gamma_hat, ddof=1, dtype=np.float32)
            return (m * s2 + m**3) / s2

        b_prior = list(map(bprior_fn, delta_hat))

        return gamma_bar, tau_2, a_prior, b_prior

    def _find_parametric_adjustments(
        self,
        standardized_data: np.ndarray,
        idx_per_site: List[List[int]],
        gamma_hat: np.ndarray,
        delta_hat: List[np.ndarray],
        gamma_bar: np.ndarray,
        tau_2: np.ndarray,
        a_prior: List[float],
        b_prior: List[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute empirical Bayes site/batch effect parameter estimates using parametric empirical priors"""

        gamma_star, delta_star = [], []

        for i, site_idxs in enumerate(idx_per_site):
            gamma_hat_adjust, delta_hat_adjust = self._iteration_solver(
                standardized_data[:, site_idxs],
                gamma_hat[i],
                delta_hat[i],
                gamma_bar[i],
                tau_2[i],
                a_prior[i],
                b_prior[i],
            )

            gamma_star.append(gamma_hat_adjust)
            delta_star.append(delta_hat_adjust)

        return np.array(gamma_star), np.array(delta_star)

    def _iteration_solver(
        self,
        standardized_data: np.ndarray,
        gamma_hat: np.ndarray,
        delta_hat: np.ndarray,
        gamma_bar: float,
        tau_2: float,
        a_prior: float,
        b_prior: float,
        convergence: float = 0.0001,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute iterative method to find the the parametric site/batch effect adjustments"""
        n = (1 - np.isnan(standardized_data)).sum(axis=1)
        gamma_hat_old = gamma_hat.copy()
        delta_hat_old = delta_hat.copy()

        def postmean(gamma_hat, gamma_bar, n, delta_star, tau_2):
            return (tau_2 * n * gamma_hat + delta_star * gamma_bar) / (
                tau_2 * n + delta_star
            )

        def postvar(sum_2, n, a_prior, b_prior):
            return (0.5 * sum_2 + b_prior) / (n / 2.0 + a_prior - 1.0)

        change = 1
        count = 0

        while change > convergence:
            gamma_hat_new = postmean(gamma_hat, gamma_bar, n, delta_hat_old, tau_2)
            sum_2 = (
                (
                    standardized_data
                    - np.dot(
                        gamma_hat_new[:, np.newaxis],
                        np.ones((1, standardized_data.shape[1])),
                    )
                )
                ** 2
            ).sum(axis=1)

            delta_hat_new = postvar(sum_2, n, a_prior, b_prior)

            change = max(
                (abs(gamma_hat_new - gamma_hat_old) / gamma_hat_old).max(),
                (abs(delta_hat_new - delta_hat_old) / delta_hat_old).max(),
            )

            gamma_hat_old = gamma_hat_new
            delta_hat_old = delta_hat_new

            count = count + 1

        return gamma_hat_new, delta_hat_new

    def _adjust_data_final(
        self,
        standardized_data: np.ndarray,
        design: np.ndarray,
        standardized_mean: np.ndarray,
        n_samples_per_site: np.ndarray,
        n_samples: int,
        idx_per_site: List[List[int]],
    ) -> np.ndarray:
        """Compute the harmonized/normalized data"""
        n_sites = self.n_sites
        var_pooled = self.var_pooled
        gamma_star = self.gamma_star
        delta_star = self.delta_star

        site_design = design[:, :n_sites]

        bayes_data = standardized_data

        for j, site_idxs in enumerate(idx_per_site):
            denominator = np.dot(
                np.sqrt(delta_star[j, :])[:, np.newaxis],
                np.ones((1, n_samples_per_site[j])),
            )
            numerator = (
                bayes_data[:, site_idxs]
                - np.dot(site_design[site_idxs, :], gamma_star).T
            )

            bayes_data[:, site_idxs] = numerator / denominator

        bayes_data = (
            bayes_data * np.dot(np.sqrt(var_pooled), np.ones((1, n_samples)))
            + standardized_mean
        )

        return bayes_data
