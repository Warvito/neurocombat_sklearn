import tempfile
import numpy as np
import pytest  # Keep pytest for running tests
from neurocombat_sklearn import CombatModel


def test_combat_model_fit_and_transform():
    np.random.seed(42)  # Set seed for reproducibility

    # Generate random data
    n_samples = 100
    n_features = 10

    data = np.random.rand(n_samples, n_features)
    sites = np.random.choice(["Site1", "Site2", "Site3"], size=n_samples).reshape(-1, 1)
    age = np.random.randint(20, 60, size=n_samples).reshape(-1, 1)
    smoker = np.random.choice(["True", "False"], size=n_samples).reshape(-1, 1)

    # Initialize and fit CombatModel
    combat = CombatModel()
    combat.fit(data, sites, discrete_covariates=smoker, continuous_covariates=age)

    # Transform the data
    transformed_data = combat.transform(
        data, sites, discrete_covariates=smoker, continuous_covariates=age
    )

    # Check that transformed data has the same shape as input data
    assert transformed_data.shape == data.shape, "Transformed data shape mismatch"

    # Check that transformed data is not identical to input data
    assert not np.allclose(
        data, transformed_data
    ), "Transformed data should differ from input data"


def test_combat_model_save_and_load():
    np.random.seed(42)  # Set seed for reproducibility

    # Generate random data
    n_samples = 100
    n_features = 10

    data = np.random.rand(n_samples, n_features)
    sites = np.random.choice(["Site1", "Site2", "Site3"], size=n_samples).reshape(-1, 1)
    age = np.random.randint(20, 60, size=n_samples).reshape(-1, 1)
    smoker = np.random.choice(["True", "False"], size=n_samples).reshape(-1, 1)

    # Initialize and fit CombatModel
    combat = CombatModel()
    combat.fit(data, sites, discrete_covariates=smoker, continuous_covariates=age)

    # Transform the data
    transformed_data = combat.transform(
        data, sites, discrete_covariates=smoker, continuous_covariates=age
    )

    # Use tempfile to manage the temporary file
    with tempfile.NamedTemporaryFile() as temp_file:
        temp_filepath = temp_file.name

        try:
            combat.save_model(temp_filepath)

            # Load the model from the temporary file
            loaded_combat = CombatModel.load_model(temp_filepath)

            # Transform the data with the loaded model
            loaded_transformed_data = loaded_combat.transform(
                data,
                sites,
                discrete_covariates=smoker,
                continuous_covariates=age,
            )

            # Check that the transformed data matches the previously transformed data
            assert np.allclose(
                transformed_data, loaded_transformed_data
            ), "Transformed data mismatch after loading model"

        except Exception as e:
            pytest.fail(f"Failed to save and load model: {e}")


def test_missing_covariates():
    np.random.seed(42)
    data = np.random.rand(100, 10)
    sites = np.random.choice(["Site1", "Site2"], size=100).reshape(-1, 1)

    combat = CombatModel()
    combat.fit(data, sites)  # No covariates provided
    transformed_data = combat.transform(data, sites)

    assert transformed_data.shape


def test_invalid_input_data():
    np.random.seed(42)
    data = np.random.rand(100, 10)
    sites = np.random.choice(["Site1", "Site2"], size=100).reshape(-1, 1)
    combat = CombatModel()

    # Test with None as input
    with pytest.raises(ValueError):
        combat.fit(None, sites)

    with pytest.raises(ValueError):
        combat.fit(data, None)

    # Test with mismatched dimensions
    with pytest.raises(ValueError):
        combat.fit(data, sites[:50])


def test_missing_data():
    np.random.seed(42)
    data = np.random.rand(100, 10)
    data[0, 0] = np.nan  # Introduce missing value
    sites = np.random.choice(["Site1", "Site2"], size=100).reshape(-1, 1)
    combat = CombatModel()

    # Test fitting with missing data
    with pytest.raises(ValueError):
        combat.fit(data, sites)


def test_unsupported_covariate_types():
    np.random.seed(42)
    data = np.random.rand(100, 10)
    sites = np.random.choice(["Site1", "Site2"], size=100).reshape(-1, 1)
    invalid_covariate = np.array(["InvalidType"] * 100).reshape(
        -1, 1
    )  # Unsupported type
    combat = CombatModel()

    # Test with unsupported covariate type
    with pytest.raises(TypeError):
        combat.fit(data, sites, continuous_covariates=invalid_covariate)


def test_unseen_discreet_covariate_in_transform():
    np.random.seed(42)  # Set seed for reproducibility

    # Generate random data
    n_samples = 100
    n_features = 10

    data = np.random.rand(n_samples, n_features)
    sites = np.random.choice(["Site1", "Site2", "Site3"], size=n_samples).reshape(-1, 1)
    age = np.random.randint(20, 60, size=n_samples).reshape(-1, 1)
    smoker = np.random.choice(["True", "False"], size=n_samples).reshape(-1, 1)
    # Initialize and fit CombatModel
    combat = CombatModel()
    combat.fit(data, sites, discrete_covariates=smoker, continuous_covariates=age)
    # ADDING UNSEEN COVARIATE
    smoker = np.random.choice(["True", "False", "UNSEEN"], size=n_samples).reshape(
        -1, 1
    )
    # Transform the data
    with pytest.raises(ValueError):
        combat.transform(
            data, sites, discrete_covariates=smoker, continuous_covariates=age
        )


def test_unseen_site_in_transform():
    np.random.seed(42)  # Set seed for reproducibility

    # Generate random data
    n_samples = 100
    n_features = 10

    data = np.random.rand(n_samples, n_features)
    sites = np.random.choice(["Site1", "Site2", "Site3"], size=n_samples).reshape(-1, 1)
    age = np.random.randint(20, 60, size=n_samples).reshape(-1, 1)
    smoker = np.random.choice(["True", "False"], size=n_samples).reshape(-1, 1)
    # Initialize and fit CombatModel
    combat = CombatModel()
    combat.fit(data, sites, discrete_covariates=smoker, continuous_covariates=age)
    # ADDING UNSEEN SITE
    sites = np.random.choice(
        ["Site1", "Site2", "Site3", "UNSEEN SITE"], size=n_samples
    ).reshape(-1, 1)
    # Transform the data
    with pytest.raises(ValueError):
        combat.transform(
            data, sites, discrete_covariates=smoker, continuous_covariates=age
        )
