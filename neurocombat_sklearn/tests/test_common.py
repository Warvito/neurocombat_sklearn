import pytest

from sklearn.utils.estimator_checks import check_estimator

from sklearn_rvm import RVR, RVC

@pytest.mark.parametrize(
    "Estimator", [RVR, RVC]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
