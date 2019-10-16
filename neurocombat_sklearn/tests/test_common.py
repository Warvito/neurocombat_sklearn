import pytest

from sklearn.utils.estimator_checks import check_estimator

from neurocombat_sklearn import CombatModel

@pytest.mark.parametrize(
    "Estimator", [CombatModel]
)
def test_all_transformers(Estimator):
    return check_estimator(Estimator)
