import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_raises

from ...sims import linear, rot_2samp
from .. import KSample
from ...independence import CCA, Dcorr


class TestKSample:
    @pytest.mark.parametrize(
        "n, obs_stat, obs_pvalue, indep_test",
        [(1000, 4.28e-7, 1.0, CCA.__name__), (100, 8.24e-5, 0.001, Dcorr.__name__)],
    )
    def test_twosamp_linear_oned(self, n, obs_stat, obs_pvalue, indep_test):
        np.random.seed(123456789)
        x, y = rot_2samp(linear, n, 1)
        stat, pvalue = KSample(indep_test).test(x, y)

        assert_almost_equal(stat, obs_stat, decimal=1)
        assert_almost_equal(pvalue, obs_pvalue, decimal=1)


class TestKSampleErrorWarn:
    """Tests errors and warnings derived from MGC."""

    def test_error_notndarray(self):
        # raises error if x or y is not a ndarray
        x = np.arange(20)
        y = [5] * 20
        z = np.arange(5)
        assert_raises(ValueError, KSample(Dcorr.__name__).test, x, y, z)

    def test_error_shape(self):
        # raises error if number of samples different (n)
        x = np.arange(100).reshape(25, 4)
        y = x.reshape(10, 10)
        z = x
        assert_raises(ValueError, KSample(Dcorr.__name__).test, x, y, z)

    def test_error_lowsamples(self):
        # raises error if samples are low (< 3)
        x = np.arange(3)
        y = np.arange(3)
        assert_raises(ValueError, KSample(CCA.__name__).test, x, y)

    def test_error_nans(self):
        # raises error if inputs contain NaNs
        x = np.arange(20, dtype=float)
        x[0] = np.nan
        assert_raises(ValueError, KSample(CCA.__name__).test, x, x)

        y = np.arange(20)
        assert_raises(ValueError, KSample(CCA.__name__).test, x, y)
