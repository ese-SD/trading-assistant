import pytest
import math
from utils.fees import FeeStructure, Linear, TieredFees, PerShare


# ----------------------
# Base class FeeStructure
# ----------------------

def test_fee_structure_valid_applications():
    # Base class should accept "on_top" and "deducted"
    class Dummy(FeeStructure):
        def compute_fee(self, nb_shares, share_price):
            return 0

    d1 = Dummy("on_top")
    d2 = Dummy("deducted")
    assert d1.application == "on_top"
    assert d2.application == "deducted"


def test_fee_structure_invalid_application():
    # Any application not "on_top"/"deducted" must raise ValueError
    with pytest.raises(ValueError):
        class Dummy(FeeStructure):
            def compute_fee(self, nb_shares, share_price):
                return 0
        Dummy("wrong")


def test_fee_structure_is_abstract():
    # FeeStructure is abstract and cannot be instantiated directly
    with pytest.raises(TypeError):
        FeeStructure("on_top")


# ----------------------
# Linear
# ----------------------

def test_linear_basic_fee_computation():
    # Fee = nb_shares * share_price * linear_fee + flat_fee
    fee_model = Linear("on_top", flat_fee=2.0, linear_fee=0.01)
    nb_shares, share_price = 100, 10
    expected = nb_shares * share_price * 0.01 + 2.0
    assert fee_model.compute_fee(nb_shares, share_price) == pytest.approx(expected)


def test_linear_with_zero_shares():
    # If no shares, fee should equal the flat fee
    fee_model = Linear("deducted", flat_fee=5.0, linear_fee=0.1)
    assert fee_model.compute_fee(0, 100) == pytest.approx(5.0)


def test_linear_large_inputs():
    # Very large inputs should not overflow and should scale linearly
    fee_model = Linear("on_top", flat_fee=0.0, linear_fee=0.001)
    nb_shares, share_price = 10**6, 1000
    result = fee_model.compute_fee(nb_shares, share_price)
    assert math.isfinite(result)
    assert result > 0


# ----------------------
# TieredFees
# ----------------------

def test_tiered_fees_exact_threshold():
    # If nb_shares matches a threshold, bisect_right moves past it
    # Example: at 100 shares, threshold should be 1000
    tiers = {100: 0.01, 1000: 0.005, float("inf"): 0.001}
    fee_model = TieredFees(tiers)
    assert fee_model.compute_fee(100, 50) == 0.005


def test_tiered_fees_between_thresholds():
    # At 150 shares, threshold should be 1000
    tiers = {100: 0.01, 1000: 0.005, float("inf"): 0.001}
    fee_model = TieredFees(tiers)
    assert fee_model.compute_fee(150, 50) == 0.005


def test_tiered_fees_above_all_thresholds():
    # Very large share count should fall into +inf tier
    tiers = {100: 0.01, 1000: 0.005, float("inf"): 0.001}
    fee_model = TieredFees(tiers)
    assert fee_model.compute_fee(10000, 50) == 0.001


def test_tiered_fees_below_first_threshold():
    # Below the smallest threshold, bisect_right should select it
    tiers = {100: 0.01, 1000: 0.005, float("inf"): 0.001}
    fee_model = TieredFees(tiers)
    assert fee_model.compute_fee(10, 50) == 0.01


def test_tiered_fees_unsorted_dict_keys():
    # Dict may not be in order, but thresholds are sorted in __init__
    tiers = {1000: 0.005, float("inf"): 0.001, 100: 0.01}
    fee_model = TieredFees(tiers)
    assert fee_model.compute_fee(150, 50) == 0.005


def test_tiered_fees_missing_infinity_key_raises():
    # If no +inf tier exists, high share counts should fail with IndexError
    tiers = {100: 0.01, 1000: 0.005}
    fee_model = TieredFees(tiers)
    with pytest.raises(IndexError):
        fee_model.compute_fee(10000, 50)


# ----------------------
# PerShare
# ----------------------

def test_per_share_basic_computation():
    # Fee = nb_shares * fee_per_share
    fee_model = PerShare(0.02)
    assert fee_model.compute_fee(100, 10) == pytest.approx(2.0)


def test_per_share_zero_shares():
    fee_model = PerShare(0.5)
    assert fee_model.compute_fee(0, 100) == 0.0


def test_per_share_negative_shares():
    # Negative shares (short position) should produce negative fees
    fee_model = PerShare(0.5)
    assert fee_model.compute_fee(-10, 100) == pytest.approx(-5.0)


def test_per_share_large_inputs():
    # Very large numbers should scale linearly
    fee_model = PerShare(1e-6)
    result = fee_model.compute_fee(10**9, 10)
    assert math.isfinite(result)
    assert result > 0
