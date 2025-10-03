import pytest
from datetime import datetime, timezone
from backtester.slippage_modeling import SlippageContext, SlippageModel
import pandas as pd




data=[
    [254.219894,  254.500000,  253.009995,  254.000000,  6306062],
    [253.470001,  254.240005,  253.309998,  254.205002,  2834864],
    [253.660004,  254.070007,  253.119995,  253.466507,  2275242],
    [253.399994,  253.779907,  253.210007,  253.669998,  1744854],
    [254.570007,  254.615005,  253.130005,  253.389999,  2297052],
    [253.990005,  254.600006,  253.610001,  254.570007,  2057798],
    [254.350006,  254.830002,  253.940002,  253.990005,  2513071],
    [254.886200,  255.919006,  253.960007,  255.000000,  5093340],
    [255.500000,  255.520004,  254.770004,  254.910004,  1159477]]

dates = pd.to_datetime([
"2025-09-29 13:30:00+00:00",
"2025-09-29 14:30:00+00:00",
"2025-09-29 15:30:00+00:00",
"2025-09-29 16:30:00+00:00",
"2025-09-29 17:30:00+00:00",
"2025-09-29 18:30:00+00:00", 
"2025-09-29 19:30:00+00:00",
"2025-09-30 13:30:00+00:00",
"2025-09-30 14:30:00+00:00"
])


arrays = [
    ["Close", "High", "Low", "Open", "Volume"],
    ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL"], 
]
columns = pd.MultiIndex.from_arrays(arrays, names=("Price", "Ticker"))

ochlv = pd.DataFrame(data, index=dates, columns=columns)



def mock_spread(context):
    return 1


def mock_mi(context):
    return 0.001


def mock_queue(context):
    return 0.1


def mock_auction_prenium(context):
    return 0.05




# -------------------------------
# Tests for SlippageContext
# -------------------------------

def test_initialization_with_required_fields():
    """SlippageContext should correctly store required fields."""
    ctx = SlippageContext(price=100.5, order_size=50, timestamp="2025-09-25 10:00:00")
    assert ctx.price == 100.5
    assert ctx.order_size == 50
    assert ctx.timestamp == "2025-09-25 10:00:00"


def test_initialization_with_extra_dict():
    """SlippageContext should store the provided extra dictionary intact."""
    extra_data = {"OCHLV": [{"High": 101, "Low": 99}]}
    ctx = SlippageContext(price=200.0, order_size=10, timestamp="2025-09-25", extra=extra_data)
    # Ensure the dict is stored
    assert ctx.extra == extra_data
    # Ensure it's the same object (not copied) so user can update if needed
    ctx.extra["new_key"] = "value"
    assert "new_key" in extra_data


def test_default_extra_is_empty_dict_and_independent():
    """
    If no extra is provided, it should default to an empty dict.
    Each instance must get its own dict, not a shared one.
    """
    ctx1 = SlippageContext(price=1.0, order_size=1, timestamp="t1")
    ctx2 = SlippageContext(price=2.0, order_size=2, timestamp="t2")

    assert ctx1.extra == {}
    assert ctx2.extra == {}
    # Modify one instance's extra; should not affect the other
    ctx1.extra["foo"] = "bar"
    assert "foo" in ctx1.extra
    assert "foo" not in ctx2.extra


def test_mutability_isolation():
    """
    Modifying the extra dict of one instance should not affect another instance.
    This protects against hidden shared state.
    """
    ctx1 = SlippageContext(price=10, order_size=5, timestamp="t1", extra={"alpha": 1})
    ctx2 = SlippageContext(price=20, order_size=10, timestamp="t2", extra={"beta": 2})

    ctx1.extra["gamma"] = 3
    assert "gamma" in ctx1.extra
    assert "gamma" not in ctx2.extra
    assert ctx2.extra == {"beta": 2}





# -------------------------------
# Tests for SlippageModeling
# -------------------------------


"""
tests a faire:
-qd on specifie rien dans le constructeur, ca renvoi le prix de base
-tester chaque composante individuellement
-


-tester params



"""



# ---------------------------------------------------
# Helpers for dummy component functions
# ---------------------------------------------------

def const_func_factory(value):
    """Return a function that always returns a fixed value, ignoring context."""
    def f(context, **kwargs):
        return value
    return f


def identity_func(context, **kwargs):
    """Return a marker value from context for testing call propagation."""
    return context.get("marker", 1)


# ---------------------------------------------------
# Constructor tests
# ---------------------------------------------------

def test_constructs_with_no_components():
    """If no functions are provided, all components should be (None, coeff)."""
    model = SlippageModel()
    assert model.spread[0] is None
    assert model.market_impact[0] is None
    assert model.queue[0] is None
    assert model.auction_premium[0] is None


def test_constructs_with_one_component_spread():
    """Only spread provided: spread is partial, others are None."""
    model = SlippageModel(spread_fct=identity_func, spread_coeff=2.0)
    assert callable(model.spread[0])
    assert model.spread[1] == 2.0
    assert model.market_impact[0] is None


def test_constructs_with_all_components_and_params():
    """All components with params should be wrapped into partials."""
    params = {
        "spread": {"window": 10},
        "mi": {"factor": 0.1},
        "queue": {"depth": 5},
        "ap": {"premium": 2}
    }
    model = SlippageModel(
        spread_fct=identity_func, spread_coeff=1.0,
        market_impact_fct=identity_func, MI_coeff=2.0,
        queue_fct=identity_func, queue_coeff=3.0,
        auct_prenium_fct=identity_func, AP_coeff=4.0,
        params=params
    )
    for comp, coeff in [model.spread, model.market_impact, model.queue, model.auction_premium]:
        assert callable(comp)
    assert model.spread[1] == 1.0
    assert model.market_impact[1] == 2.0
    assert model.queue[1] == 3.0
    assert model.auction_premium[1] == 4.0


def test_params_none_handled():
    """params=None should not crash, should default to {}."""
    model = SlippageModel(spread_fct=identity_func, spread_coeff=1.0, params=None)
    assert callable(model.spread[0])


def test_params_missing_keys_are_ignored():
    """Missing keys in params should be tolerated (defaults to {})."""
    params = {"spread": {"window": 5}}
    model = SlippageModel(spread_fct=identity_func, spread_coeff=1.0, params=params)
    assert callable(model.spread[0])
    # others should be None
    assert model.market_impact[0] is None


def test_params_non_dict_entry_raises():
    """Non-dict param entry should cause a TypeError when calling the partial."""
    params = {"spread": None}  # will cause partial(..., **None)
    with pytest.raises(TypeError):
        SlippageModel(spread_fct=identity_func, spread_coeff=1.0, params=params)


def test_coefficients_accept_various_values():
    """Coefficients can be ints, floats, negatives without issue."""
    model = SlippageModel(spread_fct=identity_func, spread_coeff=-5)
    assert model.spread[1] == -5


# ---------------------------------------------------
# Compute_fill_price happy-path tests
# ---------------------------------------------------

def test_no_components_returns_base_price():
    model = SlippageModel()
    context = {"price": 100.0}
    assert model.compute_fill_price(context) == 100.0


def test_single_component_adds_to_price():
    model = SlippageModel(spread_fct=const_func_factory(2), spread_coeff=3.0)
    context = {"price": 100.0}
    # Expected: 100 + 3 * 2 = 106
    assert model.compute_fill_price(context) == 106.0


def test_all_components_add_contributions():
    model = SlippageModel(
        spread_fct=const_func_factory(1), spread_coeff=1,
        market_impact_fct=const_func_factory(2), MI_coeff=2,
        queue_fct=const_func_factory(3), queue_coeff=3,
        auct_prenium_fct=const_func_factory(4), AP_coeff=4,
    )
    context = {"price": 100.0}
    # Expected = 100 + 1*1 + 2*2 + 3*3 + 4*4
    assert model.compute_fill_price(context) == 100 + 1 + 4 + 9 + 16


def test_components_returning_zero_dont_change_result():
    model = SlippageModel(spread_fct=const_func_factory(0), spread_coeff=10)
    context = {"price": 50.0}
    assert model.compute_fill_price(context) == 50.0


def test_result_type_is_float_even_if_components_return_ints():
    model = SlippageModel(spread_fct=const_func_factory(1), spread_coeff=1)
    context = {"price": 10}
    result = model.compute_fill_price(context)
    assert isinstance(result, float)


# ---------------------------------------------------
# Context handling tests
# ---------------------------------------------------

def test_context_as_dict_is_supported():
    model = SlippageModel(spread_fct=identity_func, spread_coeff=1)
    context = {"price": 10.0, "marker": 7}
    result = model.compute_fill_price(context)
    # Expected: 10 + 1 * 7
    assert result == 17.0


def test_context_as_slippagecontext_object_raises():
    """Current implementation uses context['price'], so SlippageContext will fail."""
    model = SlippageModel(spread_fct=identity_func, spread_coeff=1)
    ctx = SlippageContext(price=10.0, order_size=5, timestamp="t1")
    with pytest.raises(TypeError):
        # Because __getitem__ is not defined in SlippageContext
        model.compute_fill_price(ctx)


def test_context_not_mutated_by_compute():
    """compute_fill_price should not alter the context object."""
    model = SlippageModel(spread_fct=identity_func, spread_coeff=1)
    context = {"price": 5.0, "marker": 2}
    snapshot = dict(context)
    model.compute_fill_price(context)
    assert context == snapshot


def test_each_component_receives_same_context_instance():
    """All component functions should receive the exact same context object."""
    called_contexts = []

    def recorder(context, **kwargs):
        called_contexts.append(context)
        return 1

    model = SlippageModel(
        spread_fct=recorder, spread_coeff=1,
        market_impact_fct=recorder, MI_coeff=1,
    )
    context = {"price": 10.0}
    model.compute_fill_price(context)
    assert all(ctx is context for ctx in called_contexts)


# ---------------------------------------------------
# Call behavior and ordering
# ---------------------------------------------------

def test_none_components_are_skipped():
    model = SlippageModel()  # all None
    context = {"price": 1.0}
    assert model.compute_fill_price(context) == 1.0


def test_components_called_once_each_in_order():
    call_order = []

    def make_func(name, val):
        def f(context, **kwargs):
            call_order.append(name)
            return val
        return f

    model = SlippageModel(
        spread_fct=make_func("spread", 1), spread_coeff=1,
        market_impact_fct=make_func("mi", 2), MI_coeff=1,
        queue_fct=make_func("queue", 3), queue_coeff=1,
        auct_prenium_fct=make_func("ap", 4), AP_coeff=1,
    )
    context = {"price": 0.0}
    model.compute_fill_price(context)
    assert call_order == ["spread", "mi", "queue", "ap"]


def test_component_still_called_with_zero_coeff():
    called = {}

    def marker_func(context, **kwargs):
        called["was_called"] = True
        return 42

    model = SlippageModel(spread_fct=marker_func, spread_coeff=0.0)
    context = {"price": 10.0}
    model.compute_fill_price(context)
    assert called.get("was_called", False) is True


# ---------------------------------------------------
# Robustness & error surfacing
# ---------------------------------------------------

def test_missing_price_in_context_raises():
    model = SlippageModel(spread_fct=identity_func, spread_coeff=1)
    with pytest.raises(KeyError):
        model.compute_fill_price({})  # no "price" key


def test_component_function_raises_error_propagates():
    def bad_func(context, **kwargs):
        raise ValueError("test error")

    model = SlippageModel(spread_fct=bad_func, spread_coeff=1)
    with pytest.raises(ValueError, match="test error"):
        model.compute_fill_price({"price": 1.0})


def test_params_kwargs_mismatch_raises_at_calltime():
    """If params include unexpected kwargs, TypeError should be raised at compute time."""
    def func_with_no_kwargs(context):
        return 1

    params = {"spread": {"unexpected": 123}}
    model = SlippageModel(spread_fct=func_with_no_kwargs, spread_coeff=1, params=params)
    with pytest.raises(TypeError):
        model.compute_fill_price({"price": 1.0})


def test_large_coefficients_produce_finite_result():
    model = SlippageModel(spread_fct=const_func_factory(1e6), spread_coeff=1e6)
    context = {"price": 1.0}
    result = model.compute_fill_price(context)
    import math
    assert math.isfinite(result)


# ---------------------------------------------------
# Param passing tests
# ---------------------------------------------------

def test_per_component_params_bound_correctly():
    """Each component should receive its own params through partial binding."""
    received = {}

    def spread_func(context, window=None):
        received["spread_window"] = window
        return 1

    def mi_func(context, factor=None):
        received["mi_factor"] = factor
        return 1

    params = {
        "spread": {"window": 10},
        "mi": {"factor": 0.02},
    }

    model = SlippageModel(
        spread_fct=spread_func, spread_coeff=1,
        market_impact_fct=mi_func, MI_coeff=1,
        params=params
    )
    model.compute_fill_price({"price": 1.0})
    assert received["spread_window"] == 10
    assert received["mi_factor"] == 0.02


def test_absent_params_default_to_empty_dict():
    """If params for a component are not provided, partial is created without kwargs."""
    received = {}

    def func_no_kwargs(context):
        received["called"] = True
        return 1

    model = SlippageModel(spread_fct=func_no_kwargs, spread_coeff=1, params={})
    result = model.compute_fill_price({"price": 1.0})
    assert result == 2.0  # 1.0 base + 1*1 contribution
    assert received["called"] is True