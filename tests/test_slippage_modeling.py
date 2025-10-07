import pytest
from backtester.slippage_modeling import SlippageContext, SlippageModel
from functools import partial


# =====================================================
# Helpers
# =====================================================

def const_func_factory(value):
    """Return a function that always returns a fixed value, ignoring context."""
    def f(context, **kwargs):
        return value
    return f


def identity_func(context, **kwargs):
    """Return a marker from context.extra for testing call propagation."""
    return context.extra.get("marker", 1)


# =====================================================
# Tests for SlippageContext
# =====================================================

def test_context_initialization_basic():
    """SlippageContext should store its attributes correctly."""
    ctx = SlippageContext(price=101.5, order_size=100, timestamp="2025-09-25")
    assert ctx.price == 101.5
    assert ctx.order_size == 100
    assert ctx.timestamp == "2025-09-25"
    assert ctx.extra == {}


def test_context_with_extra_dict():
    """Extra dict should be stored intact and mutable."""
    extra = {"OCHLV": [{"High": 105, "Low": 100}]}
    ctx = SlippageContext(price=200.0, order_size=10, timestamp="2025-09-25", extra=extra)
    assert ctx.extra is extra
    ctx.extra["new_key"] = 42
    assert "new_key" in extra


def test_context_default_extra_is_independent():
    """Each SlippageContext should get its own empty extra dict."""
    c1 = SlippageContext(price=1.0, order_size=1, timestamp="t1")
    c2 = SlippageContext(price=2.0, order_size=2, timestamp="t2")
    c1.extra["x"] = 1
    assert "x" not in c2.extra
    assert c1.extra != c2.extra


def test_context_mutability_isolation():
    """Changing one context's extra must not affect another."""
    ctx1 = SlippageContext(price=10.0, order_size=5, timestamp="t1", extra={"a": 1})
    ctx2 = SlippageContext(price=20.0, order_size=10, timestamp="t2", extra={"b": 2})
    ctx1.extra["c"] = 3
    assert "c" in ctx1.extra
    assert "c" not in ctx2.extra


# =====================================================
# Tests for SlippageModel - Constructor
# =====================================================

def test_constructs_with_no_components():
    """If no functions are provided, all components should be (None, coeff)."""
    model = SlippageModel()
    assert all(getattr(model, comp)[0] is None for comp in
               ["spread", "market_impact", "queue", "auction_premium"])


def test_constructs_with_one_component_spread():
    """Only spread provided: spread is callable, others are None."""
    model = SlippageModel(spread_fct=identity_func, spread_coeff=2.0)
    assert callable(model.spread[0])
    assert model.spread[1] == 2.0
    assert model.market_impact[0] is None


def test_constructs_with_all_components_and_params():
    """All components with params should be wrapped as partials."""
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
        assert isinstance(coeff, float)


def test_params_none_defaults_to_empty():
    """params=None should not crash and should default to {}."""
    model = SlippageModel(spread_fct=identity_func, spread_coeff=1.0, params=None)
    assert callable(model.spread[0])


def test_missing_keys_in_params_do_not_break():
    """Missing component keys in params should default to {}."""
    params = {"spread": {"window": 5}}
    model = SlippageModel(spread_fct=identity_func, spread_coeff=1.0, params=params)
    assert callable(model.spread[0])
    assert model.market_impact[0] is None


def test_non_dict_entry_in_params_raises_typeerror():
    """Passing non-dict params (e.g., None) should raise TypeError."""
    params = {"spread": None}
    with pytest.raises(TypeError):
        SlippageModel(spread_fct=identity_func, spread_coeff=1.0, params=params)


def test_coefficients_accept_various_values():
    """Coefficients can be ints, floats, negatives without issue."""
    model = SlippageModel(spread_fct=identity_func, spread_coeff=-5)
    assert model.spread[1] == -5


# =====================================================
# Tests for compute_fill_price
# =====================================================

def test_no_components_returns_base_price():
    """Without any components, total = context.price."""
    model = SlippageModel()
    ctx = SlippageContext(price=100.0, order_size=10, timestamp="t1")
    assert model.compute_fill_price(ctx) == 100.0


def test_single_component_adds_to_price():
    """If one component returns 2 and coeff=3, result=price+6."""
    model = SlippageModel(spread_fct=const_func_factory(2), spread_coeff=3.0)
    ctx = SlippageContext(price=100.0, order_size=10, timestamp="t1")
    assert model.compute_fill_price(ctx) == pytest.approx(106.0)


def test_all_components_add_contributions():
    """Sum of all component effects should be additive."""
    model = SlippageModel(
        spread_fct=const_func_factory(1), spread_coeff=1,
        market_impact_fct=const_func_factory(2), MI_coeff=2,
        queue_fct=const_func_factory(3), queue_coeff=3,
        auct_prenium_fct=const_func_factory(4), AP_coeff=4,
    )
    ctx = SlippageContext(price=100.0, order_size=10, timestamp="t1")
    assert model.compute_fill_price(ctx) == 100 + 1 + 4 + 9 + 16


def test_component_returning_zero_no_change():
    """Component returning 0 should not alter result."""
    model = SlippageModel(spread_fct=const_func_factory(0), spread_coeff=10)
    ctx = SlippageContext(price=50.0, order_size=5, timestamp="t1")
    assert model.compute_fill_price(ctx) == 50.0


def test_result_type_is_float():
    """Final result should always be float."""
    model = SlippageModel(spread_fct=const_func_factory(1), spread_coeff=1)
    ctx = SlippageContext(price=10, order_size=5, timestamp="t1")
    assert isinstance(model.compute_fill_price(ctx), float)


# =====================================================
# Context handling & side effects
# =====================================================

def test_context_not_mutated():
    """compute_fill_price must not mutate the given SlippageContext."""
    model = SlippageModel(spread_fct=identity_func, spread_coeff=1)
    ctx = SlippageContext(price=5.0, order_size=5, timestamp="t1", extra={"marker": 2})
    before = ctx.__dict__.copy()
    model.compute_fill_price(ctx)
    assert ctx.__dict__ == before


def test_each_component_receives_same_context_instance():
    """All functions must receive the same context object."""
    calls = []
    def rec(context, **kwargs):
        calls.append(context)
        return 1
    model = SlippageModel(
        spread_fct=rec, spread_coeff=1,
        market_impact_fct=rec, MI_coeff=1
    )
    ctx = SlippageContext(price=10.0, order_size=5, timestamp="t1")
    model.compute_fill_price(ctx)
    assert all(c is ctx for c in calls)


# =====================================================
# Call behavior & ordering
# =====================================================

def test_none_components_are_skipped():
    """If function is None, should not be called."""
    model = SlippageModel()  # all None
    ctx = SlippageContext(price=1.0, order_size=1, timestamp="t1")
    assert model.compute_fill_price(ctx) == 1.0


def test_components_called_once_in_order():
    """Components should be called exactly once in correct order."""
    order = []
    def make(name, val):
        def f(context, **kw):
            order.append(name)
            return val
        return f
    model = SlippageModel(
        spread_fct=make("spread", 1), spread_coeff=1,
        market_impact_fct=make("mi", 2), MI_coeff=1,
        queue_fct=make("queue", 3), queue_coeff=1,
        auct_prenium_fct=make("ap", 4), AP_coeff=1,
    )
    ctx = SlippageContext(price=0.0, order_size=1, timestamp="t1")
    model.compute_fill_price(ctx)
    assert order == ["spread", "mi", "queue", "ap"]


def test_zero_coeff_still_calls_function():
    """Even with coeff=0.0, function should be called once."""
    called = {}
    def f(context, **kw):
        called["ok"] = True
        return 42
    model = SlippageModel(spread_fct=f, spread_coeff=0.0)
    ctx = SlippageContext(price=10.0, order_size=1, timestamp="t1")
    model.compute_fill_price(ctx)
    assert called.get("ok")


# =====================================================
# Robustness & error propagation
# =====================================================

def test_missing_price_attribute_raises():
    """If context lacks price attr, compute_fill_price should fail clearly."""
    model = SlippageModel(spread_fct=identity_func, spread_coeff=1)
    class Fake: pass
    with pytest.raises(AttributeError):
        model.compute_fill_price(Fake())


def test_component_function_raises_bubbles_up():
    """Errors from components must propagate."""
    def bad(context, **kw): raise ValueError("Boom")
    model = SlippageModel(spread_fct=bad, spread_coeff=1)
    ctx = SlippageContext(price=1.0, order_size=1, timestamp="t1")
    with pytest.raises(ValueError, match="Boom"):
        model.compute_fill_price(ctx)


def test_params_with_unexpected_kwargs_raise_at_runtime():
    """Unexpected param key should raise TypeError when partial is called."""
    def no_kwargs(context): return 1
    params = {"spread": {"oops": 99}}
    model = SlippageModel(spread_fct=no_kwargs, spread_coeff=1, params=params)
    ctx = SlippageContext(price=1.0, order_size=1, timestamp="t1")
    with pytest.raises(TypeError):
        model.compute_fill_price(ctx)


def test_large_coefficients_result_is_finite():
    """Large coeffs should not overflow to inf/NaN."""
    import math
    model = SlippageModel(spread_fct=const_func_factory(1e6), spread_coeff=1e6)
    ctx = SlippageContext(price=1.0, order_size=1, timestamp="t1")
    result = model.compute_fill_price(ctx)
    assert math.isfinite(result)


# =====================================================
# Param binding checks
# =====================================================

def test_params_bound_correctly_to_each_component():
    """Each component should receive its respective bound kwargs."""
    seen = {}
    def spread(context, window=None):
        seen["spread_window"] = window
        return 1
    def mi(context, factor=None):
        seen["mi_factor"] = factor
        return 1
    params = {"spread": {"window": 5}, "mi": {"factor": 0.3}}
    model = SlippageModel(spread_fct=spread, spread_coeff=1,
                          market_impact_fct=mi, MI_coeff=1,
                          params=params)
    ctx = SlippageContext(price=1.0, order_size=1, timestamp="t1")
    model.compute_fill_price(ctx)
    assert seen["spread_window"] == 5
    assert seen["mi_factor"] == 0.3


def test_absent_params_default_to_empty_dict():
    """If params missing, partial should still call without kwargs."""
    called = {}
    def f(context): called["yes"] = True; return 1
    model = SlippageModel(spread_fct=f, spread_coeff=1, params={})
    ctx = SlippageContext(price=1.0, order_size=1, timestamp="t1")
    result = model.compute_fill_price(ctx)
    assert result == 2.0
    assert called["yes"]
