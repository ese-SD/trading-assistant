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


#...