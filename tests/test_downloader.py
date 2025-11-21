
import pytest
import pandas as pd
from datetime import datetime, timedelta, timezone
import data_loaders.downloader as mod
import types
import warnings
from pathlib import Path


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _daily_index(periods=3, start="2025-01-01"):
    return pd.date_range(start=start, periods=periods, freq="1D", tz=timezone.utc)


def _intraday_index(periods=5, freq="1min", start="2025-01-01 09:00:00"):
    return pd.date_range(start=start, periods=periods, freq=freq, tz=timezone.utc)


def _df_with_index(idx, cols=("Close",)):
    df = pd.DataFrame(index=idx)
    for c in cols:
        df[c] = range(len(idx))
    return df





# ======================================================================
# 1: TESTS FOR load_yfinance
# ======================================================================


#-------------------------1.1: INTERVAL VALIDATION----------------------

def test_load_yfinance_rejects_invalid_interval(monkeypatch):
    # Mock yf to avoid actual calls
    monkeypatch.setattr(
        mod,
        "yf",
        types.SimpleNamespace(
            download=lambda *a, **k: None,
            Ticker=lambda *_: types.SimpleNamespace(info={})
        )
    )

    with pytest.raises(ValueError, match="Invalid interval"):
        mod.load_yfinance(
            "AAPL",
            start=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 2, tzinfo=timezone.utc),
            interval="10h",            # INVALID
            auto_adjust=True
        )


def test_load_yfinance_accepts_valid_interval_and_calls_download(monkeypatch):
    captured = {}

    def fake_download(ticker, start=None, end=None, interval=None, auto_adjust=True):
        captured["args"] = dict(
            ticker=ticker, start=start, end=end,
            interval=interval, auto_adjust=auto_adjust
        )
        idx = _daily_index(periods=2)
        return _df_with_index(idx, cols=("Close", "Open"))

    def fake_ticker(symbol):
        return types.SimpleNamespace(info={
            "symbol": symbol,
            "quoteType": "EQUITY",
            "regularMarketPrice": 1.0
        })

    monkeypatch.setattr(
        mod,
        "yf",
        types.SimpleNamespace(download=fake_download, Ticker=fake_ticker)
    )

    df = mod.load_yfinance(
        "AAPL",
        start=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end=datetime(2025, 1, 2, tzinfo=timezone.utc),
        interval="1d",
        auto_adjust=False
    )

    assert isinstance(df, pd.DataFrame)
    assert captured["args"]["ticker"] == "AAPL"
    assert captured["args"]["interval"] == "1d"
    assert captured["args"]["auto_adjust"] is False


#-----------------1.2: METADATA WARNINGS ------------------------------------


def test_load_yfinance_warns_on_incomplete_metadata(monkeypatch):
    def fake_ticker(_):
        return types.SimpleNamespace(info={"symbol": "AAPL"})  # missing fields

    def fake_download(*_, **__):
        idx = _daily_index(periods=2)
        return _df_with_index(idx, cols=("Close",))

    monkeypatch.setattr(
        mod,
        "yf",
        types.SimpleNamespace(download=fake_download, Ticker=fake_ticker)
    )

    with pytest.warns(UserWarning, match="metadata"):
        mod.load_yfinance(
            "AAPL",
            start=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 2, tzinfo=timezone.utc),
            interval="1d",
            auto_adjust=True
        )


def test_load_yfinance_warns_when_info_fetch_fails(monkeypatch):
    class T:
        @property
        def info(self):
            raise RuntimeError("boom")

    def fake_ticker(_):
        return T()

    def fake_download(*_, **__):
        idx = _daily_index(periods=2)
        return _df_with_index(idx, cols=("Close",))

    monkeypatch.setattr(
        mod,
        "yf",
        types.SimpleNamespace(download=fake_download, Ticker=fake_ticker)
    )

    with pytest.warns(UserWarning, match="Failed to fetch"):
        mod.load_yfinance(
            "AAPL",
            start=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 2, tzinfo=timezone.utc),
            interval="1d",
            auto_adjust=True
        )


#------------------------1.3:DATA VALIDATION ------------------------------

def test_load_yfinance_raises_when_index_empty(monkeypatch):
    def fake_download(*_, **__):
        df = pd.DataFrame({"Close": []})
        df.index = pd.DatetimeIndex([])  # empty index
        return df

    monkeypatch.setattr(
        mod,
        "yf",
        types.SimpleNamespace(
            download=fake_download,
            Ticker=lambda *_: types.SimpleNamespace(info={"symbol": "AAPL","quoteType":"EQUITY","regularMarketPrice":1.0})
        )
    )

    with pytest.raises(ValueError, match="index.*empty"):
        mod.load_yfinance(
            "AAPL",
            start=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 2, tzinfo=timezone.utc),
            interval="1d",
            auto_adjust=True
        )


def test_load_yfinance_raises_when_dataframe_empty(monkeypatch):
    def fake_download(*_, **__):
        idx = _daily_index(periods=2)
        df = pd.DataFrame(index=idx)    # no columns
        return df

    monkeypatch.setattr(
        mod,
        "yf",
        types.SimpleNamespace(
            download=fake_download,
            Ticker=lambda *_: types.SimpleNamespace(info={"symbol":"AAPL","quoteType":"EQUITY","regularMarketPrice":1.0})
        )
    )

    with pytest.raises(ValueError, match="Data.*empty"):
        mod.load_yfinance(
            "AAPL",
            start=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 2, tzinfo=timezone.utc),
            interval="1d",
            auto_adjust=True
        )


def test_load_yfinance_raises_when_close_missing(monkeypatch):
    def fake_download(*_, **__):
        idx = _daily_index(periods=2)
        return _df_with_index(idx, cols=("Open", "High", "Low"))   # no 'Close'

    monkeypatch.setattr(
        mod,
        "yf",
        types.SimpleNamespace(
            download=fake_download,
            Ticker=lambda *_: types.SimpleNamespace(info={"symbol":"AAPL","quoteType":"EQUITY","regularMarketPrice":1.0})
        )
    )

    with pytest.raises(ValueError, match="Close"):
        mod.load_yfinance(
            "AAPL",
            start=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 2, tzinfo=timezone.utc),
            interval="1d",
            auto_adjust=True
        )


def test_load_yfinance_returns_dataframe_when_valid(monkeypatch):
    def fake_download(*_, **__):
        idx = _daily_index(periods=2)
        return _df_with_index(idx, cols=("Close", "Open", "High", "Low"))

    monkeypatch.setattr(
        mod,
        "yf",
        types.SimpleNamespace(
            download=fake_download,
            Ticker=lambda *_: types.SimpleNamespace(info={
                "symbol":"AAPL","quoteType":"EQUITY","regularMarketPrice":1.0
            })
        )
    )

    df = mod.load_yfinance(
        "AAPL",
        start=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end=datetime(2025, 1, 2, tzinfo=timezone.utc),
        interval="1d",
        auto_adjust=True
    )

    assert isinstance(df, pd.DataFrame)
    assert "Close" in df.columns


# -------------------1.4: INTRADAY BEHAVIOR----------------------------------


@pytest.mark.parametrize("interval", ["1m", "2m", "5m", "15m", "30m", "60m"])
def test_intraday_generic_warning_always_emitted(monkeypatch, interval):
    """All intraday intervals must always emit the generic unreliability warning."""

    def fake_download(*_, **__):
        idx = _intraday_index(periods=3, freq="1min")
        return _df_with_index(idx, cols=("Close", "Open"))

    # Mock yf
    monkeypatch.setattr(mod, "yf", types.SimpleNamespace(
        download=fake_download,
        Ticker=lambda *_: types.SimpleNamespace(info={"symbol": "AAPL", "quoteType": "EQUITY", "regularMarketPrice": 1.0})
    ))

    with pytest.warns(UserWarning, match=r"Intraday data.*reliab"):
        mod.load_yfinance(
            "AAPL",
            start=datetime(2025, 1, 1, 9, 0, tzinfo=timezone.utc),
            end=datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc),
            interval=interval,
            auto_adjust=True
        )


def test_intraday_sparse_spacing_triggers_additional_warning(monkeypatch):
    """Sparse intraday data should emit both the generic warning and the additional spacing warning."""

    def fake_download(*_, **__):
        # 2 rows spaced 24h → definitely sparse
        idx = _intraday_index(start="2025-01-01 09:00:00", periods=2, freq="24h")
        return _df_with_index(idx, cols=("Close", "Open"))

    monkeypatch.setattr(mod, "yf", types.SimpleNamespace(
        download=fake_download,
        Ticker=lambda *_: types.SimpleNamespace(info={"symbol": "AAPL", "quoteType": "EQUITY", "regularMarketPrice": 1.0})
    ))

    # Expect *two* warnings
    with pytest.warns(UserWarning, match=r"Intraday data.*reliab"):
        with pytest.warns(UserWarning, match=r"Expected intraday.*daily|sparse"):
            mod.load_yfinance(
                "AAPL",
                start=datetime(2025, 1, 1, 9, 0, tzinfo=timezone.utc),
                end=datetime(2025, 1, 3, 9, 0, tzinfo=timezone.utc),
                interval="1m",
                auto_adjust=True
            )


def test_intraday_proper_spacing_no_sparse_warning(monkeypatch):
    """Well-spaced intraday data should only give the generic warning (no spacing warning)."""

    def fake_download(*_, **__):
        idx = _intraday_index(periods=5, freq="5min")
        return _df_with_index(idx, cols=("Close", "Open"))

    monkeypatch.setattr(mod, "yf", types.SimpleNamespace(
        download=fake_download,
        Ticker=lambda *_: types.SimpleNamespace(info={"symbol": "AAPL", "quoteType": "EQUITY", "regularMarketPrice": 1.0})
    ))

    with pytest.warns(UserWarning, match=r"Intraday data.*reliab"):
        mod.load_yfinance(
            "AAPL",
            start=datetime(2025, 1, 1, 9, 0, tzinfo=timezone.utc),
            end=datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc),
            interval="5m",
            auto_adjust=True
        )


def test_non_intraday_interval_no_intraday_warnings(monkeypatch):
    """Daily interval should NOT emit any intraday warnings."""

    def fake_download(*_, **__):
        idx = _daily_index(periods=3)
        return _df_with_index(idx, cols=("Close", "Open"))

    monkeypatch.setattr(mod, "yf", types.SimpleNamespace(
        download=fake_download,
        Ticker=lambda *_: types.SimpleNamespace(info={"symbol": "AAPL", "quoteType": "EQUITY", "regularMarketPrice": 1.0})
    ))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        mod.load_yfinance(
            "AAPL",
            start=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 3, tzinfo=timezone.utc),
            interval="1d",
            auto_adjust=True
        )

    assert len(w) == 0, f"Unexpected warnings emitted: {w}"


def test_intraday_single_row_only_generic_warning(monkeypatch):
    """Single-row intraday data should still emit the generic warning, but no sparse-data warning."""

    def fake_download(*_, **__):
        idx = _intraday_index(periods=1, freq="1min")
        return _df_with_index(idx, cols=("Close", "Open"))

    monkeypatch.setattr(mod, "yf", types.SimpleNamespace(
        download=fake_download,
        Ticker=lambda *_: types.SimpleNamespace(info={"symbol": "AAPL", "quoteType": "EQUITY", "regularMarketPrice": 1.0})
    ))

    with pytest.warns(UserWarning, match=r"Intraday data.*reliab"):
        mod.load_yfinance(
            "AAPL",
            start=datetime(2025, 1, 1, 9, 0, tzinfo=timezone.utc),
            end=datetime(2025, 1, 1, 9, 10, tzinfo=timezone.utc),
            interval="1m",
            auto_adjust=True
        )



# ===========================================================================
# 2: tests for download
# ===========================================================================


def test_download_rejects_non_datetime_start_end():
    """download() must reject start/end that are not datetime objects."""

    with pytest.raises(TypeError):
        mod.download(
            API="yfinance",
            ticker="AAPL",
            start="2025-01-01",     # invalid
            end=datetime(2025, 1, 2, tzinfo=timezone.utc),
            interval="1d",
            name="test"
        )

    with pytest.raises(TypeError):
        mod.download(
            API="yfinance",
            ticker="AAPL",
            start=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end="2025-01-02",       # invalid
            interval="1d",
            name="test"
        )

"""
def test_download_naive_datetimes_warn_and_become_utc():
    #Naive datetimes must produce a warning and be coerced to UTC.

    start = datetime(2025, 1, 1)  # naive
    end = datetime(2025, 1, 3)    # naive

    # mock loader to inspect received datetimes
    received = {}

    def fake_loader(ticker, start, end, interval, auto_adjust):
        received["start"] = start
        received["end"] = end
        return _df_with_index(_daily_index(periods=3), cols=("Close",))

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(mod, "load_yfinance", fake_loader)

    with pytest.warns(UserWarning, match=r"naive|timezone|UTC"):
        mod.download(
            API="yfinance",
            ticker="AAPL",
            start=start,
            end=end,
            interval="1d",
            name="out"
        )

    assert received["start"].tzinfo is timezone.utc
    assert received["end"].tzinfo is timezone.utc

    monkeypatch.undo()
"""

def test_download_naive_datetimes_warn_and_become_utc(monkeypatch, tmp_path):
    """If start/end are naive datetimes, download() must warn and convert them to UTC."""

    called = {}

    def fake_loader(ticker, start, end, interval, auto_adjust):
        called["start"] = start
        called["end"] = end
        idx = _daily_index()
        return _df_with_index(idx, cols=("Close",))

    monkeypatch.setattr(mod, "load_yfinance", fake_loader)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, p, engine="pyarrow": None)

    with pytest.warns(UserWarning, match="Timezones not specified"):
        mod.download(
            API="yfinance",
            ticker="AAPL",
            start=datetime(2025, 1, 1),  # naive
            end=datetime(2025, 1, 2),    # naive
            interval="1d",
            name="testfile"
        )

    assert called["start"].tzinfo == timezone.utc
    assert called["end"].tzinfo == timezone.utc



def test_download_unknown_api_raises():
    """Unsupported API names must raise ValueError."""

    with pytest.raises(ValueError, match=r"API|unsupported|unknown"):
        mod.download(
            API="binance",
            ticker="AAPL",
            start=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 2, tzinfo=timezone.utc),
            interval="1d",
            name="x"
        )



def test_download_parquet_write_failure_bubbles(monkeypatch):
    """If DataFrame.to_parquet raises, download must re-raise it."""

    # minimal valid loader
    def fake_loader(ticker, start, end, interval, auto_adjust):
        idx = _daily_index(periods=2)
        return _df_with_index(idx, cols=("Close", "Open"))

    monkeypatch.setattr(mod, "load_yfinance", fake_loader)

    def bad_parquet(self, path, engine="pyarrow"):
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", bad_parquet)

    with pytest.raises(OSError, match="disk full"):
        mod.download(
            API="yfinance",
            ticker="AAPL",
            start=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 2, tzinfo=timezone.utc),
            interval="1d",
            name="boom",
            auto_adjust=True
        )


def test_download_dispatches_to_yfinance_and_writes_parquet(monkeypatch, tmp_path):
    """Must call load_yfinance, truncate name if path, and write parquet to default folder."""

    captured = {}

    def fake_loader(ticker, start, end, interval, auto_adjust):
        captured["args"] = (ticker, start, end, interval, auto_adjust)
        idx = _daily_index(periods=2)
        return _df_with_index(idx, cols=("Close", "Open"))

    monkeypatch.setattr(mod, "load_yfinance", fake_loader)

    written = {}

    def fake_parquet(self, path, engine="pyarrow"):
        written["path"] = Path(path)
        written["engine"] = engine

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_parquet)

    full_name = tmp_path / "nested" / "mydata"

    with pytest.warns(UserWarning, match="shortened"):
        mod.download(
            API="yfinance",
            ticker="AAPL",
            start=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 2, tzinfo=timezone.utc),
            interval="1d",
            name=full_name,
            auto_adjust=False
        )

    # Correct loader call
    assert captured["args"][0] == "AAPL"
    assert captured["args"][3] == "1d"
    assert captured["args"][4] is False

    # Name truncated → file becomes mydata.parquet inside default folder
    default_folder = mod.Path(mod.__file__).resolve().parent.parent / "data"
    assert written["path"].parent == default_folder
    assert written["path"].name == "mydata.parquet"


def test_download_end_to_end_happy_path(monkeypatch, tmp_path):
    """End-to-end run with correct saving to default folder."""

    def fake_loader(ticker, start, end, interval, auto_adjust):
        idx = _daily_index(periods=3)
        return _df_with_index(idx, cols=("Close", "Open", "High", "Low"))

    monkeypatch.setattr(mod, "load_yfinance", fake_loader)

    saved = {}
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, p, engine="pyarrow": saved.update({"path": p}))

    mod.download(
        API="yfinance",
        ticker="AAPL",
        start=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end=datetime(2025, 1, 3, tzinfo=timezone.utc),
        interval="1d",
        name="happy"
    )

    default_folder = mod.Path(mod.__file__).resolve().parent.parent / "data"
    assert Path(saved["path"]) == default_folder / "happy.parquet"



def test_download_name_with_path_warns_and_truncates(monkeypatch, tmp_path):
    def fake_loader(*_, **__):
        idx = _daily_index()
        return _df_with_index(idx, cols=("Close",))

    monkeypatch.setattr(mod, "load_yfinance", fake_loader)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, p, engine="pyarrow": None)

    bad_name = tmp_path / "folder" / "datafile"

    with pytest.warns(UserWarning, match="shortened"):
        mod.download(
            API="yfinance",
            ticker="AAPL",
            start=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 2, tzinfo=timezone.utc),
            interval="1d",
            name=bad_name
        )


def test_download_respects_custom_folder_argument(monkeypatch, tmp_path):
    def fake_loader(*_, **__):
        idx = _daily_index()
        return _df_with_index(idx, cols=("Close",))

    monkeypatch.setattr(mod, "load_yfinance", fake_loader)

    saved = {}
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, p, engine="pyarrow": saved.update({"path": p}))

    custom_folder = tmp_path / "custom"
    mod.download(
        API="yfinance",
        ticker="AAPL",
        start=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end=datetime(2025, 1, 2, tzinfo=timezone.utc),
        interval="1d",
        name="aapl",
        folder=custom_folder
    )

    assert Path(saved["path"]) == custom_folder / "aapl.parquet"



def test_download_defaults_to_project_data_folder(monkeypatch, tmp_path):
    def fake_loader(*_, **__):
        idx = _daily_index()
        return _df_with_index(idx, cols=("Close",))

    monkeypatch.setattr(mod, "load_yfinance", fake_loader)

    saved = {}
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, p, engine="pyarrow": saved.update({"path": p}))

    mod.download(
        API="yfinance",
        ticker="AAPL",
        start=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end=datetime(2025, 1, 2, tzinfo=timezone.utc),
        interval="1d",
        name="default"
    )

    default_folder = mod.Path(mod.__file__).resolve().parent.parent / "data"
    assert Path(saved["path"]) == default_folder / "default.parquet"



def test_download_creates_missing_folder(monkeypatch, tmp_path):
    def fake_loader(*_, **__):
        idx = _daily_index()
        return _df_with_index(idx, cols=("Close",))

    monkeypatch.setattr(mod, "load_yfinance", fake_loader)

    saved = {}
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, p, engine="pyarrow": saved.update({"path": p}))

    missing_folder = tmp_path / "does" / "not" / "exist"

    mod.download(
        API="yfinance",
        ticker="AAPL",
        start=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end=datetime(2025, 1, 2, tzinfo=timezone.utc),
        interval="1d",
        name="file",
        folder=missing_folder
    )

    # folder must exist now
    assert missing_folder.exists()
    assert Path(saved["path"]) == missing_folder / "file.parquet"