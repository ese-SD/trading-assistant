import pandas as pd
from pandas import DatetimeIndex
import numpy as np
import yfinance as yf
import warnings
from datetime import datetime, timedelta, timezone
from dateutil import parser
import os
from pathlib import Path


def download(API, ticker, start, end, interval, name: Path | str , folder: Path | str = None, key='', auto_adjust=True):
    """
    Downloads the historical ticker data in a clearly defined format.

    This serves as a wrapper function, as each API is different and needs a specific function to handle it.
    Each API-specific function returns the data in the format used by yfinance.
    
    Args: 
        API (string): name of the API to get the data from, to be selected from a predefined list.
        ticker (string): name of the ticker (stock indicator) from which to download the data.
        start (datetime): starting period of the historical data to download. (inclusive)
        end (datetime): ending period of the historical data to download. (inclusive)
        interval(???): how far appart each "point" should be in time. 
        name (Path | str): name given to the file the data is saved in. If a Path is given, only the last part of 'name' will be kept, to not interfere with 'folder'.
        folder (Path | str): folder in which to save the file. If not specified, it will be saved in the 'data' folder, within the main project folder
        key (str): API key if necessary.
        auto_adjust (Bool): wether prices have to be adjusted for dividends or share splitting within the period.
            e.g.: a 100$ share will go to 98$ the day after a 2$ per share dividend has been distributed, but the value of the share has only decreased artificially.
            => auto_adjust=True will show the share as 98$ before and after the dividend.

    Raises:
        TypeError: If start or end has invalid type.
        ValueError: If arguments are invalid or if API data is missing.
        UserWarning: If API data is incomplete or possibly truncated.
    
    """
    if not isinstance(start, datetime) or not isinstance(end, datetime) :
        raise TypeError("start and stop have to be datetimes")
    
    if start.tzinfo is None or end.tzinfo is None:
        warnings.warn("Timezones not specified, default to UTC.")
        start=start.replace(tzinfo=timezone.utc); end=end.replace(tzinfo=timezone.utc)

    valid_apis=["yfinance"]
    if API not in valid_apis:
        raise ValueError(f" API {API} not recognized. chose among {valid_apis}.")
    match API:
        case "yfinance":
            data=load_yfinance(ticker, start, end, interval, auto_adjust)


    name=Path(name).with_suffix(".parquet")
    """We want name to be just the file name, not a complete path. If it is, its truncated to keep only the last part."""
    if name.parent!=Path("."):
        warnings.warn(f"'name' {name} has been shortened to {name.name}, as it did not expect a full path. ")
        name=name.name
    if folder is None :
        folder = Path(__file__).resolve().parent.parent / "data"
    else:
        """Converts to Path if folder was a str, does nothing otherwise"""
        folder= Path(folder) 
    if not folder.exists(): 
        print(f"Folder {folder} did not exist, and has been created.")
        folder.mkdir(parents=True)

    full_path=folder/name
    data.to_parquet(full_path, engine="pyarrow")
    print(f"Data saved to {full_path}.")




#devnote: ajouter auto adjust? 
def load_yfinance(ticker, start, end, interval, auto_adjust):
    """
    Loads historical data from the yfinance API.

    Args:
        ticker (str): name of the ticker (stock indicator) from which to download the data.
        start (str): starting period of the historical data to download.
        end (str): ending period of the historical data to download.
        interval(str) how far appart each "point" should be in time. 
        auto_adjust(Bool): wether to adjust share prices.

    Returns:
        A Dataframe containing the OHLCV (OHLCV+adj?) values by dates.
    
    Raises:     
        ValueError: If arguments are invalid or if data is missing.
        UserWarning: If data is incomplete or possibly truncated.
    """

    """Check args validity."""


    valid_intervals = [
        "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1d", "5d", "1wk", "1mo", "3mo" 
    ]
    if interval not in valid_intervals:
        raise ValueError(f"Invalid interval. Chose from {valid_intervals}")
    
    try:
        info = yf.Ticker(ticker).info
        if not all(k in info for k in ["symbol", "quoteType", "regularMarketPrice"]):
            warnings.warn("Ticker metadata is incomplete.")
    except Exception:
            warnings.warn("Failed to fetch ticker metadata (API issue or rate limit)")

    data= yf.download(ticker, start=start, interval=interval, end=end, auto_adjust=auto_adjust)

    """Check answer validity."""

    if data.index.empty:
        raise ValueError("Data index is empty, possible backend API issue.")
    
    if  data.empty:
        raise ValueError("Data is empty. Check if ticker is valid, or if data should be present for the" \
        "specified period.")

    """Check if yfinance didn't mistakenly return daily data instead of intraday."""
    if interval in {"1m", "2m", "5m", "15m", "30m", "60m"}:
        """Checks the most common difference from one row to the other. dropna drops the first value, which doesn't
        have a previous row."""
        warnings.warn(
            "Intraday data doesn't work reliably on the yfinance API. If intraday data is important "
            "and you see issues with yfinance, try another API", category=UserWarning
            )
    
        diffs = data.index.to_series().diff().dropna()
        if not diffs.empty and diffs.mode()[0] >= timedelta(hours=1):
            warnings.warn(
                f"Expected intraday data ({interval}) but received daily or sparse data. "
                f"Check if API failed silently."
            )
    """
    actual_start = data.index.min()
    actual_end = data.index.max()
    if actual_start>start or actual_end<end:  
        warnings.warn("Specified period ({} to {}) doesn't match downloaded data's period ({} to {})".format (start, end, actual_start, actual_end), category=UserWarning)
    """

    if "Close" not in data.columns:
        raise ValueError("Close price columns missing.")
    
    return data


if __name__ == "__main__":
    from datetime import datetime, timedelta

    start=parser.parse("2020-08-27 00:00:00")
    end=parser.parse("2020-09-03 00:00:00")

    df=download("yfinance", "AAPL", start, end, interval="1d", name="test_name")
    print(pd.read_parquet("data/test_name.parquet"))
     

