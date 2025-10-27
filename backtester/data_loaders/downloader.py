import pandas as pd
from pandas import DatetimeIndex
import numpy as np
import yfinance as yf
import warnings
from datetime import datetime, timedelta, timezone
from dateutil import parser
import os


"""devnote: 
-standardiser la timezone
-save en .parquet

"""

def download(API, ticker, start, end, interval, name, key=''):
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
        name (string): name given to the file containing the data.
        key (str): API key if necessary.

    Raises:
        les differntes erreurs si un arg est pas reconnu (api, ticker, interval)
    
    """
    if type(start)!=datetime or type(end)!=datetime:
        raise TypeError("start and stop have to be datetimes")
    

    valid_apis=["yfinance"]
    if API not in valid_apis:
        raise ValueError(f" API {API} not recognized. chose among {valid_apis}.")
    match API:
        case "yfinance":
            data=load_yfinance(ticker, start, end, interval)


    file_path = "data/" + name + ".parquet"
    data.to_parquet(file_path, engine="pyarrow")
    print("File saved to /{}.".format( os.getcwd()+file_path))




#devnote: ajouter auto adjust? 
def load_yfinance(ticker, start, end, interval):
    """
    Loads historical data from the yfinance API.

    Args:
        ticker (str): name of the ticker (stock indicator) from which to download the data.
        start (str): starting period of the historical data to download.
        end (str): ending period of the historical data to download.
        interval(str) how far appart each "point" should be in time. 

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
        raise ValueError("Invalid interval. Chose from {}".format(valid_intervals))
    
    try:
        info = yf.Ticker(ticker).info
        if not all(k in info for k in ["symbol", "quoteType", "regularMarketPrice"]):
            warnings.warn("Ticker metadata is incomplete.")
    except Exception:
            warnings.warn("Failed to fetch ticker metadata (API issue or rate limit)")

    """yfinance excludes the last day of the specified period. Add 1 day to compensate."""
    #data= yf.download(ticker, start, end+ timedelta(days=1), interval)
    data= yf.download(ticker, start=start, interval=interval, end=end)

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
        
    
    """actual_start = data.index.min()
    actual_end = data.index.max()
    if actual_start>start or actual_end<end:  
        warnings.warn("Specified period ({} to {}) doesn't match downloaded data's period ({} to {})".format (start, end, actual_start, actual_end), category=UserWarning)
    """
    if "Close" not in data.columns:
        raise ValueError("Closing price columns missing.")
    
    return data




