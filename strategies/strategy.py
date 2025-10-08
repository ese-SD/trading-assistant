from abc import ABC, abstractmethod
import importlib
from backtester.signal import Signal
from typing import List

def load_strategy(strategy_name, **kwargs):
    module = importlib.import_module(f"strategies.{strategy_name}")
    return module.Strategy(**kwargs)


#abstract class that defines what a strategy is
class Strategy(ABC):

    @property
    @abstractmethod
    def frequency(self) -> int:
        #frequency at which the strategy samples from the market, in seconds (for now)
        pass


    @property
    @abstractmethod
    def signals(self) -> List[Signal]:
        #list of signals to be picked up and executed
        pass

    @abstractmethod
    def buy(self):
        #decide if it should buy, then generates the buy signal
        pass

    @abstractmethod
    def amount_to_buy(self):
        #decides how much to buy, once the buy signal has been generated
        pass

    @abstractmethod
    def define_exit(self):
        #sets the exit condition
        pass
    
    @abstractmethod
    def update_exit(self):
        #(optional) allows for trailing stop loss, etc
        pass

    
    @abstractmethod
    def evaluate_existing_positions(self):
        #decides what to do with the currently open positions
        pass

    @abstractmethod
    def sell(self):
        #liquidates the position
        pass


    @abstractmethod
    def risk_management(self):
        """more general part of the strategy that governs all the present and futur positions based on the wallet's
        condition"""
        pass

    