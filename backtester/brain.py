import data_loaders
import strategies
from data_loaders.datafeed import Datafeed
from utils.market_forces import *
from datetime import datetime
from backtester.slippage_modeling import *
from backtester.signal import Signal



class Brain():
    
    """
    
    
    
    Args:

    slippage_context (SlippageContext): object that contains all the data that could be used to compute slippage, a sort of "snapshot" of Brain at the buy time
    execution_delay (datetime): how long orders are expected to take to be executed.
    
    
    
    """


    def __init__(self, slippage_context : SlippageContext, execution_delay : datetime):
        self.slippage_context=slippage_context 
        self.execution_delay=execution_delay 
        self.trade_history=list[Signal]

    def hook_data_feed(self, filepath):
        self.data_feed=Datafeed()
        self.data_feed.load_data(self,filepath)
    
    def hook_strategy(self, strategy_name):
        self.strategy=strategies.strategy_name#load_strategy?

    def init_meta_args(self, spread_fct, spread_coeff, model_impact_fct, model_impact_coeff, fee_structure):
        self.spread_fct=spread_fct
        self.spread_coeff=spread_coeff
        self.model_impact_fct=model_impact_fct
        self.model_impact_coeff=model_impact_coeff
        self.fee_structure=fee_structure
        #specifier les args supp de fee_structure de sorte que les 2 seuls non specifiés sont order_size/nb_shares et share_price


    def hook_wallet(self, wallet):
        self.wallet=wallet

    def hook_slippage_model(self, model : SlippageModel):
        self.slippage_model=model

    def is_signal_valid(self, signal):
        if signal.ticker != self.data_feed.ticker:
            return False



    def execute_signals(self, signals, current_price, current_time):
        #verifier le signal (specifique a chaque type de signal)
        #executer le signal (specifique a chaque type de signal)
        if current_time==self.data_feed.data.index[-1]:
            print("Can't execute signals on the last bar, because execution delay requires the signal to be executed on the next bar.")
            return 0


        for signal in signals:
            if signal.ticker!=self.data_feed.ticker:
                print("signal {} received on {} is invalid because ticker {} doesn't match the datafeed's ticker.".format(signal, current_time, signal.ticker))
                # This signal wont be executed, and will later be deleted.
                continue
            match signal.ttype:
                case "BUY_MARKET":
                    #1verify order: assez de cash, ticker. si sell, verifier si les shares sont en possession
                    #2apply fee
                    #3apply slippage(si market, pas limit)
                    #4deduce cash
                    #5increase share nb in wallet (pour market. Pour limit, ajouter a une list de limit orders)
                    #6add signal to history
                    #7delete signal(si market)
    
                    fee=self.fee_structure(current_price, signal.nb_shares)
                    if self.fee_structure.application=="deducted":
                        total
                    else: 
                        pass #self.wallet.cash-=fee?????????

                    # Verify order: 
                    

                    # Apply delay:
                    true_execution_time= self.data_feed.data.index[self.data_feed.data.index.get_loc(current_time) + 1]
                    self.slippage_context.price=self.data_feed.data[true_execution_time]["Open"]

                    # Apply slippage
                    new_cost_per_share=self.slippage_model(self.slippage_context)
                    
                    # Compute how many shares are bought
                    nb_shares=signal.money/new_cost_per_share

                    # Update wallet
                    self.wallet.cash-=signal.money-fee
                    self.wallet.stocks[signal.ticker]+=nb_shares
                    
                    # Update history
                    self.trade_history.append(signal)
                    #pr deducted_fee: retirer le fee a l'argent de l'order, plutot que du wallet?


                case "SELL_MARKET":
                    pass

                case "BUY_LIMIT":
                    pass

                case "SELL_LIMIT":
                    pass


        # All remaining signals are invalid and have to be deleted
        # ou renvoyer feedback des signaux invalide?


    def update_context(self):
        pass


    def run(self,):
        if self.data_feed is None or self.strategy is None or self.wallet is None:
            raise ValueError("Undefined strategy or datafeed.")
        #si la frequence de la strategy est incompatible avec le data feed, erreur
        #if self.strategy.frequency!=data_feed
        """
        iterer sur le datafeed, et a chaque iteration, appeler next de strategy"""
        for i in Datafeed.data:
            pass
    
    



if __name__ == "__main__":
    pass