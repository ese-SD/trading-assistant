
"""
class Position():
    def __init__(self, asset_name, id):
        self.asset_name=asset_name
        self.id=id

    def __eq__(self, other):
        if isinstance(other, Position):
            return self.id == other.id
        return False


    def open(self):
        raise NotImplementedError("Subclass must implement this method")
    

    def close(self):
         raise NotImplementedError("Subclass must implement this method")



"""


class LongPosition():
    def __init__(self, asset_name):
        self.asset_name=asset_name
        self.average_buy_price=0.0


    def buy(self, price, time, stock_amount):
        self.buy_time=time
        self.average_buy_price=(self.stock_amount*self.average_buy_price+stock_amount*price)/self.stock_amount+stock_amount
        self.stock_amount+=stock_amount


    def sell(self,amount,time,price):
        if self.stock_amount<amount:
            raise ValueError(f"Not enough stocks: trying to sell {amount} stocks, but only {self.stock_amount} stocks in the position")
        else: 
            self.stock_amount-=amount
            realized_pnl=(price-self.average_buy_price)*amount
            proceeds=amount*price
            return realized_pnl, proceeds




class ShortPosition(Position):

    def __init__(self, asset_name, id):
        super().__init__(asset_name, id)

    def open(self, price, time, stock_amount, )

    
#ajouter short, futur, option, leviers