class Signal():
    """
    This class contains the informations necessary to place and then fulfill an order.

    It allows a Strategy to place various typs of orders, which will get picked up by Brain to be verified and then fulfilled.
    
    Attributes:
        id (int): id of the signal/order.
        ttype (str): type of signal.
        ticker (str): the ticker on which the order is placed.
        share_nb (float): the number of shares intended to be bought.
        price (float): the price at which the shares should be bought.
        money (float): amount of money sent in the order.
        
    """
    def __init__(self, id, ttype, ticker, share_nb, price, money):
        """
        Args:
        id (int): used by Brain for the historic, and to delete limit orders, among other things.
        ttype (str): the type of the order, on which depends how it will be handled by Brain.
        ticker (str): the ticker on which the order is placed.
        share_nb (float): the number of share to buy (or *want* to buy, in case of market orders, because of slippage)
        price (float): share price at the time of the signal. For limits, it sets the exact price at which to fulfill the order.
        money (float): the actual amount in money of the order. This also determines how much shares are bought for BUY_MARKET orders(because slippage).
        
        """
        types=["BUY_MARKET", "BUY_LIMIT", "SELL_MARKET", "SELL_LIMIT"]
        if (type(id)==int and ttype in types and type(ticker)==str and type(share_nb)==float and type(price)==float and type(money)==float):
            self.id=id
            self.type=type
            self.ticker=ticker
            self.share_nb=share_nb
            self.price=price
            self.money=money
        else:
            raise ValueError("Invalid arguments.")