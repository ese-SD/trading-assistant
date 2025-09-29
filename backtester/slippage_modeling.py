from dataclasses import dataclass, field
from typing import Any
from math import log,sqrt,exp,pi




@dataclass
class SlippageContext:
    """
    Snapshot of the data in Brain that can be used to compute slippage.
    All the data that isnt common to all slippage models is stored in extra.
    """
    price: float
    order_size: int
    timestamp: str
    extra: dict[str, Any] = field(default_factory=dict)




class SlippageModel:
    """
    Models the slippage when filling an order based on the order and current market conditions.

    The user can customize and create their own slippage model by chosing how slippage's various components are modeled.
    
    """


    def __init__(self, spread_fct, spread_coeff,  market_impact_fct, MI_coeff, queue_fct, queue_coeff, auct_prenium_fct, AP_coeff):
        self.spread = (spread_fct, spread_coeff)
        self.market_impact = (market_impact_fct, MI_coeff)
        self.queue = (queue_fct, queue_coeff)
        self.auction_premium = (auct_prenium_fct, AP_coeff)

    def compute_fill_price(self, context):
        self.context=context
        total=context["price"]
        for fct, coeff in [self.spread, self.market_impact, self.queue, self.auction_premium]:
            if fct is not None:
                total += coeff * fct(self.context)
        return total
    


#----------------------------------------------------------------SPREAD MODELS--------------------------------------------------------


def model_spread_CS(H1,L1,H2,L2, upper_bound, close, correct_overnight=True, return_vol=False):
    """ 
    This functions approximates the spread using OHCLV data, according to the Corwin–Schultz formula.

    With no access to level 2 data (bid-ask), its impossible to compute the exact spread.
    This function aims to estimate spread using the Corwin–Schultz Spread Estimator (2012).
    It is useful only for daily data, and can occasionaly give absurd values, which is why they're capped.
    !! It works well with liquid assets, less with illiquid ones !!

    H1 (float): highest price of day t
    L1 (float): lowest price od day t
    H2 (float): highest price of day t+1
    L2 (float): lowest price of day t+1
    upper_bound (int): caps spread value. More volatile stocks has a smaller higher cap (bps)
    correct_overnight (Bool): Corrects for the overnight variation in high-low ratio.
    return_vol (Bool): CS can also return estimated volatility. This argument lets the user chose to do it or not.
    
    """
    #close: arg optionnel
    if correct_overnight:
        if(H2)<close:
            diff=close-H2
            H2+=diff
            L2-=diff
        elif (L2>close):
            diff=L2-close
            H2-=diff
            L2-=diff



    H=max(H1,H2)
    L=min(L1,L2)
    gamma=pow(log(H/L),2)
    beta=pow(log(H1/L1),2)+pow(log(H2/L2),2)
    alpha=(
        (sqrt(2*beta)-sqrt(beta))/(3-2*sqrt(2))
        -
        sqrt(gamma/(3-2*sqrt(2)))
    )
    spread=2*(exp(alpha)-1)/(1+exp(alpha))

    """Capping values to a realistic range to correct absurd values."""
    spread=min(max(spread,0),upper_bound)

    if return_vol:
        k1=4*log(2)
        k2=sqrt(8/pi)
        vol=(
            (sqrt(beta/2)-sqrt(beta)/k2)
        )+(
            sqrt(gamma/pow(k2,2)*(3-2*sqrt(2)))
        )
        return spread, vol
    else: return spread











#------------------------------------------------------------MARKET IMPACT MODELS--------------------------------------------------------


def model_market_impact_sqrt(daily_volume, liquidity, order_size, volatility):
    """effective on low frequency strategies"""
    return volatility* sqrt(order_size/daily_volume)


def market_impact_Almgren_Chriss():
    pass
