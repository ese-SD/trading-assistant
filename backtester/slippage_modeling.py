from dataclasses import dataclass, field
from typing import Any
from math import log,sqrt,exp,pi
from functools import partial, partialmethod



@dataclass
class SlippageContext:
    """
    Dataclass that contains the necessary infos to model slippage.
    All the data that isnt common to all slippage models is stored in extra.
    """
    price: float
    order_size: int
    timestamp: str
    extra: dict[str, Any] = field(default_factory=dict)




class SlippageModel:
    """
    Models slippage when filling an order based on the order and current market conditions.

    User can customize and create their own slippage model by chosing different component function and their parameters.
    """

    def __init__(self, spread_fct=None, spread_coeff=0.0,
                 market_impact_fct=None, MI_coeff=0.0,
                 queue_fct=None, queue_coeff=0.0,
                 auct_prenium_fct=None, AP_coeff=0.0,
                 params=None):
        
        """
        Initialize a SlippageModel with optional components.

        Args:
            spread_fct (callable, optional): Function to model the spread component.
            spread_coeff (float, optional): Coefficient to scale the spread contribution. Defaults to 0.0.
            market_impact_fct (callable, optional): Function to model the market impact component.
            MI_coeff (float, optional): Coefficient to scale the market impact contribution. Defaults to 0.0.
            queue_fct (callable, optional): Function to model queue effects.
            queue_coeff (float, optional): Coefficient to scale the queue effect contribution. Defaults to 0.0.
            auct_prenium_fct (callable, optional): Function to model auction premium.
            AP_coeff (float, optional): Coefficient to scale the auction premium contribution. Defaults to 0.0.
            params (dict, optional): Dictionary of parameter dictionaries for each component.
                Example:
                    {
                        "spread": {"window": 10},
                        "mi": {"factor": 0.02}
                    }
        """
        
        params = params or {}

        partial_spread = partial(spread_fct, **params.get("spread", {})) if spread_fct else None
        partial_mi= partial(market_impact_fct, **params.get("mi", {})) if market_impact_fct else None
        partial_queue = partial(queue_fct, **params.get("queue", {})) if queue_fct else None
        partial_ap = partial(auct_prenium_fct, **params.get("ap", {})) if auct_prenium_fct else None
        self.spread = (partial_spread, spread_coeff)
        self.market_impact = (partial_mi, MI_coeff)
        self.queue = (partial_queue, queue_coeff)
        self.auction_premium = (partial_ap, AP_coeff)

    def compute_fill_price(self, context):
        """
        Compute the adjusted fill price given a trading context.

        Args:
            context (SlippageContext): Snapshot of trading conditions when filling the order.

        Returns:
            total (float): The adjusted fill price after applying all slippage components.
        """
        self.context=context
        total=context["price"]
        for fct, coeff in [self.spread, self.market_impact, self.queue, self.auction_premium]:
            if fct is not None:
                total += coeff * fct(context)
        return total
    


#----------------------------------------------------------------SPREAD MODELS--------------------------------------------------------


def model_spread_CS(context, upper_bound, close, correct_overnight=True, return_vol=False):
    """
    Estimate bid–ask spread using the Corwin–Schultz (2012) high–low spread estimator.

    Args:

    context (SlippageContext): Must contain in `context.extra["OCHLV"]` at least the last two days of OHLC data.
    upper_bound (float): Maximum allowed spread value (cap).
    close (float): Closing price of day t+1, used if applying the overnight correction.
    correct_overnight (bool, default=True): Adjusts for overnight price jumps between the previous close and next day's range.
    return_vol (bool, default=False): If True, also return the estimated volatility.

    Returns:
    spread (float):Estimated spread, capped between 0 and `upper_bound`.
    vol (float, optional): Estimated volatility (only if `return_vol=True`).

    Notes:
    Works best with liquid assets and daily data; may produce extreme values for illiquid assets.
    """

    # gets args from context
    H1=context.extra["OCHLV"][-2]["High"]
    H2=context.extra["OCHLV"][-1]["High"]
    L1=context.extra["OCHLV"][-2]["Low"]
    L2=context.extra["OCHLV"][-1]["Low"]



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


def model_market_impact_sqrt(context):
    """effective on low frequency strategies"""
    daily_volume=context.extra["OCHLV"][-1]["Volume"]
    volatility=0#?????
    return volatility* sqrt(context.order_size/daily_volume)


def market_impact_Almgren_Chriss(context):
    pass








#------------------------------------------------------------QUEUE MODELS--------------------------------------------------------




#------------------------------------------------------------AUCTION PRENIUM MODELS--------------------------------------------------------



