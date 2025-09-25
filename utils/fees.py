#handles the various types of fees structures uses by different brookers.
from abc import ABC, abstractmethod
import bisect



class FeeStructure(ABC):
    def __init__(self, application: str):
        if application not in ("on_top", "deducted"):
            raise ValueError("application must be 'on_top' or 'deducted'")
        self.application = application

    @abstractmethod
    def compute_fee(self, nb_shares: float, share_price: float) -> float:
        pass

    #specifier fees achat et vente?





class Linear(FeeStructure):
    def __init__(self, application, flat_fee, linear_fee):
        self.flat_fee=flat_fee
        self.linear_fee=linear_fee
    
    def compute_fee(self, nb_shares, share_price):
        return share_price * nb_shares* self.linear_fee+ self.flat_fee

class TieredFees(FeeStructure):
    def __init__(self, tiers: dict[int, float]):
        #tiers is a dict with a nb of shares as keys, and a % of fee as values; the last key should always be +inf
        self.tiers = tiers
        self.thresholds = sorted(tiers.keys())

    def compute_fee(self, nb_shares, share_price):
        pos = bisect.bisect_right(self.thresholds, nb_shares)
        threshold = self.thresholds[pos]
        return self.tiers[threshold]


class PerShare(FeeStructure):
    def __init__(self, fee_per_share):
        self.fee_per_share=fee_per_share

    def compute_fee(self, nb_shares, share_price):
        return self.fee_per_share*nb_shares
 