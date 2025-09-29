from dataclasses import dataclass, field
from typing import Any

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
