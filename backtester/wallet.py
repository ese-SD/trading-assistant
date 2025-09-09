from dataclasses import dataclass, field
from strategies.position import LongPosition

@dataclass
class Wallet:
    balance: float = 0.0
    positions: list[LongPosition]

    