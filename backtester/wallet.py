from dataclasses import dataclass, field

@dataclass
class Wallet:
    cash: float = 0.0
    # Ticker : number of that share held.
    # Can be negative (for short positions)
    stocks: dict[str:float]

    