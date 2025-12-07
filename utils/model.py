from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class Locomotive:
    id: str
    max_tonnage: float
    cost_per_km: float

@dataclass
class Wagon:
    id: str
    type: str
    capacity_t: float
