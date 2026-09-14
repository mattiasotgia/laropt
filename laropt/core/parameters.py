
from dataclasses import dataclass
from typing import Literal, List, Dict, Any


@dataclass
class ParamSpec:
    name: str
    low: float
    high: float
    kind: Literal['float', 'int']
    size: int = 1
    decimals: int = 2
    fcl_paths: List[str] = None
    is_per_plane: bool = False

    def sample(self, x: float):
        '''
        Sample a value given a normalized x in [0,1].
        For ints, ensures uniform discrete coverage instead of rounding.
        '''
        if self.kind == 'int':
            n_bins = int(self.high - self.low + 1)
            idx = int(x * n_bins)   # map x to integer bin
            idx = min(idx, n_bins - 1)  # handle x==1 edge
            val = self.low + idx
        else:
            val = self.low + x * (self.high - self.low)
            val = round(val, self.decimals)

        if self.size == 1:
            return val
        return [val] * self.size


@dataclass
class ResultSpec:
    name: str
    dtype: Literal['REAL', 'INTEGER'] = 'REAL'   # or INTEGER


class ParamSet:
    def __init__(self, values: Dict[str, Any]):
        self.values = values

    def to_flat_dict(self):
        flat = {}
        for k, v in self.values.items():
            if isinstance(v, list):
                for i, val in enumerate(v):
                    flat[f'{k}_{i}'] = val
            else:
                flat[k] = v
        return flat

    def __str__(self):
        lines = ["ParamSet:"]
        for k, v in self.values.items():
            if isinstance(v, list):
                vals = ", ".join(str(x) for x in v)
                lines.append(f"  {k}: [{vals}]")
            else:
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)