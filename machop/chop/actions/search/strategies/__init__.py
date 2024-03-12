# from .rl import StrategyRL
from chop.actions.search.strategies.brute_force.brute_force import SearchStrategyBruteForce
from .optuna import SearchStrategyOptuna
from .base import SearchStrategyBase

SEARCH_STRATEGY_MAP = {
    # "rl": StrategyRL,
    "optuna": SearchStrategyOptuna,
    "brute_force": SearchStrategyBruteForce,
}


def get_search_strategy_cls(name: str) -> SearchStrategyBase:
    if name not in SEARCH_STRATEGY_MAP:
        raise ValueError(
            f"{name} must be defined in {list(SEARCH_STRATEGY_MAP.keys())}."
        )
    return SEARCH_STRATEGY_MAP[name]