# Model class containing all parameters
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar, root
from typing import Callable
from src.utils import sigmoid


class Trader:
    """
    Represents an individual trader characterized by beliefs, risk preferences,
    and portfolio allocation between crypto and stablecoin positions.

    Parameters
    ----------
    uuid : int
        Unique identifier of the trader.
    mu : float
        Trader's expected return (belief) on the crypto asset.
    weight : float
        Weight of the trader in the population (e.g. probability mass).
    wc : float
        Portfolio weight allocated to crypto (can be negative for borrowing).
    ws : float
        Portfolio weight allocated to stablecoin (can be negative for borrowing).

    Raises
    ------
    TypeError
        If any input argument has an invalid type.
    """

    def __init__(self, uuid: float, mu: float, weight: float, wc: float, ws: float):
        if not isinstance(uuid, int):
            raise TypeError("uuid must be an int")
        if not isinstance(mu, float):
            raise TypeError("mu must be an float")
        if not isinstance(weight, float):
            raise TypeError("weight must be an float")
        if not isinstance(wc, float):
            raise TypeError("wc must be an float")
        if not isinstance(ws, float):
            raise TypeError("ws must be an float")
        self.uuid = uuid
        self.mu = mu
        self.weight = weight
        self.wc = wc
        self.ws = ws


class BeliefModel:
    """
    Model of a lending protocol with heterogeneous traders holding beliefs
    over crypto returns. The model determines portfolio choices, market
    utilization rates, and equilibrium interest rates.

    The equilibrium is defined as a fixed point where traders' optimal
    portfolio decisions are consistent with market utilization ratios.
    """

    def __init__(
        self,
        investment_time: float = 1 / 12,
        mean_beliefs: float = 0.05,
        std_beliefs: float = 1.0,
        traders_initial_budget: float = 1.0,
        std_crypto_returns: float = 0.5,
        gamma: float = 0.1,
        hc: float = 0.30,
        hs: float = 0.30,
        irm_crypto: Callable[[float], float] = lambda x: 0.1 * x,
        irm_stablecoin: Callable[[float], float] = lambda x: 0.1 * x,
        rho_c: float = 0.05,
        rho_s: float = 0.05,
        exogenous_supply_crypto: Callable[[float], float] = lambda x: 0.1 + 0.1 * x,
        exogenous_supply_stablecoin: Callable[[float], float] = lambda x: 0.1 + 0.1 * x,
        size_pop_traders: int = 250,
    ):
        """
        Initializes the belief-based lending model.

        Parameters
        ----------
        investment_time : float, optional
            Investment horizon (in years), default is 1/12.
        mean_beliefs : float, optional
            Mean of traders' belief distribution on crypto returns.
        std_beliefs : float, optional
            Standard deviation of traders' belief distribution.
        traders_initial_budget : float, optional
            Initial budget allocated to each trader.
        std_crypto_returns : float, optional
            Volatility of crypto returns.
        gamma : float, optional
            Risk aversion coefficient.
        hc : float, optional
            Over-collateralisation ratio for crypto collateral (between 0 and 1).
        hs : float, optional
            Over-collateralisation ratio for stablecoin collateral (between 0 and 1).
        irm_crypto : Callable[[float], float], optional
            Interest rate model for crypto lending as a function of utilization.
        irm_stablecoin : Callable[[float], float], optional
            Interest rate model for stablecoin lending as a function of utilization.
        rho_c : float, optional
            Fraction of the crypto interests that is effectively distributed to lenders.
        rho_s : float, optional
            Fraction of the collateral interests that is effectively distributed to lenders.
        exogenous_supply_crypto : Callable[[float], float], optional
            Exogenous supply of crypto as a function of the interest rate.
        exogenous_supply_stablecoin : Callable[[float], float], optional
            Exogenous supply of stablecoin as a function of the interest rate.
        size_pop_traders : int, optional
            Number of representative traders in the population (each representative
            trader is weighted by a probability mass so that the effective number of traders
            is one).

        Raises
        ------
        ValueError
            If parameters violate admissible economic or mathematical constraints.
        """
        self.investment_time = investment_time
        # Traders parameters
        if not (0 < hc < 1):
            raise ValueError(f"hc must be between 0 and 1, got {hc}")
        self.hc = hc

        if not (0 < hs < 1):
            raise ValueError(f"hs must be between 0 and 1, got {hs}")
        self.hs = hs

        if not (0 < rho_c < 1):
            raise ValueError(f"rho_c must be between 0 and 1, got {hc}")
        self.rho_c = rho_c

        if not (0 < rho_s < 1):
            raise ValueError(f"rho_s must be between 0 and 1, got {hs}")
        self.rho_s = rho_s

        if std_crypto_returns <= 0:
            raise ValueError(
                f"variance_crypto_returns must be positive, got {std_crypto_returns}"
            )
        self.std_crypto_returns = std_crypto_returns

        self.mean_beliefs = mean_beliefs

        self.traders_initial_budget = traders_initial_budget

        if std_beliefs <= 0:
            raise ValueError(f"variance_beliefs must be positive, got {std_beliefs}")
        self.std_beliefs = std_beliefs
        if gamma <= 0:
            raise ValueError(f"gamma should be positive, got {gamma}")
        self.gamma = gamma
        self.traders_population = _create_traders_population(
            size_pop=size_pop_traders,
            mean_beliefs=mean_beliefs,
            std_beliefs=std_beliefs,
        )

        self.irm_crypto = irm_crypto
        self.irm_stablecoin = irm_stablecoin

        # Exogenous supply parameters
        self.exogenous_supply_crypto = exogenous_supply_crypto
        self.exogenous_supply_stablecoin = exogenous_supply_stablecoin

    def crypto_interest_rate(self, uc: float) -> float:
        """
        Computes the crypto interest rate given the utilization ratio.

        Parameters
        ----------
        uc : float
            Utilization ratio of the crypto market (between 0 and 1).

        Returns
        -------
        float
            Crypto lending interest rate.

        Raises
        ------
        ValueError
            If utilization is outside [0, 1] or the resulting rate is negative.
        """
        if not (0 <= uc <= 1):
            raise ValueError(f"uc must be between 0 and 1, got {uc}")
        rc = self.irm_crypto(uc)
        if rc < 0:
            raise ValueError(f"rc must be positive, got {rc}")
        return rc

    def stablecoin_interest_rate(self, us: float) -> float:
        if not (0 <= us <= 1):
            raise ValueError(f"us must be between 0 and 1, got {us}")
        rs = self.irm_crypto(us)
        if rs < 0:
            raise ValueError(f"rs must be positive, got {rs}")
        return rs

    def solve_traders_problem(self, uc: float, us: float) -> None:
        """
        Solves each trader's portfolio optimization problem given utilization ratios.

        This method updates traders' portfolio weights and computes the aggregated
        borrowing and collateral levels in the protocol.

        Parameters
        ----------
        uc : float
            Crypto utilization ratio.
        us : float
            Stablecoin utilization ratio.

        Raises
        ------
        ValueError
            If utilization ratios are outside [0, 1].
        """
        if not ((0 <= uc <= 1) and (0 <= us <= 1)):
            raise ValueError(f"uc and us must be between 0 and 1, got {uc} and {us}")
        for trader in self.traders_population:
            self._optimize_trader_portfolio(trader=trader, uc=uc, us=us)

        (
            borrow_crypto,
            collat_crypto,
            collat_stablecoin,
            borrow_stablecoin,
        ) = _compute_aggregated_borrows_and_collateral(
            self.traders_population, self.traders_initial_budget
        )

        self.borrow_crypto = borrow_crypto
        self.collat_crypto = collat_crypto
        self.collat_stablecoin = collat_stablecoin
        self.borrow_stablecoin = borrow_stablecoin

    def _optimize_trader_portfolio(self, trader: Trader, uc: float, us: float) -> None:
        """
        Solves the individual trader's optimal portfolio allocation problem.

        The trader maximizes a mean-variance utility subject to borrowing and
        collateral constraints induced by utilization ratios.

        Parameters
        ----------
        trader : Trader
            Trader whose portfolio is optimized.
        uc : float
            Crypto utilization ratio.
        us : float
            Stablecoin utilization ratio.
        """
        rc = self.crypto_interest_rate(uc)
        rs = self.stablecoin_interest_rate(us)
        t = self.investment_time
        s = self.std_crypto_returns

        def f(w):
            tau_c = (self.rho_c * uc * max(w, 0) - max(-w, 0)) * rc
            tau_s = (self.rho_s * us * max(1 - w, 0) - max(w - 1, 0)) * rs
            return (
                1
                + (w * trader.mu + tau_c + tau_s) * t
                - 0.5 * self.gamma * t * s**2 * w**2
            )

        opt = minimize_scalar(
            lambda x: -f(x), bounds=(-1 / self.hs, 1 + 1 / self.hc), method="bounded"
        )
        trader.wc = opt.x
        trader.ws = 1 - opt.x

    def compute_market_utilization_ratios(
        self, uc: float, us: float
    ) -> tuple[float, float]:
        """
        Computes market utilization ratios implied by traders' optimal behavior and
        exogenous lending:

        utiliszation = borrow / (collat + exogenous_lending)

        Parameters
        ----------
        uc : float
            Initial crypto utilization ratio.
        us : float
            Initial stablecoin utilization ratio.

        Returns
        -------
        tuple[float, float]
            Updated utilization ratios (uc_market, us_market).
        """
        self.solve_traders_problem(uc=uc, us=us)
        rc = self.crypto_interest_rate(uc)
        rs = self.crypto_interest_rate(us)
        lending_crypto = self.exogenous_supply_crypto(rc)
        lending_stablecoin = self.exogenous_supply_stablecoin(rs)
        uc_market = min(
            1, max(0, self.borrow_crypto / (self.collat_crypto + lending_crypto))
        )
        us_market = min(
            1,
            max(
                0,
                self.borrow_stablecoin / (self.collat_stablecoin + lending_stablecoin),
            ),
        )
        return uc_market, us_market

    def compute_equilibrium_utilization_ratios(self):
        """
        Computes equilibrium utilization ratios as a fixed point.

        The equilibrium is defined by utilization ratios that are consistent
        with traders' optimal portfolio choices and market clearing.

        Returns
        -------
        tuple[float, float]
            Equilibrium crypto and stablecoin utilization ratios.
        """
        uc_0, us_0 = 0.5, 0.5

        def f(z):
            zc, zs = z
            uc, us = sigmoid(zc), sigmoid(zs)
            market_uc, market_us = self.compute_market_utilization_ratios(uc, us)
            return [market_uc - uc, market_us - us]

        equilibrium = root(f, x0=[uc_0, us_0], method="hybr")
        zc_eq, zs_eq = equilibrium.x
        return sigmoid(zc_eq), sigmoid(zs_eq)


def _compute_aggregated_borrows_and_collateral(
    traders: list[Trader], traders_initial_budget: float
):
    borrow_crypto = 0
    collat_crypto = 0
    borrow_stablecoin = 0
    collat_stablecoin = 0

    for trader in traders:
        borrow_crypto += max(-trader.wc, 0) * trader.weight
        collat_crypto += max(trader.wc, 0) * trader.weight
        borrow_stablecoin += max(-trader.ws, 0) * trader.weight
        collat_stablecoin += max(trader.ws, 0) * trader.weight

    borrow_crypto *= traders_initial_budget
    collat_crypto *= traders_initial_budget
    borrow_stablecoin *= traders_initial_budget
    collat_stablecoin *= traders_initial_budget

    return (
        borrow_crypto,
        collat_crypto,
        collat_stablecoin,
        borrow_stablecoin,
    )


def _create_traders_population(
    size_pop: int, mean_beliefs: float, std_beliefs: float
) -> list[Trader]:
    """
    Generates a population of traders with heterogeneous beliefs.

    Beliefs are discretized over a normal distribution support and weighted
    by the corresponding CDF mass.

    Parameters
    ----------
    size_pop : int
        Number of traders.
    mean_beliefs : float
        Mean belief about crypto returns.
    std_beliefs : float
        Standard deviation of beliefs.

    Returns
    -------
    list[Trader]
        List of initialized traders.
    """
    support = np.linspace(
        mean_beliefs - 3 * std_beliefs, mean_beliefs + 3 * std_beliefs, size_pop
    )
    traders = []
    for i, k in enumerate(support):
        traders.append(
            Trader(
                i,
                k,
                norm.cdf(k, loc=mean_beliefs, scale=std_beliefs),
                float("nan"),
                float("nan"),
            )
        )
    return traders
