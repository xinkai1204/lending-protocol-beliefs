import numpy as np
import pandas as pd

from src.model import BeliefModel
from src.utils import generate_piecewise_linear_irm, generate_linear_supply

investment_time = 1 / 12
std_beliefs = 0.05
traders_initial_budget = 1.0
std_crypto_returns = 0.02
gamma = 0.5
hc = 0.30
hs = 0.30

irm_crypto = generate_piecewise_linear_irm(0.92, 2.35 / 100, 10.35 / 100)
irm_stablecoin = generate_piecewise_linear_irm(0.92, 5 / 100, 15 / 100)

rho_c = 0.95
rho_s = 0.95

exogenous_supply_crypto = generate_linear_supply(1, 20)
exogenous_supply_stablecoin = generate_linear_supply(1, 20)

size_pop_traders = 250


def solve_equilibrium_for_mean(mean_beliefs: float) -> dict:
    model = BeliefModel(
        investment_time=investment_time,
        std_beliefs=std_beliefs,
        mean_beliefs=float(mean_beliefs),
        traders_initial_budget=traders_initial_budget,
        std_crypto_returns=std_crypto_returns,
        gamma=gamma,
        hc=hc,
        hs=hs,
        irm_crypto=irm_crypto,
        irm_stablecoin=irm_stablecoin,
        rho_c=rho_c,
        rho_s=rho_s,
        exogenous_supply_crypto=exogenous_supply_crypto,
        exogenous_supply_stablecoin=exogenous_supply_stablecoin,
        size_pop_traders=size_pop_traders,
    )

    uc_eq, us_eq = model.compute_equilibrium_utilization_ratios()

    rc_eq = model.crypto_interest_rate(uc_eq)
    rs_eq = model.stablecoin_interest_rate(us_eq)

    uc_mkt, us_mkt = model.compute_market_utilization_ratios(uc_eq, us_eq)

    return {
        "mean_beliefs": float(mean_beliefs),
        "uc_eq": float(uc_eq),
        "us_eq": float(us_eq),
        "rc_eq": float(rc_eq),
        "rs_eq": float(rs_eq),
        "uc_market_check": float(uc_mkt),
        "us_market_check": float(us_mkt),
        "uc_gap": float(uc_mkt - uc_eq),
        "us_gap": float(us_mkt - us_eq),
    }


def main():
    # ============================================================
    # Comparative statics
    # ============================================================
    mean_grid = np.linspace(-0.20, 0.20, 20)  # e.g., from -20% to +20%

    rows = []
    for mb in mean_grid:
        try:
            rows.append(solve_equilibrium_for_mean(mb))
        except Exception as e:
            rows.append({"mean_beliefs": float(mb), "error": str(e)})

    df = pd.DataFrame(rows)

    cols_show = ["mean_beliefs", "uc_eq", "us_eq", "rc_eq", "rs_eq", "uc_gap", "us_gap"]
    print(df[cols_show].to_string(index=False))

    # Save results
    df.to_csv("comparative_statics_mean_beliefs.csv", index=False)
    print("\nSaved: comparative_statics_mean_beliefs.csv")

    # Plots
    try:
        import matplotlib.pyplot as plt

        ok = df.dropna(subset=["uc_eq", "us_eq", "rc_eq", "rs_eq"]).copy()

        plt.figure()
        plt.plot(ok["mean_beliefs"], ok["uc_eq"], marker="o")
        plt.xlabel("mean_beliefs")
        plt.ylabel("uc_eq")
        plt.title("Equilibrium crypto utilization vs mean_beliefs")
        plt.show()

        plt.figure()
        plt.plot(ok["mean_beliefs"], ok["us_eq"], marker="o")
        plt.xlabel("mean_beliefs")
        plt.ylabel("us_eq")
        plt.title("Equilibrium stablecoin utilization vs mean_beliefs")
        plt.show()

        plt.figure()
        plt.plot(ok["mean_beliefs"], ok["rc_eq"], marker="o")
        plt.xlabel("mean_beliefs")
        plt.ylabel("rc_eq")
        plt.title("Equilibrium crypto interest rate vs mean_beliefs")
        plt.show()

        plt.figure()
        plt.plot(ok["mean_beliefs"], ok["rs_eq"], marker="o")
        plt.xlabel("mean_beliefs")
        plt.ylabel("rs_eq")
        plt.title("Equilibrium stablecoin interest rate vs mean_beliefs")
        plt.show()

    except Exception as e:
        print("\nPlotting skipped:", e)


if __name__ == "__main__":
    main()
