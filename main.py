from src.model import BeliefModel
from src.utils import generate_piecewise_linear_irm, generate_linear_supply


# Model Parameters
investment_time = 1 / 12
std_beliefs = 0.05
mean_beliefs = -0.07
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

# Model instanciation
model = BeliefModel(
    investment_time=investment_time,
    std_beliefs=std_beliefs,
    mean_beliefs=mean_beliefs,
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

x, y = model.compute_equilibrium_utilization_ratios()
print("uc_eq = ", x, "us_eq = ", y)

x1, y1 = model.compute_market_utilization_ratios(x, y)
print("uc_market = ", x1, "us_market = ", y1)
