# http://gouthamanbalaraman.com/blog/valuing-european-option-heston-model-quantLib.html

import QuantLib as ql
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps, cumtrapz, romb
import math

# option data
maturity_date = ql.Date(27, 7, 2022)
spot_price = 2.878
strike_price = 2.550
volatility = 0.1738 # the historical vols for a year
dividend_rate =  0
option_type = ql.Option.Call

risk_free_rate = 0.02256
day_count = ql.Actual365Fixed()
calendar = ql.China()

calculation_date = ql.Date(26, 7, 2022)
ql.Settings.instance().evaluationDate = calculation_date

# construct the European Option
payoff = ql.PlainVanillaPayoff(option_type, strike_price)
exercise = ql.EuropeanExercise(maturity_date)
european_option = ql.VanillaOption(payoff, exercise)

# construct the Heston process

v0 = 0.010466  # spot variance, need calibration
kappa = 26.822806 # strength of mean reversion, need calibration
theta = 0.539689  # mean of mean reversion, need calibration
sigma = 0.104909 # vol of vol, need calibration
rho = 0.082499 # correlation, need calibration

spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, risk_free_rate, day_count))
dividend_yield = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, dividend_rate, day_count))
heston_process = ql.HestonProcess(flat_ts,
                                  dividend_yield,
                                  spot_handle,
                                  v0,
                                  kappa,
                                  theta,
                                  sigma,
                                  rho)

engine = ql.AnalyticHestonEngine(ql.HestonModel(heston_process),0.01, 1000)
european_option.setPricingEngine(engine)
h_price = european_option.NPV()
print("The Heston model price is",h_price)