# http://gouthamanbalaraman.com/blog/heston-calibration-scipy-optimize-quantlib-python.html

import QuantLib as ql
from math import pow, sqrt
import numpy as np
from scipy.optimize import root
import pandas as pd
import numpy as np

day_count = ql.Actual365Fixed()
calendar = ql.China()
calculation_date = ql.Date(26, 7, 2022)

spot = 2.878
ql.Settings.instance().evaluationDate = calculation_date

risk_free_rate = 0.02256
dividend_rate = 0.0
yield_ts = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, risk_free_rate, day_count))
dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, dividend_rate, day_count))

expiration_dates = ql.Date(27,7,2022)
raw_data = pd.read_csv("C:/Users/84637/Desktop/option call.csv")
strikes = list(raw_data['行权价'])
data = list(raw_data['IV'])


def setup_helpers(engine, expiration_dates, strikes, 
                  data, ref_date, spot, yield_ts, 
                  dividend_ts):
    heston_helpers = []
    grid_data = []
    for j, s in enumerate(strikes):
        t = (expiration_dates - ref_date )
        p = ql.Period(t, ql.Days)
        vols = data[j]
        helper = ql.HestonModelHelper(
            p, calendar, spot, s, 
            ql.QuoteHandle(ql.SimpleQuote(vols)),
            yield_ts, dividend_ts)
        helper.setPricingEngine(engine)
        heston_helpers.append(helper)
        grid_data.append((expiration_dates, s))
    return heston_helpers, grid_data

def cost_function_generator(model, helpers,norm=False):
    def cost_function(params):
        params_ = ql.Array(list(params))
        model.setParams(params_)
        error = [h.calibrationError() for h in helpers]
        if norm:
            return np.sqrt(np.sum(np.abs(error)))
        else:
            return error
    return cost_function

def calibration_report(helpers, grid_data, detailed=False):
    avg = 0.0
    if detailed:
        print("%15s %25s %15s %15s %20s" % (
            "Strikes", "Expiry", "Market Value", 
             "Model Value", "Relative Error (%)"))
        print("="*100)
    for i, opt in enumerate(helpers):
        err = (opt.modelValue()/opt.marketValue() - 1.0)
        date,strike = grid_data[i]
        if detailed:
            print("%15.2f %25s %14.5f %15.5f %20.7f " % (
                strike, str(date), opt.marketValue(), 
                opt.modelValue(), 
                100.0*(opt.modelValue()/opt.marketValue() - 1.0)))
        avg += abs(err)
    avg = avg*100.0/len(helpers)
    if detailed: 
        print("-"*100)
    summary = "Average Abs Error (%%) : %5.9f" % (avg)
    print(summary)
    return avg
    
def setup_model(_yield_ts, _dividend_ts, _spot, 
                init_condition=(0.02,0.2,0.5,0.1,0.01)):
    theta, kappa, sigma, rho, v0 = init_condition
    process = ql.HestonProcess(_yield_ts, _dividend_ts, 
                           ql.QuoteHandle(ql.SimpleQuote(_spot)), 
                           v0, kappa, theta, sigma, rho)
    model = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(model) 
    return model, engine
summary= []

# theta, kappa, sigma, rho, v0 = (0.02, 0.2, 0.5, 0.1, 0.01)
# theta, kappa, sigma, rho, v0 = (0.07, 0.5, 0.1, 0.1, 0.1)

model1, engine1 = setup_model(yield_ts, dividend_ts, spot, init_condition=(0.02,0.2,0.5,0.1,0.01))
heston_helpers1, grid_data1 = setup_helpers(engine1, expiration_dates, strikes, data, calculation_date, spot, yield_ts, dividend_ts)
initial_condition = list(model1.params())

lm = ql.LevenbergMarquardt(1e-8, 1e-8, 1e-8)
model1.calibrate(heston_helpers1, lm, ql.EndCriteria(500, 300, 1.0e-8,1.0e-8, 1.0e-8))
theta, kappa, sigma, rho, v0 = model1.params()
print("theta = %f, kappa = %f, sigma = %f, rho = %f, v0 = %f" % (theta, kappa, sigma, rho, v0))
error = calibration_report(heston_helpers1, grid_data1)
summary.append(["QL LM1", error] + list(model1.params()))