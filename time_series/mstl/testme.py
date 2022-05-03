'''
Created on May 3, 2022

@author: zollen
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import MSTL

sns.set_style("darkgrid")
sns.set_context("poster")


url = "https://raw.githubusercontent.com/tidyverts/tsibbledata/master/data-raw/vic_elec/VIC2015/demand.csv"
df = pd.read_csv(url)

df["Date"] = df["Date"].apply(
    lambda x: pd.Timestamp("1899-12-30") + pd.Timedelta(x, unit="days")
)

df["ds"] = df["Date"] + pd.to_timedelta((df["Period"] - 1) * 30, unit="m")

timeseries = df[["ds", "OperationalLessIndustrial"]]
timeseries.columns = [
    "ds",
    "y",
]  # Rename to OperationalLessIndustrial to y for simplicity.

# Filter for first 149 days of 2012.
start_date = pd.to_datetime("2012-01-01")
end_date = start_date + pd.Timedelta("149D")
mask = (timeseries["ds"] >= start_date) & (timeseries["ds"] < end_date)
timeseries = timeseries[mask]

# Resample to hourly
timeseries = timeseries.set_index("ds").resample("H").sum()

print(timeseries.head())



# Compute date time variables used later in plotting
timeseries["week"] = timeseries.index.isocalendar().week
timeseries["day_of_month"] = timeseries.index.day
timeseries["month"] = timeseries.index.month

if False:
    # Plot the time series
    ax = timeseries.plot(y="y", figsize=[20, 10], legend=False)
    ax.set_ylabel("Demand (MW)")
    ax.set_xlabel("Time")
    ax.set_title("Electricity demand in Victoria, Australia")
    plt.tight_layout()   
    plt.show()


mstl = MSTL(timeseries["y"], periods=(24, 24 * 7), stl_kwargs={"seasonal_deg": 0})
res = mstl.fit() 

print(res.trend.head())
print(res.seasonal.head())
print(res.resid.head())

if True:
    # Start with the plot from the results object `res`
    plt.rc("figure", figsize=(16, 20))
    plt.rc("font", size=13)
    fig = res.plot()
    
    # Make plot pretty
    axs = fig.get_axes()
    
    ax_last = axs[-1]
    ax_last.xaxis.set_ticks(pd.date_range(start="2012-01-01", freq="MS", periods=5))
    plt.setp(ax_last.get_xticklabels(), rotation=0, horizontalalignment="center")
    for ax in axs[:-1]:
        ax.get_shared_x_axes().join(ax, ax_last)
        ax.xaxis.set_ticks(pd.date_range(start="2012-01-01", freq="MS", periods=5))
        ax.set_xticklabels([])
    axs[0].set_ylabel("y")
    axs[0].set_title("Time series decomposition of electricity demand")
    ax_last.set_xlabel("Time")
    
    plt.tight_layout()
    plt.show()