import os
import numpy as np
import statsmodels.api as sm

from real_warfarin_pkpd_data_loader import generate_data as generate_warfarin_pkpd
from real_battery_nasa_data_loader import generate_data as generate_battery_nasa

os.makedirs("data", exist_ok=True)

# 1. Real Sunspots
data_ss = sm.datasets.sunspots.load_pandas().data
y_ss = data_ss['SUNACTIVITY'].values.reshape(-1, 1)
t_ss = np.arange(len(y_ss), dtype=float)
np.save("data/real_sunspots_ground_truth.npy", y_ss)
np.save("data/real_sunspots_time_points.npy", t_ss)
print("Saved real_sunspots")

# 2. Real Nile
data_nile = sm.datasets.nile.load_pandas().data
y_nile = data_nile['volume'].values.reshape(-1, 1)
t_nile = np.arange(len(y_nile), dtype=float)
np.save("data/real_nile_ground_truth.npy", y_nile)
np.save("data/real_nile_time_points.npy", t_nile)
print("Saved real_nile")

# 3. Real Macro (US Real GDP)
data_macro = sm.datasets.macrodata.load_pandas().data
y_macro = data_macro['realgdp'].values.reshape(-1, 1)
t_macro = np.arange(len(y_macro), dtype=float)
np.save("data/real_macro_ground_truth.npy", y_macro)
np.save("data/real_macro_time_points.npy", t_macro)
print("Saved real_macro")

# 4. Real Theophylline
t_theo = np.array([0.00, 0.25, 0.57, 1.12, 2.02, 3.82, 5.10, 7.03, 9.05, 12.12, 24.37], dtype=float)
y_theo = np.array([0.74, 2.84, 6.57, 10.50, 9.66, 8.58, 8.36, 7.47, 6.89, 5.94, 3.28], dtype=float).reshape(-1, 1)
np.save("data/real_theophylline_ground_truth.npy", y_theo)
np.save("data/real_theophylline_time_points.npy", t_theo)
print("Saved real_theophylline")

# 4b. Warfarin PK/PD fixture bridge
generate_warfarin_pkpd()
print("Saved real_warfarin_pkpd")

# 4c. NASA battery capacity fixture bridge
generate_battery_nasa()
print("Saved real_battery_nasa_capacity")

# 5. Real CO2 (Mauna Loa)
data_co2 = sm.datasets.co2.load_pandas().data
yearly = data_co2.resample('YE').mean().dropna()
y_co2 = yearly['co2'].values.reshape(-1, 1)
t_co2 = np.arange(len(y_co2), dtype=float)
np.save("data/real_co2_ground_truth.npy", y_co2)
np.save("data/real_co2_time_points.npy", t_co2)
print("Saved real_co2")
