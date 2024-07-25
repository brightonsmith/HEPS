import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Provided RPM and Power data points
rpm_data = np.array([1800, 2400, 2600, 7000])
power_data = np.array([0, 140, 220, 1500])

# Define a function to represent a typical 4-stroke engine power band
def power_band(rpm, a, b, c, d):
    return a * rpm**3 + b * rpm**2 + c * rpm + d

# Use curve fitting to find the best fit parameters
popt, _ = curve_fit(power_band, rpm_data, power_data)

# Generate RPM values for plotting the smooth curve
rpm_smooth = np.linspace(1800, 7000, 500)
power_smooth = power_band(rpm_smooth, *popt) # Power in W

# Define kV ratings and calculate corresponding RPM values
kv_ratings = np.array([90, 100, 200, 225, 250, 275])  # kV ratings of alternators
voltage = 24.0  # Voltage in volts
rpm_kv = kv_ratings * voltage  # RPM = kV * Voltage

# Interpolate power values for the calculated RPM
power_kv = power_band(rpm_kv, *popt)

# Plot the power curve
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlabel('RPM')
ax.set_ylabel('Power (W)')
ax.plot(rpm_smooth, power_smooth, label='Power Curve', color='tab:blue')
ax.grid(True)

# Plot and label the kV points on the curve
for i, kv in enumerate(kv_ratings):
    ax.scatter(rpm_kv[i], power_kv[i], color='tab:red')
    ax.annotate(rf'{kv} $K_v$', (rpm_kv[i], power_kv[i]), 
                textcoords="offset points", xytext=(0,10), ha='center', color='tab:red')

# Title and legend
plt.title(rf'Saito FA-125a Power Curve with Alternator $K_v$ Points at Fixed Voltage ({voltage} V)')
plt.legend(['Power Curve', r'Alternator  $K_v$ Points'], loc='upper left')
plt.tight_layout()

plt.show()
