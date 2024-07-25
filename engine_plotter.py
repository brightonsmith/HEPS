import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

rpm_data = np.array([1800, 2400, 2600, 7000])
power_data = np.array([0, 140, 220, 1500])

torque_data = power_data / (rpm_data * (2 * np.pi / 60)) 

def power_band(rpm, a, b, c, d):
    return a * rpm**3 + b * rpm**2 + c * rpm + d

popt, _ = curve_fit(power_band, rpm_data, power_data)


rpm_smooth = np.linspace(1800, 7000, 500)
power_smooth = power_band(rpm_smooth, *popt) # Power in W
torque_smooth = power_smooth / (rpm_smooth * (2 * np.pi / 60))  # Torque in Nm

fig, ax1 = plt.subplots(figsize=(10, 6))

# Power vs RPM
color_power = 'tab:blue'
ax1.set_xlabel('RPM')
ax1.set_ylabel('Power (W)', color=color_power)
line1, = ax1.plot(rpm_smooth, power_smooth, label='Power', color=color_power)
points1 = ax1.scatter(rpm_data, power_data, color=color_power)
ax1.tick_params(axis='y', labelcolor=color_power)
ax1.grid(True)

# Torque vs RPM
ax2 = ax1.twinx()  
color_torque = 'tab:green'
ax2.set_ylabel('Torque (Nm)', color=color_torque)
line2, = ax2.plot(rpm_smooth, torque_smooth, label='Torque', color=color_torque)
points2 = ax2.scatter(rpm_data, torque_data, color=color_torque)
ax2.tick_params(axis='y', labelcolor=color_torque)

# Combine legends for both Power and Torque
lines = [line1, points1, line2, points2]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper left')  # Unified legend location

# Title and layout adjustments
plt.title('Saito FA-125a Power and Torque Curves')
fig.tight_layout()

plt.show()