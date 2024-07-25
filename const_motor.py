import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from prop_plotter import read_and_filter_data, read_reference_data, plot_difference_prop

def load_data(kv_values, directory='./P20x6'):
    data = []
    for kv in kv_values:
        df = pd.read_csv(f'{directory}/kv{kv}.csv')
        df['KV'] = kv
        data.append(df)
    return data

def fit_polynomials(data):
    poly_coeffs = {}
    for df in data:
        kv = df['KV'].iloc[0]
        coeffs = np.polyfit(df['Electrical power (W)'] * 6, df['Thrust (lbf)'] * 6, 3)
        poly_coeffs[kv] = coeffs
    return poly_coeffs

def interpolate_coefficients(poly_coeffs, target_kv):
    kv_values = sorted(poly_coeffs.keys())
    coeffs_matrix = np.array([poly_coeffs[kv] for kv in kv_values])
    interp_funcs = [interp1d(kv_values, coeffs_matrix[:, i], kind='linear', fill_value='extrapolate') for i in range(coeffs_matrix.shape[1])]
    target_coeffs = np.array([interp_func(target_kv) for interp_func in interp_funcs])
    return target_coeffs

def plot_polynomial_fit(poly_coeffs, target_coeffs, df_16, df_18):
    power_range = np.linspace(0, 2500, 500)
    plt.figure(figsize=(10, 6))

    # colors = ['orange', 'green', 'red', 'purple', 'brown']
    # for idx, (kv, coeffs) in enumerate(poly_coeffs.items()):
    #     poly_eqn = np.poly1d(coeffs)
    #     plt.plot(power_range, poly_eqn(power_range), color=colors[idx % len(colors)], label=f'KV{kv}')

    # Plot for 16in props
    hexa_thrust_16 = df_16['Thrust (lbf)'] * 6
    hexa_power_16 = df_16['Electrical power (W)'] * 6

    hexa_thrust_18 = df_18['Thrust (lbf)'] * 6
    hexa_power_18 = df_18['Electrical power (W)'] * 6

    hexa_coeffs_16 = np.polyfit(hexa_power_16, hexa_thrust_16, 3)
    hexa_poly_eqn_16 = np.poly1d(hexa_coeffs_16)
    power_range_16 = np.linspace(hexa_power_16.min(), hexa_power_16.max(), 500)
    plt.plot(power_range_16, hexa_poly_eqn_16(power_range_16), color='orange', label='P16x5.4 (current)')

    # Plot for 18in props
    hexa_coeffs_18 = np.polyfit(hexa_power_18, hexa_thrust_18, 3)
    hexa_poly_eqn_18 = np.poly1d(hexa_coeffs_18)
    power_range_18 = np.linspace(hexa_power_18.min(), 2050, 500)
    plt.plot(power_range_18, hexa_poly_eqn_18(power_range_18), color='green', label='P18x6.1')

    # Plot for 20in props
    target_poly_eqn = np.poly1d(target_coeffs)
    plt.plot(power_range, target_poly_eqn(power_range), color='blue', label='P20x6')

    # Plot the maximum engine power line
    plt.axvline(x=1654 * 0.5, color='blue', linestyle='--', label=r'Engine Power at 50% Efficiency')
    plt.axvline(x=1654 * 0.75, color='green', linestyle='--', label=r'Engine Power at 75% Efficiency')
    plt.axvline(x=1654, color='red', linestyle='--', label='Maximum Engine Power')

    # Adding the shaded horizontal region
    plt.axhspan(21, 26, color='yellow', alpha=0.3, label='Weight Estimate Region')

    plt.xlim(left=0, right=2000)
    plt.ylim(bottom=0, top=40)
    plt.xlabel('Electrical Power (W)')
    plt.ylabel('Thrust (lbf)')
    plt.title('Thrust vs Electrical Power for Different Propellers with kV490 Motors (Hexacopter)')
    plt.legend()
    plt.grid(True)
    plt.show()

    plot_difference_prop(hexa_poly_eqn_16, hexa_poly_eqn_18, target_poly_eqn, power_range_18)

# Directory containing your CSV files
directory = './P20x6'
csv_files = os.listdir(directory)

# Extract KV values from filenames
kv_values = [int(filename.split('.')[0][2:]) for filename in csv_files]

data = load_data(kv_values, directory)
poly_coeffs = fit_polynomials(data)

# Interpolate coefficients for the KV490 motor
target_kv = 490
target_coeffs = interpolate_coefficients(poly_coeffs, target_kv)

P16_file_name = 'prop_data/test2_arm4_CCW_noknob.csv' # User input
P18_file_name = 'P18x61/kV490.csv' # User input

df_16 = read_and_filter_data(P16_file_name)
df_18 = read_reference_data(P18_file_name)

plot_polynomial_fit(poly_coeffs, target_coeffs, df_16, df_18)
