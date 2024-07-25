import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.interpolate import CubicSpline

num_props = 6 # User input

# Function to read data and filter based on ESC throttle
def read_and_filter_data(file_name):
    df = pd.read_csv(file_name)
    
    # Find the index where ESC throttle starts decreasing
    for i in range(1, len(df)):
        if df['ESC (µs)'].iloc[i] < df['ESC (µs)'].iloc[i - 1 ] - 5:
            df = df[:i]
            break
    
    return df

# Function to read reference data
def read_reference_data(file_name):
    ref_df = pd.read_csv(file_name)
    return ref_df

# Function to convert decimal hours to hh:mm:ss
def hours_to_hms(y, pos):
    h = int(y)
    m = int((y - h) * 60)
    s = int((y * 3600) % 60)
    return f'{h:02}:{m:02}:{s:02}'

def plot_difference_uav(quad_poly_eqn, hexa_poly_eqn, octa_poly_eqn, power_range):
    quad_thrust = quad_poly_eqn(power_range)
    hexa_thrust = hexa_poly_eqn(power_range)
    octa_thrust = octa_poly_eqn(power_range)

    plt.figure(figsize=(10, 6))
    plt.plot(power_range, octa_thrust - quad_thrust, label=r'Octacopter $-$ Quadcopter')
    plt.plot(power_range, octa_thrust - hexa_thrust, label=r'Octacopter $-$ Hexacopter')
    plt.plot(power_range, hexa_thrust - quad_thrust, label=r'Hexacopter $-$ Quadcopter')

    plt.axvline(x=1654*0.5, color='blue', linestyle='--', label=r'Engine Power at 50% Efficiency')
    plt.axvline(x=1654*0.75, color='green', linestyle='--', label=r'Engine Power at 75% Efficiency')
    plt.axvline(x=1654, color='red', linestyle='--', label='Maximum Engine Power')

    plt.xlim(left=0, right=2500)
    plt.ylim(bottom=0, top=8)
    plt.xlabel('Electrical Power (W)')
    plt.ylabel('Thrust Difference (lbf)')
    plt.title('Thrust Difference vs Electrical Power for Different Configurations')
    plt.legend()
    plt.grid(True)
    plt.show()

def uav_config(df):
    quad_thrust = df['Thrust (lbf)'] * 4
    quad_power = df['Electrical power (W)'] * 4

    hexa_thrust = df['Thrust (lbf)'] * 6
    hexa_power = df['Electrical power (W)'] * 6
    
    octa_thrust = df['Thrust (lbf)'] * 8
    octa_power = df['Electrical power (W)'] * 8

    plt.figure(figsize=(10, 6))

    # Quad plot
    quad_coeffs = np.polyfit(quad_power, quad_thrust, 3)
    quad_poly_eqn = np.poly1d(quad_coeffs)
    power_range = np.linspace(quad_power.min(), quad_power.max(), 500)
    plt.plot(power_range, quad_poly_eqn(power_range), color='orange', label='Quadcopter')

    # Hexa plot
    hexa_coeffs = np.polyfit(hexa_power, hexa_thrust, 3)
    hexa_poly_eqn = np.poly1d(hexa_coeffs)
    power_range = np.linspace(hexa_power.min(), hexa_power.max(), 500)
    plt.plot(power_range, hexa_poly_eqn(power_range), color='green', label='Hexacopter')

    # Octa plot
    octa_coeffs = np.polyfit(octa_power, octa_thrust, 3)
    octa_poly_eqn = np.poly1d(octa_coeffs)
    power_range = np.linspace(octa_power.min(), octa_power.max(), 500)
    plt.plot(power_range, octa_poly_eqn(power_range), color='blue', label='Octacopter')

    # Plot the maximum engine power line
    plt.axvline(x=1654*0.5, color='blue', linestyle='--', label=r'Engine Power at 50% Efficiency')
    plt.axvline(x=1654*0.75, color='green', linestyle='--', label=r'Engine Power at 75% Efficiency')
    plt.axvline(x=1654, color='red', linestyle='--', label='Maximum Engine Power')

    # Adding the shaded horizontal region
    plt.axhspan(21, 26, color='yellow', alpha=0.3, label='Weight Estimate Region')

    plt.xlim(left=0, right=2500)
    plt.ylim(bottom=0, top=40)
    plt.xlabel('Electrical Power (W)')
    plt.ylabel('Thrust (lbf)')
    plt.title('Thrust vs Electrical Power for Different Configurations')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot difference curves
    plot_difference_uav(quad_poly_eqn, hexa_poly_eqn, octa_poly_eqn, power_range)
    
# Thrust vs Power for Hexacopter
def thrust_power(main_df, ref_df):
    main_thrust = main_df['Thrust (lbf)'] * num_props
    main_electrical_power = main_df['Electrical power (W)'] * num_props
    
    ref_thrust = ref_df['Thrust (lbf)'] * num_props
    ref_electrical_power = ref_df['Electrical power (W)'] * num_props
    
    plt.figure(figsize=(10, 6))
    
    # Plot main data with polynomial fit
    main_coeffs = np.polyfit(main_thrust, main_electrical_power, 3)
    main_poly_eqn = np.poly1d(main_coeffs)
    main_poly_fit = main_poly_eqn(np.linspace(main_thrust.min(), main_thrust.max(), 500))
    plt.plot(main_poly_fit, np.linspace(main_thrust.min(), main_thrust.max(), 500), color='green', label='Experimental')
    
    # Plot reference data with cubic spline
    ref_spline = CubicSpline(ref_electrical_power, ref_thrust)
    ref_spline_fit = ref_spline(np.linspace(ref_electrical_power.min(), ref_electrical_power.max(), 500))
    plt.plot(np.linspace(ref_electrical_power.min(), ref_electrical_power.max(), 500), ref_spline_fit, color='blue', label='Reference')

    # Adding scatter points
    scatter_thrust = np.arange(14, 32, 2)  # Scatter points for thrust (y-values)
    scatter_electrical_power = main_poly_eqn(scatter_thrust)  # Interpolated x-values using the polynomial fit
    
    plt.scatter(scatter_electrical_power, scatter_thrust, color='purple', zorder=5)
    for x, y in zip(scatter_electrical_power, scatter_thrust):
        plt.text(x, y + 1, f'({x:.2f}, {y})', fontsize=9, ha='right')

    # Adding a linear trendline
    linear_coeffs = np.polyfit(scatter_electrical_power, scatter_thrust, 1)
    slope = linear_coeffs[0]
    print(f"Watts per pound: {1/slope:.6f}")

    # Plot the maximum engine power line
    plt.axvline(x=1654*0.5, color='blue', linestyle='--', label=r'Engine Power at 50% Efficiency')
    plt.axvline(x=1654*0.75, color='green', linestyle='--', label=r'Engine Power at 75% Efficiency')
    plt.axvline(x=1654, color='red', linestyle='--', label='Maximum Engine Power')

    # Adding the shaded horizontal region
    plt.axhspan(21, 26, color='yellow', alpha=0.3, label='Weight Estimate Region')
    
    plt.xlim(left=0, right=2500)
    plt.ylim(bottom=0, top=40)
    plt.xlabel('Electrical Power (W)')
    plt.ylabel('Thrust (lbf)')
    plt.title('Thrust vs Electrical Power for Hexacopter')
    plt.legend()
    plt.grid(True)
    plt.show()

def thrust_RPM(main_df, ref_df):
    main_thrust = main_df['Thrust (lbf)'] 
    main_RPM = main_df['Motor Optical Speed (RPM)'] 

    ref_thrust = ref_df['Thrust (lbf)']
    ref_RPM = ref_df['Rotation speed (rpm)'] 
    
    plt.figure(figsize=(10, 6))
    
    # Plot main data with polynomial fit
    main_coeffs = np.polyfit(main_RPM, main_thrust, 3)
    main_poly_eqn = np.poly1d(main_coeffs)
    main_poly_fit = np.linspace(main_RPM.min(), main_RPM.max(), 500)
    plt.plot(main_poly_fit, main_poly_eqn(main_poly_fit), color='green', label='Experimental')
    
    # Plot reference data with cubic spline
    ref_spline = CubicSpline(ref_RPM, ref_thrust)
    ref_spline_fit = np.linspace(ref_RPM.min(), ref_RPM.max(), 500)
    plt.plot(ref_spline_fit, ref_spline(ref_spline_fit), color='blue', label='Reference')

    # Adding scatter points
    scatter_thrust = np.arange(2, 6, 0.5)  # Scatter points for thrust (y-values)
    scatter_RPM = np.interp(scatter_thrust, main_thrust, main_RPM)  # Interpolated x-values using main data
    
    plt.scatter(scatter_RPM, scatter_thrust, color='purple', zorder=5)
    for x, y in zip(scatter_RPM, scatter_thrust):
        plt.text(x, y + 0.1, f'({x:.2f}, {num_props * y})', fontsize=9, ha='right')

    # Adding the shaded horizontal region
    plt.axhspan(21/6, 26/6, color='yellow', alpha=0.3, label='Weight Estimate Region')

    plt.xlim(left=0, right=6100)
    plt.ylim(bottom=0, top=6)
    plt.xlabel('RPM')
    plt.ylabel('Thrust (lbf)')
    plt.title('Thrust vs RPM')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()

def prop_efficiency_RPM(df):
    prop_efficiency = df['Propeller Mech. Efficiency (lbf/W)'] 
    RPM = df['Motor Optical Speed (RPM)']
    thrust = df['Thrust (lbf)']
    
    plt.figure(figsize=(10, 6))
    
    # Plot data with polynomial fit
    coeffs = np.polyfit(RPM, prop_efficiency, 3)
    poly_eqn = np.poly1d(coeffs)
    poly_fit = np.linspace(RPM.min(), RPM.max(), 500)
    plt.plot(poly_fit, poly_eqn(poly_fit), color='green')
    
    # Calculate the RPM range for the given thrust values
    min_thrust = 21 / 6
    max_thrust = 26 / 6
    operational_rpm_range = df[(thrust >= min_thrust) & (thrust <= max_thrust)]['Motor Optical Speed (RPM)']
    
    if not operational_rpm_range.empty:
        min_rpm = operational_rpm_range.min()
        max_rpm = operational_rpm_range.max()
        
        # Shade the vertical region
        plt.axvspan(min_rpm, max_rpm, color='yellow', alpha=0.3, label='Estimated Operational RPM Region')
    
    plt.xlim(left=0, right=6100)
    plt.ylim(bottom=0)
    plt.xlabel('RPM')
    plt.ylabel('Propeller Efficiency (lbf/W)')
    plt.title('Propeller Efficiency vs RPM')
    plt.legend()
    plt.grid(True)
    plt.show()


def thrust_current(main_df, ref_df):
    main_thrust = main_df['Thrust (lbf)'] 
    main_current = main_df['Current (A)'] 

    ref_thrust = ref_df['Thrust (lbf)']
    ref_current = ref_df['Current (A)'] 
    
    plt.figure(figsize=(10, 6))
    
    # Plot main data with polynomial fit
    main_coeffs = np.polyfit(main_current, main_thrust, 3)
    main_poly_eqn = np.poly1d(main_coeffs)
    main_poly_fit = np.linspace(main_current.min(), main_current.max(), 500)
    plt.plot(main_poly_fit, main_poly_eqn(main_poly_fit), color='green', label='Experimental (22.2 V)')
    
    # Plot reference data with cubic spline
    ref_spline = CubicSpline(ref_current, ref_thrust)
    ref_spline_fit = np.linspace(ref_current.min(), ref_current.max(), 500)
    plt.plot(ref_spline_fit, ref_spline(ref_spline_fit), color='blue', label='Reference (24 V)')

    # Adding scatter points
    scatter_thrust = np.arange(2, 6, 0.5)  # Scatter points for thrust (y-values)
    scatter_current = np.interp(scatter_thrust, main_thrust, main_current)  # Interpolated x-values using main data
    
    plt.scatter(scatter_current, scatter_thrust, color='purple', zorder=5)
    for x, y in zip(scatter_current, scatter_thrust):
        plt.text(x + 0.1, y - 0.1, f'({x:.2f}, {num_props * y})', fontsize=9, ha='left')

    # Adding the shaded horizontal region
    plt.axhspan(21/6, 26/6, color='yellow', alpha=0.3, label='Weight Estimate Region')

    plt.xlim(left=0, right=24)
    plt.ylim(bottom=0, top=6.5)
    plt.xlabel('Current (A)')
    plt.ylabel('Thrust (lbf)')
    plt.title('Thrust vs Current')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()

# ToF vs Weight (blank as of now)
def tof_weight():
    # Create a blank plot with the specified labels and title
    plt.figure(figsize=(10, 6))

    # Set the axis labels
    plt.xlabel('Weight (lbs)')
    plt.ylabel('Time of Flight (hh:mm:ss)')

    # Set the title
    plt.title('Time of Flight vs Weight for Hexacopter')

    plt.xlim(left=0, right=30)
    plt.ylim(bottom=0, top=3)

    # Format the y-axis labels as hh:mm:ss
    formatter = ticker.FuncFormatter(hours_to_hms)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.grid(True)
    plt.show()

def plot_difference_prop(poly_16_eqn, poly_18_eqn, poly_20_eqn, power_range):
    thrust_16 = poly_16_eqn(power_range)
    thrust_18 = poly_18_eqn(power_range)
    thrust_20 = poly_20_eqn(power_range)

    plt.figure(figsize=(10, 6))
    plt.plot(power_range, thrust_20 - thrust_16, label=r'P20 $-$ P16')
    plt.plot(power_range, thrust_20 - thrust_18, label=r'P20 $-$ P18')
    plt.plot(power_range, thrust_18 - thrust_16, label=r'P18 $-$ P16')

    # Plot the maximum engine power line
    plt.axvline(x=1654 * 0.5, color='blue', linestyle='--', label=r'Engine Power at 50% Efficiency')
    plt.axvline(x=1654 * 0.75, color='green', linestyle='--', label=r'Engine Power at 75% Efficiency')
    plt.axvline(x=1654, color='red', linestyle='--', label='Maximum Engine Power')

    plt.xlim(left=0, right=2000)
    plt.ylim(bottom=0, top=3)
    plt.xlabel('Electrical Power (W)')
    plt.ylabel('Thrust Difference (lbf)')
    plt.title('Thrust Difference vs Electrical Power for Different Motor-Prop Pairs in Hexacopter Configuration')
    plt.legend()
    plt.grid(True)
    plt.show()

def prop_comparison(df_16, df_18, df_20):
    # Hexacopter thrust and power calculations
    hexa_thrust_16 = df_16['Thrust (lbf)'] * 6
    hexa_power_16 = df_16['Electrical power (W)'] * 6

    hexa_thrust_18 = df_18['Thrust (lbf)'] * 6
    hexa_power_18 = df_18['Electrical power (W)'] * 6

    hexa_thrust_20 = df_20['Thrust (lbf)'] * 6
    hexa_power_20 = df_20['Electrical power (W)'] * 6

    plt.figure(figsize=(10, 6))

    # Plot for 16in props
    hexa_coeffs_16 = np.polyfit(hexa_power_16, hexa_thrust_16, 3)
    hexa_poly_eqn_16 = np.poly1d(hexa_coeffs_16)
    power_range_16 = np.linspace(hexa_power_16.min(), hexa_power_16.max(), 500)
    plt.plot(power_range_16, hexa_poly_eqn_16(power_range_16), color='orange', label='P16x5.4 with kV490 (current)')

    # Plot for 18in props
    hexa_coeffs_18 = np.polyfit(hexa_power_18, hexa_thrust_18, 3)
    hexa_poly_eqn_18 = np.poly1d(hexa_coeffs_18)
    power_range_18 = np.linspace(hexa_power_18.min(), 2050, 500)
    plt.plot(power_range_18, hexa_poly_eqn_18(power_range_18), color='green', label='P18x6.1 with kV230')

    # Plot for 20in props
    hexa_coeffs_20 = np.polyfit(hexa_power_20, hexa_thrust_20, 3)
    hexa_poly_eqn_20 = np.poly1d(hexa_coeffs_20)
    power_range_20 = np.linspace(hexa_power_20.min(), hexa_power_20.max(), 500)
    plt.plot(power_range_20, hexa_poly_eqn_20(power_range_20), color='blue', label='P20x6 with kV230')

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
    plt.title('Thrust vs Electrical Power for Different Motor-Prop Pairs in Hexacopter Configuration')
    plt.legend()
    plt.grid(True)
    plt.show()

    plot_difference_prop(hexa_poly_eqn_16, hexa_poly_eqn_18, hexa_poly_eqn_20, power_range_18)


# Main function to execute the script
def main(main_file_name, ref_file_name, P18_file_name, P20_file_name):
    main_df = read_and_filter_data(main_file_name)
    ref_df = read_reference_data(ref_file_name)
    df_18 = read_reference_data(P18_file_name)
    df_20 = read_reference_data(P20_file_name)

    #uav_config(main_df)
    #thrust_power(main_df, ref_df)
    #thrust_RPM(main_df, ref_df)
    prop_efficiency_RPM(main_df)
    # thrust_current(main_df, ref_df)
    #tof_weight()
    #prop_comparison(main_df, df_18, df_20)

if __name__ == '__main__':
    main_file_name = 'prop_data/test2_arm4_CCW_noknob.csv' # User input
    ref_file_name = 'prop_data/T-motor_16x54_static_KV490.csv' # User input
    P18_file_name = 'P18x61/kV230.csv' # User input
    P20_file_name = 'P20x6/kV230.csv' # User input
    main(main_file_name, ref_file_name, P18_file_name, P20_file_name)