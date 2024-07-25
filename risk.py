import numpy as np
from prop_plotter import read_and_filter_data, read_reference_data, hours_to_hms

# User config
max_engine_output = 1640  # Maximum theoretical engine output in watts
efficiency_levels = np.arange(0.4, 1.01, 0.1)  # 40% to 100% in steps of 10%
discrete_efficiency_levels = np.arange(0.4, 1.01, 0.1)  # 40% to 100% in steps of 5%
weight_range = np.arange(21, 27)  # 21 to 26 lbs
voltage_battery = 22.2 # V
capcity_battery = 8000 # mAh
energy_battery = (voltage_battery * capcity_battery) / 1000 #Wh

def preprocess_and_sort(func):
    def wrapper(df, value):
        # Scale electrical power and thrust for hexacopter (6 props)
        hexa_power = 6 * df['Electrical power (W)']
        hexa_thrust = 6 * df['Thrust (lbf)']
        
        # Sort based on hexa_power to ensure interpolation works correctly
        sorted_indices = np.argsort(hexa_power)
        hexa_power_sorted = hexa_power.iloc[sorted_indices]
        hexa_thrust_sorted = hexa_thrust.iloc[sorted_indices]
        
        return func(hexa_power_sorted, hexa_thrust_sorted, value)
    
    return wrapper

@preprocess_and_sort
def interpolate_thrust(hexa_power_sorted, hexa_thrust_sorted, power):
    # Interpolate thrust based on the sorted scaled power
    interpolated_thrust = np.interp(power, hexa_power_sorted, hexa_thrust_sorted)
    return interpolated_thrust

@preprocess_and_sort
def interpolate_power(hexa_power_sorted, hexa_thrust_sorted, thrust):
    # Interpolate power based on the sorted scaled thrust
    interpolated_power = np.interp(thrust, hexa_thrust_sorted, hexa_power_sorted)
    return interpolated_power

def build_hover_likelihood_matrix(filename, read_csv_func):
    # Load motor-propeller data for the specific propeller size
    motor_prop_data = read_csv_func(filename)

    hover_likelihood_matrix = np.empty((len(weight_range), len(discrete_efficiency_levels)), dtype=object)

    # Iterate over each weight and efficiency level
    for i, weight in enumerate(weight_range):
        for j, efficiency in enumerate(discrete_efficiency_levels):
            # Calculate power based on efficiency
            power = efficiency * max_engine_output

            # Interpolate thrust for the calculated power
            interpolated_thrust = interpolate_thrust(motor_prop_data, power)

            # Calculate thrust relative to weight
            thrust_percent = (interpolated_thrust - weight) / weight * 100

            # Assign label based on thrust percentage
            if thrust_percent < -10:
                hover_likelihood_matrix[i, j] = 'Very Low'
            elif -10 <= thrust_percent < -5:
                hover_likelihood_matrix[i, j] = 'Low'
            elif -5 <= thrust_percent <= 5:
                hover_likelihood_matrix[i, j] = 'Medium'
            elif 5 < thrust_percent <= 10:
                hover_likelihood_matrix[i, j] = 'High'
            elif thrust_percent > 10:
                hover_likelihood_matrix[i, j] = 'Very High'

    return hover_likelihood_matrix

def build_ToF_matrix(filename, read_csv_func):
    motor_prop_data = read_csv_func(filename)

    ToF_matrix = np.empty((len(weight_range), len(discrete_efficiency_levels)), dtype=object)

    for i, weight in enumerate(weight_range):
        power_battery_min = None
        fill_value = None
        for j, efficiency in enumerate(efficiency_levels):
            power_req = interpolate_power(motor_prop_data, weight)
            power_engine = efficiency * max_engine_output
            thrust_engine = interpolate_thrust(motor_prop_data, power_engine)
            
            if fill_value is not None:
                if round(efficiency, 2) in np.round(discrete_efficiency_levels, 2):
                    discrete_index = np.where(np.round(discrete_efficiency_levels, 2) == round(efficiency, 2))[0][0]
                    ToF_matrix[i, discrete_index] = fill_value
                continue
            
            # Sustainable flight
            if thrust_engine >= weight:
                if power_battery_min is not None:
                    fill_value = f">{hours_to_hms(energy_battery / power_battery_min, None)}"
                else:
                    fill_value = f">{hours_to_hms(energy_battery / power_req, None)}"  # Placeholder for initial low bound if not set
                if round(efficiency, 2) in np.round(discrete_efficiency_levels, 2):
                    discrete_index = np.where(np.round(discrete_efficiency_levels, 2) == round(efficiency, 2))[0][0]
                    ToF_matrix[i, discrete_index] = fill_value
            # Drain battery
            else:
                power_battery = power_req - power_engine
                if round(efficiency, 2) in np.round(discrete_efficiency_levels, 2):
                    discrete_index = np.where(np.round(discrete_efficiency_levels, 2) == round(efficiency, 2))[0][0]
                    ToF_matrix[i, discrete_index] = hours_to_hms(energy_battery / power_battery, None)
                power_battery_min = power_battery if power_battery_min is None else min(power_battery_min, power_battery)

    return ToF_matrix

def build_latex_hover_likelihood_matrix(matrix):
    # Header row labels
    header_row = " & "
    for efficiency in discrete_efficiency_levels:
        header_row += f"{round(efficiency*100)}" + r"\% & "
    header_row = header_row[:-2] + r" \\" + "\n"
    
    latex_code = r"\begin{table}[h]" + "\n"
    latex_code += r"    \centering" + "\n"
    latex_code += r"    \begin{tabular}{|c|" + "c|" * len(matrix[0]) + "}" + "\n"
    latex_code += r"        \hline" + "\n"
    latex_code += header_row
    latex_code += r"        \hline" + "\n"
    
    # Loop through each row of the matrix
    for i, row in enumerate(matrix):
        latex_code += f"        {weight_range[i]} lbs & "
        for entry in row:
            if entry == 'Very High':
                latex_code += r"\cellcolor{ForestGreen!80} Very High & "
            elif entry == 'High':
                latex_code += r"\cellcolor{ForestGreen!40} High & "
            elif entry == 'Medium':
                latex_code += r"\cellcolor{yellow!60} Medium & "
            elif entry == 'Low':
                latex_code += r"\cellcolor{red!20} Low & "
            elif entry == 'Very Low':
                latex_code += r"\cellcolor{red!80} Very Low & "
        latex_code = latex_code[:-2] + r"\\" + "\n"
        latex_code += r"        \hline" + "\n"
    
    # Footer with captions and labels
    latex_code += r"    \end{tabular}" + "\n"
    latex_code += r"    \caption{Hover flight likelihood matrix for}" + "\n"
    latex_code += r"    \label{tab:hover_likelihood_matrix}" + "\n"
    latex_code += r"\end{table}" + "\n"
    
    return latex_code

def build_latex_ToF_matrix(matrix):
    # Header row labels
    header_row = " & "
    for efficiency in discrete_efficiency_levels:
        header_row += f"{round(efficiency*100)}" + r"\% & "
    header_row = header_row[:-2] + r" \\" + "\n"
    
    latex_code = r"\begin{table}[h]" + "\n"
    latex_code += r"    \centering" + "\n"
    latex_code += r"    \begin{tabular}{|c|" + "c|" * len(matrix[0]) + "}" + "\n"
    latex_code += r"        \hline" + "\n"
    latex_code += header_row
    latex_code += r"        \hline" + "\n"
    
    # Loop through each row of the matrix
    for i, row in enumerate(matrix):
        latex_code += f"        {weight_range[i]} lbs & "
        for entry in row:
            if '>' in entry:
                latex_code += r"\cellcolor{ForestGreen!80} " + entry +  " & "
            else:
                latex_code += r"\cellcolor{red!80} " + entry +  " & "
        latex_code = latex_code[:-2] + r"\\" + "\n"
        latex_code += r"        \hline" + "\n"
    
    # Footer with captions and labels
    latex_code += r"    \end{tabular}" + "\n"
    latex_code += r"    \caption{Hovering time of flight matrix for}" + "\n"
    latex_code += r"    \label{tab:hover_ToF_matrix}" + "\n"
    latex_code += r"\end{table}" + "\n"
    
    return latex_code

# 16 in props
# hover_likelihood_matrix_16in = build_hover_likelihood_matrix('prop_data/test2_arm4_CCW_noknob.csv', read_and_filter_data)
# latex_code = build_latex_hover_likelihood_matrix(hover_likelihood_matrix_16in)
# print(latex_code)

ToF_matrix_16in = build_ToF_matrix('prop_data/test2_arm4_CCW_noknob.csv', read_and_filter_data)
print(ToF_matrix_16in)
# latex_code = build_latex_ToF_matrix(ToF_matrix_16in)
# print(latex_code)

# # 18 in props
# # hover_likelihood_matrix_18in = build_hover_likelihood_matrix('P18x61/kV230.csv', read_reference_data)
# # latex_code = build_latex_hover_likelihood_matrix(hover_likelihood_matrix_18in)
# # print(latex_code)

# ToF_matrix_18in = build_ToF_matrix('P18x61/kV230.csv', read_reference_data)
# latex_code = build_latex_ToF_matrix(ToF_matrix_18in)
# print(latex_code)

# # 20 in props
# # hover_likelihood_matrix_20in = build_hover_likelihood_matrix('P20x6/kV230.csv', read_reference_data)
# # latex_code = build_latex_hover_likelihood_matrix(hover_likelihood_matrix_20in)
# # print(latex_code)

# ToF_matrix_20in = build_ToF_matrix('P20x6/kV230.csv', read_reference_data)
# latex_code = build_latex_ToF_matrix(ToF_matrix_20in)
# print(latex_code)

