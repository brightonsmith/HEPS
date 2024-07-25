"""
For plotting sensor data
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from serial2csv import get_iteration, get_available_test_types

def check_existence(arg):
    csv_files = os.listdir('./sensor_data')
    for filename in csv_files:
        if arg in filename:
            return True
    return False

def get_available_sensor_types(test_type, iteration):
    sensor_types = []
    csv_files = os.listdir('./sensor_data')
    for filename in csv_files:
        parts = filename.split('_')
        if len(parts) == 3 and parts[1] == test_type and parts[2].startswith(f"{iteration}.csv"):
            sensor_type = parts[0]
            sensor_types.append(sensor_type)
    return sensor_types

def get_sensor_type(available_sensor_types):
    for i, sensor_type in enumerate(available_sensor_types, start=1):
        print(f"{i}: {sensor_type}")
    choice = get_valid_input("Enter a number: ", range(1, len(available_sensor_types) + 1))

    if choice == '':
        return choice
    else:
        return available_sensor_types[choice - 1]

def get_valid_input(prompt, valid_options):
    while True:
        try:
            user_input = input(prompt)
            if user_input == '':
                return user_input
            choice = int(user_input)
            if choice in valid_options:
                return choice
            else:
                print(f"Invalid choice. Please enter a number between {valid_options[0]} and {valid_options[-1]}.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

def get_sensor_variable(x_filepath):
    # Read headers from the CSV file
    try:
        df = pd.read_csv(x_filepath)
        headers = list(df.columns)
    except FileNotFoundError:
        print(f"File not found: {x_filepath}")
        return None
    except pd.errors.EmptyDataError:
        print(f"Empty file or no headers found: {x_filepath}")
        return None
    
    # Display headers for selection
    print(f"Select a variable from {x_filepath}:")
    for i, header in enumerate(headers, start=1):
        print(f"{i}: {header}")
    
    # Get user choice
    choice = get_valid_input("Enter a number: ", range(1, len(headers) + 1))
    return headers[choice - 1]

def filter_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def main():
    while True:
        available_test_types = get_available_test_types()
        print("Available tests:", ','.join(available_test_types))
        test_type = input("Enter test type: ")
        
        if not check_existence(test_type):
            print(f"No CSV files found for test type '{test_type}' in ./sensor_data directory.")
            continue

        while True:
            iteration = get_iteration(test_type, available_test_types)
            if not check_existence(test_type + '_' + str(iteration)):
                    print(f"No CSV files found for test type '{test_type}' with iteration {iteration} in ./sensor_data directory.")
                    continue
            break 
        break
            
    while True:
        available_sensor_types = get_available_sensor_types(test_type, iteration)
        print("Available sensors:", ', '.join(available_sensor_types))

        # x-data
        print("Select an x-variable sensor type:")
        x_sensor = get_sensor_type(available_sensor_types)
        x_filepath = f"./sensor_data/{x_sensor}_{test_type}_{iteration}.csv"

        if not os.path.exists(x_filepath):
            print(f"File not found: {x_filepath}")
            continue

        x_variable = get_sensor_variable(x_filepath)

        if x_variable is None:
            continue
        
        # y-data
        while True:
            try:
                num_y_sensors = int(input("Enter number of y-variables: "))
                break
            except ValueError:
                print("Invalid input. Please enter an integer.")

        y_sensors = []

        for i in range(num_y_sensors):
            if i == 0:
                print("Select a y-variable sensor type or press Enter to use same sensor:")
            else:
                print("Select a y-variable sensor type:")
            y_sensor = get_sensor_type(available_sensor_types)

            if y_sensor == '':
                y_sensors = [x_sensor] * num_y_sensors
                break
            else:
                y_sensors.append(y_sensor)

        y_filepaths = []
        y_variables = []

        for y_sensor in y_sensors:
            y_filepath = f"./sensor_data/{y_sensor}_{test_type}_{iteration}.csv"

            if not os.path.exists(y_filepath):
                print(f"File not found: {y_filepath}")
                continue

            y_variable = get_sensor_variable(y_filepath)

            if y_variable is None:
                continue

            y_filepaths.append(y_filepath)
            y_variables.append(y_variable)

        if not y_filepaths:
            continue

        # Read data for x-variable
        x_data = pd.read_csv(x_filepath)

        # Extract the clean variable names and units
        x_variable_clean = x_variable.split('(')[0].strip()
        x_unit = '(' + x_variable.split('(')[1].strip()

        # Plot data
        plt.figure(figsize=(10, 6))  # Adjust figure size as needed

        for y_filepath, y_variable in zip(y_filepaths, y_variables):
            # Read data for y-variable
            y_data = pd.read_csv(y_filepath)

            # Filter outliers in y-data
            y_data_filtered = filter_outliers(y_data, y_variable)
            x_data_filtered = x_data.loc[y_data_filtered.index]

            # Extract the clean variable names and units
            y_variable_clean = y_variable.split('(')[0].strip()
            y_unit = '(' + y_variable.split('(')[1].strip()

            # Plot scatter
            plt.scatter(x_data_filtered[x_variable], y_data_filtered[y_variable], label=f"{y_variable_clean} {y_unit}")

        # Add labels and title
        plt.xlabel(f"{x_variable_clean.capitalize()} {x_unit}")
        plt.ylabel(f"{y_variable_clean.capitalize()} {y_unit}")
        plt.title(f"{', '.join([y.split('(')[0].strip().capitalize() for y in y_variables])} vs {x_variable_clean.capitalize()} for {test_type} test, iteration {iteration}")

        # Add legend
        plt.legend()

        # Save plot
        plot_path = f"./figures/{x_variable_clean}_{'_'.join([y.split('(')[0].strip() for y in y_variables])}_{test_type}_{iteration}.png"
        plt.savefig(plot_path)
        plt.close()


        print(f"Saved plot to {plot_path}")

        # Ask if the user wants to continue
        continue_choice = input("Do you want to create another plot? (y/n): ").strip().lower()
        if continue_choice != 'y':
            break

if __name__ == "__main__":
    main()
