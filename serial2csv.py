"""
For converting native SPIFFS data to csv
"""

import serial
import time
import os

def establish_serial_connection(port, baudrate):
    while True:
        try:
            ser = serial.Serial(port, baudrate)
            return ser
        except serial.SerialException:
            print(f"Failed to connect to {port}. Retrying in 2 seconds...")
            time.sleep(2)

def get_available_test_types():
    test_types = []
    csv_files = os.listdir('./sensor_data')
    for filename in csv_files:
        parts = filename.split('_')
        if len(parts) == 3 and parts[1] not in test_types:
            test_types.append(parts[1])

    return test_types

def get_test_type(available_test_types):
    for i, test_type in enumerate(available_test_types, start=1):
        print(f"{i}: {test_type}")
    choice = input("Enter a number for an existing test or create a new one: ")
    try:
        choice = int(choice)
        return available_test_types[choice - 1]
    except ValueError:
        return choice
    
def get_iteration(test_type, available_test_types):
    iterations = []
    csv_files = os.listdir('./sensor_data')

    if test_type in available_test_types:
        for filename in csv_files:
            parts = filename.split('_')
            if len(parts) == 3 and parts[1] == test_type:
                iterations.append(int(parts[2].split('.')[0]))
        max_iteration = max(iterations)
    else:
        max_iteration = -1

    while True:
        try:
            iteration = int(input(f"Enter iteration number (current highest is {max_iteration}): "))
            if iteration >= 0 and iteration <= max_iteration + 1:
                break
            else:
                print(f"Please enter a non-negative integer less than or equal to {max_iteration + 1}.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    return iteration

def get_sensor_type(sensor_code):
    sensor_mapping = {
        'I': 'IMU',
        'B': 'BME',
        'V': 'INA',
        'H': 'hall',
        'T': 'therm'
    }
    return sensor_mapping.get(sensor_code, 'Unknown')

def read_csv(ser, sensor_code, test_type, iteration):
    sensor_type = get_sensor_type(sensor_code)
    if sensor_type == 'Unknown':
        print(f"Invalid sensor code: {sensor_code}")
        return
    
    filename = f'sensor_data/{sensor_type}_{test_type}_{str(iteration)}.csv'

    time.sleep(0.5)
    ser.write(sensor_code.encode())

    # Open a file to save the data
    with open(filename, 'w') as file:
        while True:
            try:
                line = ser.readline().decode('utf-8').strip()
                if line == "EOF":
                    break
                file.write(line + '\n')
                print(line)
            except UnicodeDecodeError as e:
                print(f"Decode error: {e} - data may be corrupted or in a different encoding.")

    print(f"\n{sensor_type} CSV saved to {filename}")
    print("----------------------------------------------------")

def main():
    # Set up the serial connection
    ser = establish_serial_connection('COM4', 115200)  # Adjust 'COM3' to your port
    ser.write(b'r') 

    available_test_types = get_available_test_types()
    print("Available tests:", ','.join(available_test_types))

    test_type = get_test_type(available_test_types)
    iteration = get_iteration(test_type, available_test_types)

    # Read CSVs for each sensor type
    # read_csv(ser, 'I', test_type, iteration)   # IMU
    # read_csv(ser, 'B', test_type, iteration)   # BME
    read_csv(ser, 'V', test_type, iteration)   # INA
    #read_csv(ser, 'H', test_type, iteration)  # Hall
    read_csv(ser, 'T', test_type, iteration)  # Therm

    ser.close()

if __name__ == '__main__':
    main()
