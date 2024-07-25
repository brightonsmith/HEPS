import serial
import time

def establish_serial_connection(port, baudrate):
    while True:
        try:
            ser = serial.Serial(port, baudrate)
            return ser
        except serial.SerialException:
            print(f"Failed to connect to {port}. Retrying in 2 seconds...")
            time.sleep(2)

def main():
    # Specify your serial port and baud rate
    port = 'COM3'  # Replace with your port (e.g., 'COM3' on Windows or '/dev/ttyUSB0' on Linux)
    baudrate = 115200  # Replace with your baudrate

    # Establish serial connection
    ser = establish_serial_connection(port, baudrate)
    print(f"Connected to {port} at {baudrate} baud.")

    # Read and display serial output
    try:
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                print(line)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        ser.close()

if __name__ == "__main__":
    main()