import serial
import time
import sys
import os

def find_serial_port():
    """
    Try to automatically detect the Arduino serial port.
    """
    possible_ports = ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyACM0', '/dev/ttyACM1']
    for port in possible_ports:
        if os.path.exists(port):
            return port
    return None

def main():
    port = find_serial_port()
    if not port:
        print("Error: No Arduino serial port found.")
        sys.exit(1)

    try:
        # Open serial connection
        ser = serial.Serial(port, baudrate=115200, timeout=1)
        time.sleep(2)  # Wait for Arduino to reset after connection
        print(f"Connected to Arduino on {port}")

        while True:
            if ser.in_waiting > 0:  # Check if data is available
                line = ser.readline().decode('utf-8', errors='replace').strip()
                if line:
                    print(f"Received: {line}")

    except serial.SerialException as e:
        print(f"Serial error: {e}")
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    main()
