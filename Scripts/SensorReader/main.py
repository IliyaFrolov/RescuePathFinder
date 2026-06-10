import threading
import time
import csv

from ReadIMU.imu import parse_line, print_sensor_data
from ReadCamera.camera import stream_mjpeg
from Assessor.assessor import assess

# ── Config ────────────────────────────────────────────────────────────────────
SERIAL_PORT      = "/dev/ttyACM0"   # or /dev/ttyACM0 — check with: ls /dev/tty*
BAUD_RATE        = 115200

DRONE_IP         = "192.168.1.113"
PORT             = 5000
STREAM_URL       = f"http://{DRONE_IP}:{PORT}/video_feed"
AUTH             = None

ASSESS_INTERVAL  = 2.0   # seconds between route assessments

IMU_OUTPUTS_FILE_PATH = "outputs/imu_outputs.csv"
# ─────────────────────────────────────────────────────────────────────────────

# Shared state — all threads write here, assessor reads from here
latest = {
    "imu":    None,   # most recent SensorData
    "camera": None,   # most recent scene analysis dict
}


def on_new_imu(data):
    """Called each time a new IMU reading arrives. Add extra IMU logic here."""
    print_sensor_data(data)


def start_imu_thread(stop_event: threading.Event) -> threading.Thread:
    def _run():
        import serial

        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            print(f"[IMU] Serial connected on {SERIAL_PORT} @ {BAUD_RATE} baud")
        except serial.SerialException as e:
            print(f"[IMU] Could not open port: {e}")
            return

        while not stop_event.is_set():
            try:
                raw  = ser.readline().decode("utf-8", errors="replace")
                data = parse_line(raw)
                if data:
                    latest["imu"] = data
                    on_new_imu(data)
            except serial.SerialException as e:
                print(f"[IMU] Read error: {e}")
                break

        ser.close()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def start_camera_thread(stop_event: threading.Event) -> threading.Thread:
    t = threading.Thread(
        target=stream_mjpeg,
        args=(STREAM_URL, AUTH, latest, stop_event),
        daemon=True,
    )
    t.start()
    return t


def start_assessor_thread(stop_event: threading.Event) -> threading.Thread:
    """Periodically assesses route accessibility from latest sensor data."""
    def _run():
        print(f"[Assessor] Running every {ASSESS_INTERVAL}s")
        while not stop_event.is_set():
            result = assess(latest["imu"], latest["camera"])
            print("\n" + "─" * 60)
            print(result)
            print("─" * 60)
            time.sleep(ASSESS_INTERVAL)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t

def write_imu_to_csv():
    imu_data = latest["imu"]
    
    # Writing to CSV
    with open(IMU_OUTPUTS_FILE_PATH, mode='w', newline='', encoding='utf-8') as file:
        # Define CSV field names based on keys of the first dict
        fieldnames = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write header row
        writer.writeheader()
        
        # Write data rows
        for row in data:
            writer.writerow(row)

if __name__ == "__main__":
    stop_event = threading.Event()

    print("=== Drone sensor system starting ===\n")

    imu_thread      = start_imu_thread(stop_event)
    camera_thread   = start_camera_thread(stop_event)
    assessor_thread = start_assessor_thread(stop_event)

    print("All systems running — press Ctrl+C to stop.\n")

    try:
        while True:
            # Main thread is free here — access latest whenever you need it
            imu    = latest["imu"]
            camera = latest["camera"]
            
            if imu and camera:
                print(f"[Main] Temp: {imu.temp_c:.1f}°C | Objects: {camera['object_count']}")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down...")
        stop_event.set()
