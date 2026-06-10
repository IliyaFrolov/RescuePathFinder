import serial
import threading
from dataclasses import dataclass
from datetime import datetime


@dataclass
class SensorData:
    timestamp:  str
    accel_x:    float
    accel_y:    float
    accel_z:    float
    imu_temp:   float
    gyro_x:     float
    gyro_y:     float
    gyro_z:     float
    humidity:   float
    temp_c:     float

    def __str__(self):
        return (
            f"[{self.timestamp}]\n"
            f"  Accel  (x/y/z) : {self.accel_x:>8.3f}  {self.accel_y:>8.3f}  {self.accel_z:>8.3f}  m/s²\n"
            f"  Gyro   (x/y/z) : {self.gyro_x:>8.3f}  {self.gyro_y:>8.3f}  {self.gyro_z:>8.3f}  °/s\n"
            f"  IMU Temp       : {self.imu_temp:>8.2f} °C\n"
            f"  Humidity       : {self.humidity:>8.2f} %\n"
            f"  Temperature    : {self.temp_c:>8.2f} °C"
        )

    def orientation_hint(self) -> str:
        """Basic tilt detection from accelerometer."""
        hints = []
        if self.accel_x > 5:
            hints.append("tilting FORWARD")
        elif self.accel_x < -5:
            hints.append("tilting BACKWARD")
        if self.accel_y > 5:
            hints.append("tilting RIGHT")
        elif self.accel_y < -5:
            hints.append("tilting LEFT")
        if abs(self.accel_z - 9.81) > 2:
            hints.append("unstable vertical")
        return ", ".join(hints) if hints else "level"

    def spin_hint(self) -> str:
        """Flags fast rotation on any axis."""
        threshold = 50  # °/s — tune to your drone
        axes = []
        if abs(self.gyro_x) > threshold: axes.append(f"roll {self.gyro_x:+.1f}°/s")
        if abs(self.gyro_y) > threshold: axes.append(f"pitch {self.gyro_y:+.1f}°/s")
        if abs(self.gyro_z) > threshold: axes.append(f"yaw {self.gyro_z:+.1f}°/s")
        return ", ".join(axes) if axes else "stable"


def parse_line(line: str) -> SensorData | None:
    """Parse a CSV line from the Arduino into a SensorData object."""
    try:
        parts = line.strip().split(",")
        
        return SensorData(
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3],
            accel_x   = float(parts[0]),
            accel_y   = float(parts[1]),
            accel_z   = float(parts[2]),
            imu_temp  = float(parts[3]),
            gyro_x    = float(parts[4]),
            gyro_y    = float(parts[5]),
            gyro_z    = float(parts[6]),
            humidity  = float(parts[7]),
            temp_c    = float(parts[8]),
        )
    except ValueError:
        return None  # header row or garbled data


def print_sensor_data(data: SensorData):
    print(data)
    print(f"  Orientation    : {data.orientation_hint()}")
    print(f"  Rotation       : {data.spin_hint()}")
    print()

