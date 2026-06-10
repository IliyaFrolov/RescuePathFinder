"""
assessor.py — Route accessibility assessment.

Combines IMU and camera data to assess whether a route is passable.

Two-layer model:
  1. Hard gates  — binary pass/fail, any failure = INACCESSIBLE
  2. Soft scoring — weighted 0-1 score across soft criteria → CLEAR / CAUTION / MARGINAL

To extend with new data sources (e.g. lidar, GPS, air quality):
  - Add hard thresholds to HardThresholds
  - Add soft thresholds + weight to SoftCriteria
  - Feed new values into assess()
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


# ── Thresholds — tune these to your environment ───────────────────────────────

@dataclass
class HardThresholds:
    """Any of these failing makes the route immediately INACCESSIBLE."""
    max_temp_c:          float = 60.0   # °C
    max_humidity_pct:    float = 90.0   # %
    max_accel_tilt:      float = 7.0    # m/s² on x or y axis (severe tilt)
    max_gyro_rate:       float = 100.0  # °/s on any axis (severe spin)
    max_blocking_objects: int  = 3      # objects detected blocking path
    min_corridor_px:     float = 30.0   # px clearance from frame edge


@dataclass
class SoftCriteria:
    """
    Each entry is (weight, ideal_value, warn_threshold).
    Scores are normalised to 0-1 where 1 = ideal, 0 = at/beyond warn threshold.
    Weights must sum to 1.0.
    """
    temp_weight:      float = 0.20
    temp_warn:        float = 40.0    # °C — above this, score degrades

    humidity_weight:  float = 0.15
    humidity_warn:    float = 70.0    # % — above this, score degrades

    stability_weight: float = 0.30   # highest weight — unstable drone = risky
    accel_warn:       float = 3.0    # m/s² tilt magnitude before score degrades
    gyro_warn:        float = 50.0   # °/s rotation before score degrades

    clearance_weight: float = 0.20
    clearance_warn:   float = 100.0  # px — below this, score degrades

    obstacle_weight:  float = 0.15
    obstacle_warn:    int   = 1      # any object detected starts degrading score


HARD = HardThresholds()
SOFT = SoftCriteria()

# ─────────────────────────────────────────────────────────────────────────────


class RouteStatus(Enum):
    INACCESSIBLE = "INACCESSIBLE"   # hard gate failed
    MARGINAL     = "MARGINAL"       # score 0.0 – 0.49
    CAUTION      = "CAUTION"        # score 0.5 – 0.79
    CLEAR        = "CLEAR"          # score 0.8 – 1.0
    UNKNOWN      = "UNKNOWN"        # insufficient data


@dataclass
class AssessmentResult:
    timestamp:    str
    status:       RouteStatus
    score:        float | None        # None if INACCESSIBLE or UNKNOWN
    hard_failures: list[str]          # reasons for any hard gate failures
    soft_scores:  dict[str, float]    # per-criterion scores for transparency
    notes:        list[str]           # human-readable observations

    def __str__(self):
        lines = [
            f"[{self.timestamp}] Route status: {self.status.value}",
        ]
        if self.score is not None:
            lines.append(f"  Confidence score : {self.score:.2f}")
        if self.hard_failures:
            lines.append(f"  Hard failures    : {', '.join(self.hard_failures)}")
        if self.soft_scores:
            scores_str = "  |  ".join(f"{k}: {v:.2f}" for k, v in self.soft_scores.items())
            lines.append(f"  Soft scores      : {scores_str}")
        if self.notes:
            for note in self.notes:
                lines.append(f"  ⚠ {note}")
        return "\n".join(lines)


# ── Normalisation helpers ─────────────────────────────────────────────────────

def _score_below(value: float, ideal: float, warn: float) -> float:
    """Score for metrics where lower is better (e.g. temp, humidity, obstacles).
    Returns 1.0 at or below ideal, 0.0 at or beyond warn."""
    if value <= ideal:
        return 1.0
    if value >= warn:
        return 0.0
    return 1.0 - (value - ideal) / (warn - ideal)


def _score_above(value: float, ideal: float, warn: float) -> float:
    """Score for metrics where higher is better (e.g. corridor clearance).
    Returns 1.0 at or above ideal, 0.0 at or below warn."""
    if value >= ideal:
        return 1.0
    if value <= warn:
        return 0.0
    return (value - warn) / (ideal - warn)


# ── Main assessment function ──────────────────────────────────────────────────

def assess(imu_data, camera_data) -> AssessmentResult:
    """
    Assess route accessibility from latest IMU and camera data.

    Args:
        imu_data:    SensorData | None  (from imu/imu.py)
        camera_data: dict | None        (from camera/camera.py analyze_scene)

    Returns:
        AssessmentResult
    """
    timestamp     = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    hard_failures = []
    notes         = []

    # ── Guard: need at least one source ──────────────────────────────────────
    if imu_data is None and camera_data is None:
        return AssessmentResult(
            timestamp=timestamp,
            status=RouteStatus.UNKNOWN,
            score=None,
            hard_failures=[],
            soft_scores={},
            notes=["No sensor data available yet"],
        )

    # ── Layer 1: Hard gates ───────────────────────────────────────────────────
    if imu_data is not None:
        if imu_data.temp_c > HARD.max_temp_c:
            hard_failures.append(f"temp {imu_data.temp_c:.1f}°C > {HARD.max_temp_c}°C")

        if imu_data.humidity > HARD.max_humidity_pct:
            hard_failures.append(f"humidity {imu_data.humidity:.1f}% > {HARD.max_humidity_pct}%")

        tilt = max(abs(imu_data.accel_x), abs(imu_data.accel_y))
        if tilt > HARD.max_accel_tilt:
            hard_failures.append(f"tilt {tilt:.1f} m/s² > {HARD.max_accel_tilt} m/s²")

        max_gyro = max(abs(imu_data.gyro_x), abs(imu_data.gyro_y), abs(imu_data.gyro_z))
        if max_gyro > HARD.max_gyro_rate:
            hard_failures.append(f"rotation {max_gyro:.1f}°/s > {HARD.max_gyro_rate}°/s")

    if camera_data is not None:
        if camera_data["object_count"] > HARD.max_blocking_objects:
            hard_failures.append(f"{camera_data['object_count']} objects blocking path")

        for obj in camera_data.get("objects", []):
            ox, oy = obj["offset_from_centre"]
            clearance = abs(ox)   # horizontal clearance from centre as proxy
            if clearance > 0 and clearance < HARD.min_corridor_px:
                hard_failures.append(f"corridor too narrow ({clearance:.0f}px clearance)")
                break

    if hard_failures:
        return AssessmentResult(
            timestamp=timestamp,
            status=RouteStatus.INACCESSIBLE,
            score=None,
            hard_failures=hard_failures,
            soft_scores={},
            notes=[f"Hard gate failed: {f}" for f in hard_failures],
        )

    # ── Layer 2: Soft scoring ─────────────────────────────────────────────────
    soft_scores   = {}
    weighted_sum  = 0.0
    total_weight  = 0.0

    if imu_data is not None:
        # Temperature
        t_score = _score_below(imu_data.temp_c, ideal=25.0, warn=SOFT.temp_warn)
        soft_scores["temp"]     = t_score
        weighted_sum           += t_score * SOFT.temp_weight
        total_weight           += SOFT.temp_weight
        if imu_data.temp_c > SOFT.temp_warn:
            notes.append(f"Elevated temperature: {imu_data.temp_c:.1f}°C")

        # Humidity
        h_score = _score_below(imu_data.humidity, ideal=50.0, warn=SOFT.humidity_warn)
        soft_scores["humidity"] = h_score
        weighted_sum           += h_score * SOFT.humidity_weight
        total_weight           += SOFT.humidity_weight
        if imu_data.humidity > SOFT.humidity_warn:
            notes.append(f"High humidity: {imu_data.humidity:.1f}%")

        # Stability (combined accel tilt + gyro)
        tilt      = max(abs(imu_data.accel_x), abs(imu_data.accel_y))
        max_gyro  = max(abs(imu_data.gyro_x), abs(imu_data.gyro_y), abs(imu_data.gyro_z))
        a_score   = _score_below(tilt,     ideal=1.0, warn=SOFT.accel_warn)
        g_score   = _score_below(max_gyro, ideal=5.0, warn=SOFT.gyro_warn)
        s_score   = (a_score + g_score) / 2
        soft_scores["stability"] = s_score
        weighted_sum            += s_score * SOFT.stability_weight
        total_weight            += SOFT.stability_weight
        if s_score < 0.6:
            notes.append("Drone showing instability — approach with caution")

    if camera_data is not None:
        # Corridor clearance — use smallest horizontal offset as proxy
        offsets = [abs(o["offset_from_centre"][0]) for o in camera_data.get("objects", [])]
        min_clearance = min(offsets) if offsets else 999.0
        c_score = _score_above(min_clearance, ideal=200.0, warn=SOFT.clearance_warn)
        soft_scores["clearance"] = c_score
        weighted_sum            += c_score * SOFT.clearance_weight
        total_weight            += SOFT.clearance_weight

        # Obstacle count
        count   = camera_data["object_count"]
        o_score = _score_below(float(count), ideal=0.0, warn=float(SOFT.obstacle_warn + 2))
        soft_scores["obstacles"] = o_score
        weighted_sum            += o_score * SOFT.obstacle_weight
        total_weight            += SOFT.obstacle_weight
        if count > 0:
            notes.append(f"{count} object(s) detected in path")

    # Normalise in case only one source was available
    final_score = weighted_sum / total_weight if total_weight > 0 else 0.0

    if final_score >= 0.8:
        status = RouteStatus.CLEAR
    elif final_score >= 0.5:
        status = RouteStatus.CAUTION
    else:
        status = RouteStatus.MARGINAL

    return AssessmentResult(
        timestamp=timestamp,
        status=status,
        score=round(final_score, 3),
        hard_failures=[],
        soft_scores={k: round(v, 2) for k, v in soft_scores.items()},
        notes=notes,
    )
