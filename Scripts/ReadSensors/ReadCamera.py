import cv2
import requests
import numpy as np
from datetime import datetime
from dataclasses import dataclass

# ── Config ────────────────────────────────────────────────────────────────────
DRONE_IP   = "192.168.1.113"
PORT       = 5000
STREAM_URL = f"http://{DRONE_IP}:{PORT}/video_feed"
AUTH       = None
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Detection:
    x: int          # top-left x
    y: int          # top-left y
    width: int
    height: int
    center_x: float
    center_y: float
    area: int
    label: str = "object"  # extend if you can read text labels


def extract_bounding_boxes(frame: np.ndarray) -> list[Detection]:
    """
    Finds bounding boxes drawn on the frame by the on-Pi model.
    Looks for rectangular contours in unusual/highlight colours
    (green, red, cyan etc.) that object detectors typically use.
    """
    detections = []

    # ── Tune these HSV ranges to match your model's box colour ───────────────
    # Common detector colours:
    BOX_COLOURS = {
        "green":  ((40,  70,  70), (80,  255, 255)),
        "red_lo": ((0,   120, 120), (10,  255, 255)),   # red wraps in HSV
        "red_hi": ((170, 120, 120), (180, 255, 255)),
        "cyan":   ((85,  100, 100), (100, 255, 255)),
        "yellow": ((20,  100, 100), (35,  255, 255)),
    }
    # ─────────────────────────────────────────────────────────────────────────

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    for colour, (lower, upper) in BOX_COLOURS.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # Clean up noise
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(
        combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        # Filter out tiny blobs (noise) and huge ones (full frame border)
        area = cv2.contourArea(contour)
        frame_area = frame.shape[0] * frame.shape[1]
        if area < 500 or area > frame_area * 0.9:
            continue

        # Only keep roughly rectangular shapes
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        if len(approx) < 4:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        detections.append(Detection(
            x=x, y=y,
            width=w, height=h,
            center_x=round(x + w / 2, 1),
            center_y=round(y + h / 2, 1),
            area=int(area),
        ))

    return detections


def try_read_labels(frame: np.ndarray, detections: list[Detection]) -> list[Detection]:
    """
    Attempts to OCR text labels near each bounding box using Tesseract.
    Optional — requires:  pip install pytesseract && sudo apt install tesseract-ocr
    """
    try:
        import pytesseract
        for det in detections:
            # Crop the region just above the box where labels are usually drawn
            label_y1 = max(0, det.y - 30)
            label_y2 = det.y
            label_x1 = det.x
            label_x2 = det.x + det.width
            roi = frame[label_y1:label_y2, label_x1:label_x2]
            if roi.size == 0:
                continue
            text = pytesseract.image_to_string(
                roi, config="--psm 7 --oem 3"  # single line mode
            ).strip()
            if text:
                det.label = text
    except ImportError:
        pass  # Tesseract not installed — labels stay as "object"
    return detections


def analyze_scene(detections: list[Detection], frame_shape: tuple) -> dict:
    """
    Derives useful drone navigation info from the detections.
    """
    h, w = frame_shape[:2]
    frame_center_x = w / 2
    frame_center_y = h / 2

    analysis = {
        "object_count": len(detections),
        "objects": [],
        "path_clear": len(detections) == 0,
    }

    for det in detections:
        # Where is the object relative to the drone's view?
        offset_x = det.center_x - frame_center_x
        offset_y = det.center_y - frame_center_y
        h_pos = "left" if offset_x < -w * 0.15 else ("right" if offset_x > w * 0.15 else "centre")
        v_pos = "above" if offset_y < -h * 0.15 else ("below" if offset_y > h * 0.15 else "centre")

        analysis["objects"].append({
            "label":    det.label,
            "position": f"{v_pos}-{h_pos}",
            "box":      (det.x, det.y, det.width, det.height),
            "offset_from_centre": (round(offset_x), round(offset_y)),
        })

    return analysis


def process_frame(frame: np.ndarray, frame_number: int):
    detections = extract_bounding_boxes(frame)
    detections = try_read_labels(frame, detections)
    analysis   = analyze_scene(detections, frame.shape)
    timestamp  = datetime.now().strftime("%H:%M:%S.%f")[:-3]

    print(f"\n[{timestamp}] Frame #{frame_number:05d}")
    if analysis["path_clear"]:
        print("  ✓ Path clear — no objects detected")
    else:
        print(f"  ⚠ {analysis['object_count']} object(s) detected:")
        for obj in analysis["objects"]:
            print(f"    · [{obj['label']}] at {obj['position']} | "
                  f"box={obj['box']} | "
                  f"offset from centre={obj['offset_from_centre']}px")

    # Optional: show the frame with debug overlay
    # cv2.imshow("Drone Feed", frame)
    # cv2.waitKey(1)

    return analysis


def stream_mjpeg():
    print(f"Connecting to {STREAM_URL} ...")
    resp = requests.get(STREAM_URL, auth=AUTH, stream=True, timeout=10)
    resp.raise_for_status()
    print("Connected. Press Ctrl+C to stop.\n")

    frame_number = 0
    buffer = b""

    for chunk in resp.iter_content(chunk_size=4096):
        buffer += chunk
        start = buffer.find(b"\xff\xd8")
        end   = buffer.find(b"\xff\xd9")

        if start != -1 and end != -1 and end > start:
            jpg_data = buffer[start : end + 2]
            buffer   = buffer[end + 2:]

            img_array = np.frombuffer(jpg_data, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if frame is not None:
                process_frame(frame, frame_number)
                frame_number += 1


if __name__ == "__main__":
    try:
        stream_mjpeg()
    except KeyboardInterrupt:
        print("\nStopped.")