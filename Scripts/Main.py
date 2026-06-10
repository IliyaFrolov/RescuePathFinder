import math
import os
from Map import Map

def load_waypoints(file_path):
    """
    Load an ArduPilot/Mission Planner .waypoints file and return a list of dicts.
    Handles validation and malformed lines safely.
    """
    # Validate file existence
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File does not exist: {file_path}")

    waypoints = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        raise RuntimeError(f"Failed to read file: {e}")

    # Mission Planner's first line usually: QGC WPL <version>
    if not lines:
        raise ValueError("File is empty.")

    header = lines[0].strip()
    if not header.startswith("QGC WPL"):
        raise ValueError("Invalid .waypoints file header.")

    for index, line in enumerate(lines[1:], start=2):
        line = line.strip()
        if not line:
            continue  # skip blank lines

        parts = line.split("\t")
        if len(parts) != 12:
            raise ValueError(
                f"Malformed waypoint at line {index}: expected 12 fields, got {len(parts)}"
            )

        try:
            # waypoint = {
            #     "id": int(parts[0]),
            #     "current": int(parts[1]),
            #     "frame": int(parts[2]),
            #     "command": int(parts[3]),
            #     "param1": float(parts[4]),
            #     "param2": float(parts[5]),
            #     "param3": float(parts[6]),
            #     "param4": float(parts[7]),
            #     "latitude": float(parts[8]),
            #     "longitude": float(parts[9]),
            #     "altitude": float(parts[10]),
            #     "autocontinue": int(parts[11])
            # }
            
            waypoint = {int(parts[0]): (float(parts[8]), float(parts[9]))}
        except ValueError as e:
            raise ValueError(f"Line {index} has invalid numeric value: {e}")

        waypoints.append(waypoint)

    return waypoints


# Example usage
if __name__ == "__main__":
    path = "C:\\Users\\Loan User\\Source\\RescuePathFinder\\Waypoints\\waypoints_test.waypoints"  # Put your filename here

    try:
        waypoints = load_waypoints(path)
        connections = [
            ("1", "2"),
            ("2", "3"),
            ("3", "4"),
            ("4", "5"),
            ("5", "6"),
            ("4", "7"),
            ("7", "6")
        ]
        print("Loaded waypoints:")
        
        for w in waypoints:
            print(w)
            
        drone_map = Map(waypoints=waypoints, connections=connections)
        print(drone_map.map.graph)
        
    except Exception as e:
        print(f"Error: {e}")
    

# 1. Your custom, offline list of latitude/longitude waypoints
# Format: { waypoint_id: (latitude, longitude) }
# waypoints = {
#     "Basecamp": (34.0522, -118.2437),
#     "Waypoint_1": (34.0530, -118.2500),
#     "Waypoint_2": (34.0600, -118.2400),
#     "Destination": (34.0650, -118.2550)
# }

# # 5. Define valid paths (edges) between your waypoints and add weights
# # (Only connect nodes that actually have a physical path or line-of-sight between them)
# valid_connections = [
#     ("Basecamp", "Waypoint_1"),
#     ("Basecamp", "Waypoint_2"),
#     ("Waypoint_1", "Destination"),
#     ("Waypoint_2", "Destination")
# ]

