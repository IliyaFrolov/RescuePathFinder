import networkx as nx
import math

class Map:
    
    def __init__(self, waypoints, connections):
        self.map = nx.Graph()
        for waypoint in waypoints:
            print(waypoint)
            node_id, coords = waypoint.items()
            ""
            self.map.add_node(node_id, pos=coords)
        
        for start, end in connections:
            distance = self.haversine_distance(waypoints[start], waypoints[end])
            self.map.add_edge(start, end, weight=distance)
    
    def get_source_target_distance(self, source: str, target: str):
        pass
    
    # Haversine function to calculate earth-surface distance in kilometers
    def haversine_distance(self, coord1, coord2):
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        
        R = 6371.0  # Earth's radius in kilometers

        # Convert coordinates to radians
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        # Apply Haversine formula
        a = (math.sin(delta_phi / 2) ** 2 +
            math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c