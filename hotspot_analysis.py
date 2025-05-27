
# ===================================================================
# FILE: hotspot_analysis.py  
# OBJECTIVE 2: GEOGRAPHICAL HOTSPOT IDENTIFICATION
# ===================================================================

from mrjob.job import MRJob
import math
import csv
from io import StringIO

class GeographicalHotspotAnalysis(MRJob):
    """
    Identify pollution hotspots using grid-based spatial clustering
    """
    
    def configure_args(self):
        super(GeographicalHotspotAnalysis, self).configure_args()
        self.add_passthru_arg('--grid-size', type=float, default=0.1,
                            help='Grid size for spatial clustering')
        self.add_passthru_arg('--pollution-threshold', type=float, default=0.01,
                            help='Minimum pollution level to consider')
    
    def mapper(self, _, line):
        """Grid-based spatial grouping"""
        try:
            if line.startswith('X'):
                return
                
            # Parse CSV line
            csv_reader = csv.reader(StringIO(line))
            parts = next(csv_reader)
            
            if len(parts) < 55:
                return
            
            x, y = float(parts[0]), float(parts[1])
            avg_pollution = float(parts[2])  # i_mid column
            area = float(parts[54])
            
            if avg_pollution > self.options.pollution_threshold:
                # Create spatial grid
                grid_size = self.options.grid_size
                grid_x = round(int(x / grid_size) * grid_size, 4)
                grid_y = round(int(y / grid_size) * grid_size, 4)
                grid_key = f"grid_{grid_x}_{grid_y}"
                
                # Calculate pollution density per km²
                pollution_density = avg_pollution / (area / 1000000) if area > 0 else 0
                
                yield grid_key, {
                    'location': f"{x},{y}",
                    'pollution': avg_pollution,
                    'density': pollution_density,
                    'area': area
                }
        except (ValueError, IndexError):
            pass
    
    def reducer(self, grid_key, values):
        """Identify hotspots within each grid"""
        locations = list(values)
        
        if len(locations) >= 2:  # Cluster requires multiple points
            total_pollution = sum(loc['pollution'] for loc in locations)
            avg_density = sum(loc['density'] for loc in locations) / len(locations)
            max_pollution = max(loc['pollution'] for loc in locations)
            
            # Hotspot criteria: high pollution concentration + multiple locations
            hotspot_score = (total_pollution * len(locations) * avg_density) / 1000
            
            if hotspot_score > 0.1:  # Threshold for hotspot classification
                yield "HOTSPOT", {
                    'grid': grid_key,
                    'locations_count': len(locations),
                    'total_pollution': round(total_pollution, 6),
                    'average_density': round(avg_density, 6),
                    'max_pollution': round(max_pollution, 6),
                    'hotspot_score': round(hotspot_score, 6),
                    'sample_locations': [loc['location'] for loc in locations[:5]]
                }

if __name__ == '__main__':
    GeographicalHotspotAnalysis.run()
