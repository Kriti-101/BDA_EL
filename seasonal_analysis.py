
# ===================================================================
# FILE: seasonal_analysis.py
# OBJECTIVE 1: SEASONAL POLLUTION PATTERN ANALYSIS
# ===================================================================

from mrjob.job import MRJob
from mrjob.step import MRStep
import json
import math
import csv
from io import StringIO

class SeasonalPollutionAnalysis(MRJob):
    """
    Analyze seasonal pollution patterns across different locations
    """
    
    def mapper(self, _, line):
        """Extract location and monthly pollution data"""
        try:
            if line.startswith('X'):  # Skip header
                return
            
            # Parse CSV line
            csv_reader = csv.reader(StringIO(line))
            parts = next(csv_reader)
            
            if len(parts) < 55:
                return
            
            x, y = float(parts[0]), float(parts[1])
            location = f"{x},{y}"
            
            # Extract monthly pollution intensities (i_mid columns)
            monthly_data = {
                'jan': float(parts[4]),   # i_mid_jan
                'feb': float(parts[7]),   # i_mid_feb  
                'mar': float(parts[10]),  # i_mid_mar
                'apr': float(parts[13]),  # i_mid_apr
                'may': float(parts[16]),  # i_mid_may
                'jun': float(parts[19]),  # i_mid_jun
                'jul': float(parts[22]),  # i_mid_jul
                'aug': float(parts[25]),  # i_mid_aug
                'sep': float(parts[28]),  # i_mid_sep
                'oct': float(parts[31]),  # i_mid_oct
                'nov': float(parts[34]),  # i_mid_nov
                'dec': float(parts[37])   # i_mid_dec
            }
            
            # Emit each month's data with location
            for month, value in monthly_data.items():
                if value > 0:  # Only consider positive pollution values
                    yield f"season_{month}", (location, value)
                    
        except (ValueError, IndexError) as e:
            pass
    
    def reducer(self, key, values):
        """Calculate seasonal statistics"""
        month = key.split('_')[1]
        location_values = list(values)
        
        values_only = [v[1] for v in location_values]
        
        if values_only:
            total = sum(values_only)
            count = len(values_only)
            avg = total / count
            max_val = max(values_only)
            min_val = min(values_only)
            
            # Calculate standard deviation
            variance = sum((x - avg) ** 2 for x in values_only) / count
            std_dev = math.sqrt(variance)
            
            yield month, {
                'average_pollution': round(avg, 6),
                'max_pollution': round(max_val, 6),
                'min_pollution': round(min_val, 6),
                'std_deviation': round(std_dev, 6),
                'active_locations': count,
                'total_pollution': round(total, 6)
            }

if __name__ == '__main__':
    SeasonalPollutionAnalysis.run()