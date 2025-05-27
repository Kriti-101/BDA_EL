
# ===================================================================
# FILE: severity_classification.py
# OBJECTIVE 3: POLLUTION SEVERITY CLASSIFICATION  
# ===================================================================

from mrjob.job import MRJob
from mrjob.step import MRStep
import math
import csv
from io import StringIO

class PollutionSeverityClassification(MRJob):
    """
    Classify locations based on pollution severity levels
    Categories: Low, Medium, High, Critical
    """
    
    def steps(self):
        return [
            MRStep(mapper=self.mapper_extract_features,
                  reducer=self.reducer_calculate_thresholds),
            MRStep(mapper=self.mapper_classify,
                  reducer=self.reducer_aggregate_classes)
        ]
    
    def mapper_extract_features(self, _, line):
        """Extract pollution features for threshold calculation"""
        try:
            if line.startswith('X'):
                return
                
            # Parse CSV line
            csv_reader = csv.reader(StringIO(line))
            parts = next(csv_reader)
            
            if len(parts) < 55:
                return
            
            x, y = float(parts[0]), float(parts[1])
            avg_pollution = float(parts[2])  # i_mid
            high_pollution = float(parts[4])  # i_high
            area = float(parts[54])
            
            if avg_pollution > 0:  # Only consider locations with pollution
                # Calculate additional features
                pollution_density = avg_pollution / (area / 1000000) if area > 0 else 0
                
                # Calculate seasonal variation from monthly data
                monthly_values = []
                for i in [4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37]:  # i_mid_month columns
                    val = float(parts[i])
                    if val > 0:
                        monthly_values.append(val)
                
                seasonal_variance = 0
                if len(monthly_values) > 1:
                    mean_val = sum(monthly_values) / len(monthly_values)
                    seasonal_variance = sum((x - mean_val) ** 2 for x in monthly_values) / len(monthly_values)
                
                yield "THRESHOLD_CALC", {
                    'location': f"{x},{y}",
                    'avg_pollution': avg_pollution,
                    'high_pollution': high_pollution,
                    'density': pollution_density,
                    'variance': seasonal_variance
                }
        except (ValueError, IndexError):
            pass
    
    def reducer_calculate_thresholds(self, key, values):
        """Calculate classification thresholds"""
        data = list(values)
        pollution_values = [d['avg_pollution'] for d in data]
        
        if pollution_values:
            pollution_values.sort()
            n = len(pollution_values)
            
            # Calculate percentile-based thresholds
            low_threshold = pollution_values[int(n * 0.25)]
            medium_threshold = pollution_values[int(n * 0.50)]
            high_threshold = pollution_values[int(n * 0.75)]
            critical_threshold = pollution_values[int(n * 0.90)]
            
            # Re-emit all data with thresholds for second phase
            for item in data:
                item['thresholds'] = {
                    'low': low_threshold,
                    'medium': medium_threshold,
                    'high': high_threshold,
                    'critical': critical_threshold
                }
                yield "CLASSIFY", item
    
    def mapper_classify(self, key, value):
        """Apply classification"""
        if key == "CLASSIFY":
            pollution = value['avg_pollution']
            thresholds = value['thresholds']
            
            # Classify based on thresholds
            if pollution <= thresholds['low']:
                category = "LOW"
            elif pollution <= thresholds['medium']:
                category = "MEDIUM"
            elif pollution <= thresholds['high']:
                category = "HIGH"
            else:
                category = "CRITICAL"
            
            yield category, {
                'location': value['location'],
                'pollution': pollution,
                'density': value['density'],
                'variance': value['variance']
            }
    
    def reducer_aggregate_classes(self, category, values):
        """Aggregate classification results"""
        locations = list(values)
        
        if locations:
            yield category, {
                'count': len(locations),
                'avg_pollution': round(sum(loc['pollution'] for loc in locations) / len(locations), 6),
                'avg_density': round(sum(loc['density'] for loc in locations) / len(locations), 6),
                'sample_locations': [loc['location'] for loc in locations[:10]]
            }

if __name__ == '__main__':
    PollutionSeverityClassification.run()
