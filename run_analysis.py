# Water Pollution MapReduce Analysis
# Complete implementation for analyzing CSV water pollution dataset

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

# ===================================================================
# FILE: hotspot_analysis.py  
# OBJECTIVE 2: GEOGRAPHICAL HOTSPOT IDENTIFICATION
# ===================================================================

from mrjob.job import MRJob
import math

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

# ===================================================================
# FILE: correlation_analysis.py
# OBJECTIVE 4: RUNOFF-POLLUTION CORRELATION ANALYSIS
# ===================================================================

from mrjob.job import MRJob
from mrjob.step import MRStep
import math
import csv
from io import StringIO

class RunoffPollutionCorrelation(MRJob):
    """
    Analyze correlation between water runoff and pollution levels
    """
    
    def steps(self):
        return [
            MRStep(mapper=self.mapper_extract_pairs,
                  reducer=self.reducer_calculate_correlation),
            MRStep(mapper=self.mapper_correlation_strength,
                  reducer=self.reducer_final_analysis)
        ]
    
    def mapper_extract_pairs(self, _, line):
        """Extract runoff-pollution pairs for each location"""
        try:
            if line.startswith('X'):
                return
                
            # Parse CSV line
            csv_reader = csv.reader(StringIO(line))
            parts = next(csv_reader)
            
            if len(parts) < 55:
                return
            
            x, y = float(parts[0]), float(parts[1])
            location = f"{x},{y}"
            
            # Extract monthly pollution and runoff data
            monthly_pairs = []
            pollution_cols = [4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37]  # i_mid_month columns
            runoff_cols = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]   # runoff_month columns
            
            for p_col, r_col in zip(pollution_cols, runoff_cols):
                pollution = float(parts[p_col])
                runoff = float(parts[r_col])
                
                if pollution > 0 and runoff > 0:
                    monthly_pairs.append((pollution, runoff))
            
            if len(monthly_pairs) >= 3:  # Need sufficient data points
                yield location, monthly_pairs
                
        except (ValueError, IndexError):
            pass
    
    def reducer_calculate_correlation(self, location, values):
        """Calculate Pearson correlation for each location"""
        all_pairs = []
        for pair_list in values:
            all_pairs.extend(pair_list)
        
        if len(all_pairs) >= 3:
            # Calculate Pearson correlation
            n = len(all_pairs)
            sum_x = sum(pair[0] for pair in all_pairs)  # pollution
            sum_y = sum(pair[1] for pair in all_pairs)  # runoff
            sum_xx = sum(pair[0] ** 2 for pair in all_pairs)
            sum_yy = sum(pair[1] ** 2 for pair in all_pairs)
            sum_xy = sum(pair[0] * pair[1] for pair in all_pairs)
            
            # Pearson correlation formula
            numerator = n * sum_xy - sum_x * sum_y
            denominator_sq = (n * sum_xx - sum_x ** 2) * (n * sum_yy - sum_y ** 2)
            
            if denominator_sq > 0:
                correlation = numerator / math.sqrt(denominator_sq)
                
                yield "CORRELATION", {
                    'location': location,
                    'correlation': round(correlation, 4),
                    'data_points': n,
                    'avg_pollution': round(sum_x / n, 6),
                    'avg_runoff': round(sum_y / n, 6)
                }
    
    def mapper_correlation_strength(self, key, value):
        """Classify correlation strength"""
        if key == "CORRELATION":
            corr = abs(value['correlation'])
            
            if corr >= 0.7:
                strength = "STRONG"
            elif corr >= 0.4:
                strength = "MODERATE"
            elif corr >= 0.2:
                strength = "WEAK"
            else:
                strength = "NEGLIGIBLE"
            
            yield strength, value
    
    def reducer_final_analysis(self, strength, values):
        """Final analysis of correlation patterns"""
        correlations = list(values)
        
        if correlations:
            positive_corr = [c for c in correlations if c['correlation'] > 0]
            negative_corr = [c for c in correlations if c['correlation'] < 0]
            
            yield strength, {
                'total_locations': len(correlations),
                'positive_correlations': len(positive_corr),
                'negative_correlations': len(negative_corr),
                'avg_correlation': round(sum(c['correlation'] for c in correlations) / len(correlations), 4),
                'avg_pollution': round(sum(c['avg_pollution'] for c in correlations) / len(correlations), 6),
                'avg_runoff': round(sum(c['avg_runoff'] for c in correlations) / len(correlations), 6),
                'sample_locations': [c['location'] for c in correlations[:5]]
            }

if __name__ == '__main__':
    RunoffPollutionCorrelation.run()

# ===================================================================
# FILE: run_analysis.py
# MASTER EXECUTION SCRIPT
# ===================================================================

import subprocess
import sys
import os

def run_mapreduce_analysis(data_file):
    """
    Run all MapReduce analyses on the water pollution dataset
    """
    
    print("=== Water Pollution MapReduce Analysis ===\n")
    
    if not os.path.exists(data_file):
        print(f"Error: Data file '{data_file}' not found!")
        return
    
    analyses = [
        {
            'name': 'Seasonal Pollution Analysis',
            'script': 'seasonal_analysis.py',
            'description': 'Analyzing monthly pollution variations...'
        },
        {
            'name': 'Geographical Hotspot Analysis', 
            'script': 'hotspot_analysis.py',
            'args': ['--grid-size=0.1', '--pollution-threshold=0.01'],
            'description': 'Identifying pollution hotspots...'
        },
        {
            'name': 'Pollution Severity Classification',
            'script': 'severity_classification.py', 
            'description': 'Classifying pollution severity levels...'
        },
        {
            'name': 'Runoff-Pollution Correlation',
            'script': 'correlation_analysis.py',
            'description': 'Analyzing runoff-pollution correlations...'
        }
    ]
    
    for analysis in analyses:
        print(f"Running {analysis['name']}...")
        print(f"Description: {analysis['description']}")
        
        # Build command
        cmd = ['python', analysis['script']]
        if 'args' in analysis:
            cmd.extend(analysis['args'])
        cmd.append(data_file)
        
        try:
            # Run the analysis
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✓ Analysis completed successfully")
                print("Results:")
                print(result.stdout)
            else:
                print("✗ Analysis failed")
                print("Error:", result.stderr)
                
        except subprocess.TimeoutExpired:
            print("✗ Analysis timed out (5 minutes)")
        except Exception as e:
            print(f"✗ Error running analysis: {e}")
        
        print("-" * 60)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python run_analysis.py <data_file>")
        print("Example: python run_analysis.py water_pollution_data.txt")
        sys.exit(1)
    
    data_file = sys.argv[1]
    run_mapreduce_analysis(data_file)

# ===================================================================
# FILE: requirements.txt
# ===================================================================

"""
mrjob==0.7.4
"""

# ===================================================================
# FILE: setup.sh
# SETUP SCRIPT
# ===================================================================

"""
#!/bin/bash
# Setup script for Water Pollution MapReduce Analysis

echo "Setting up Water Pollution MapReduce Analysis..."

# Install required packages
pip install mrjob

# Make scripts executable
chmod +x *.py
chmod +x setup.sh

echo "Setup completed!"
echo ""
echo "Usage:"
echo "1. Place your data file in the same directory"
echo "2. Run: python run_analysis.py your_data_file.txt"
echo ""
echo "Or run individual analyses:"
echo "- python seasonal_analysis.py your_data_file.txt"
echo "- python hotspot_analysis.py --grid-size=0.1 your_data_file.txt"
echo "- python severity_classification.py your_data_file.txt" 
echo "- python correlation_analysis.py your_data_file.txt"
"""