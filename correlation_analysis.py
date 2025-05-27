
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