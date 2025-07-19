# Water Pollution MapReduce Analysis

Complete MapReduce implementation for analyzing water pollution datasets with geographical coordinates, monthly pollution data, and runoff information.

## File Structure

```
water-pollution-mapreduce/
│
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── setup.sh                      # Setup script
├── run_analysis.py              # Master execution script
│
├── seasonal_analysis.py         # Objective 1: Seasonal patterns
├── hotspot_analysis.py          # Objective 2: Geographical hotspots  
├── severity_classification.py   # Objective 3: Pollution severity
├── correlation_analysis.py      # Objective 4: Runoff correlation
│
├── data/
│   └── water_pollution_data.csv  # Your CSV dataset (place here)
│
└── results/
    ├── seasonal_results.txt      # Output from seasonal analysis
    ├── hotspot_results.txt       # Output from hotspot analysis
    ├── severity_results.txt      # Output from severity classification
    └── correlation_results.txt   # Output from correlation analysis
```

## Dataset Format

Your dataset should be a CSV file with the following columns:
```
X,Y,i_mid,i_low,i_high,i_mid_jan,i_low_jan,i_high_jan,i_mid_feb,i_low_feb,i_high_feb,i_mid_mar,i_low_mar,i_high_mar,i_mid_apr,i_low_apr,i_high_apr,i_mid_may,i_low_may,i_high_may,i_mid_jun,i_low_jun,i_high_jun,i_mid_jul,i_low_jul,i_high_jul,i_mid_aug,i_low_aug,i_high_aug,i_mid_sep,i_low_sep,i_high_sep,i_mid_oct,i_low_oct,i_high_oct,i_mid_nov,i_low_nov,i_high_nov,i_mid_dec,i_low_dec,i_high_dec,runoff_jan,runoff_feb,runoff_mar,runoff_apr,runoff_may,runoff_jun,runoff_jul,runoff_aug,runoff_sep,runoff_oct,runoff_nov,runoff_dec,mpw,area
```

Sample data row:
```
9.8125,37.32917,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,21753.3,22500000
```

Where:
- `X, Y`: Geographical coordinates
- `i_mid, i_low, i_high`: Overall pollution intensity (mid, low, high estimates)
- `i_mid_month, i_low_month, i_high_month`: Monthly pollution data
- `runoff_month`: Monthly runoff data
- `mpw`: Microplastics in water
- `area`: Area coverage

## Installation & Setup

### 1. Clone or Create Project Directory
```bash
mkdir water-pollution-mapreduce
cd water-pollution-mapreduce
```

### 2. Install Dependencies
```bash
pip install mrjob
```

### 3. Create Required Directories
```bash
mkdir data results
```

### 4. Place Your Data
Put your CSV water pollution dataset in the `data/` directory:
```bash
cp your_dataset.csv data/water_pollution_data.csv
```

## Execution Instructions

### Option 1: Run All Analyses (Recommended)
```bash
python run_analysis.py data/water_pollution_data.csv
```

### Option 2: Run Individual Analyses

#### 1. Seasonal Pollution Analysis
Analyzes monthly pollution variations across locations.
```bash
python seasonal_analysis.py data/water_pollution_data.csv > results/seasonal_results.txt
```

#### 2. Geographical Hotspot Analysis
Identifies high-pollution geographical clusters.
```bash
python hotspot_analysis.py --grid-size=0.1 --pollution-threshold=0.01 data/water_pollution_data.csv > results/hotspot_results.txt
```

Parameters:
- `--grid-size`: Spatial grid size for clustering (default: 0.1)
- `--pollution-threshold`: Minimum pollution level to consider (default: 0.01)

#### 3. Pollution Severity Classification
Classifies locations into Low, Medium, High, Critical categories.
```bash
python severity_classification.py data/water_pollution_data.csv > results/severity_results.txt
```

#### 4. Runoff-Pollution Correlation Analysis
Analyzes correlation between water runoff and pollution levels.
```bash
python correlation_analysis.py data/water_pollution_data.csv > results/correlation_results.txt
```

## Analysis Objectives

### 1. Seasonal Pollution Analysis
- **Purpose**: Identify seasonal patterns in pollution levels
- **Output**: Monthly statistics including average, max, min pollution levels
- **Key Insights**: Seasonal variations, peak pollution months

### 2. Geographical Hotspot Identification
- **Purpose**: Find geographical areas with high pollution concentration
- **Method**: Grid-based spatial clustering
- **Output**: Hotspot locations with pollution scores
- **Key Insights**: Pollution clusters, high-risk areas

### 3. Pollution Severity Classification
- **Purpose**: Categorize locations by pollution severity
- **Categories**: Low, Medium, High, Critical
- **Method**: Percentile-based thresholds
- **Key Insights**: Distribution of pollution severity

### 4. Runoff-Pollution Correlation
- **Purpose**: Analyze relationship between runoff and pollution
- **Method**: Pearson correlation coefficient
- **Output**: Correlation strength and patterns
- **Key Insights**: Water flow impact on pollution levels

## Sample Output

### Seasonal Analysis Output
```json
"jan": {
    "average_pollution": 0.045123,
    "max_pollution": 0.234567,
    "min_pollution": 0.000123,
    "std_deviation": 0.067890,
    "active_locations": 1250,
    "total_pollution": 56.403750
}
```

### Hotspot Analysis Output
```json
"HOTSPOT": {
    "grid": "grid_9.8_37.3",
    "locations_count": 15,
    "total_pollution": 2.456789,
    "average_density": 0.123456,
    "max_pollution": 0.456789,
    "hotspot_score": 4.567890,
    "sample_locations": ["9.8125,37.32917", "9.7875,37.31234"]
}
```

### Severity Classification Output
```json
"HIGH": {
    "count": 234,
    "avg_pollution": 0.156789,
    "avg_density": 0.089123,
    "sample_locations": ["9.8125,37.32917", "10.1250,36.98765"]
}
```

### Correlation Analysis Output
```json
"STRONG": {
    "total_locations": 45,
    "positive_correlations": 38,
    "negative_correlations": 7,
    "avg_correlation": 0.7834,
    "avg_pollution": 0.089123,
    "avg_runoff": 0.456789
}
```

## Troubleshooting

### Common Issues

1. **"No module named 'mrjob'"**
   ```bash
   pip install mrjob
   ```

2. **"Permission denied"**
   ```bash
   chmod +x *.py
   ```

3. **"File not found"**
   - Ensure your data file is in the correct location
   - Check file path and name

4. **Empty results**
   - Check if your data has the expected format
   - Verify column indices match your dataset
   - Ensure pollution values are numeric

### Data Format Issues

The code now properly handles CSV format with comma separation. If you encounter issues:

1. **Different CSV dialects**: The code uses Python's standard CSV parser which handles most formats
2. **Quoted fields**: CSV parser handles quoted fields automatically
3. **Different encodings**: Ensure your CSV is UTF-8 encoded

### Performance Optimization

For large CSV datasets:
1. Increase grid size for hotspot analysis: `--grid-size=0.5`
2. Increase pollution threshold: `--pollution-threshold=0.1`
3. Use local mode: Add `-r local` to commands

## Advanced Usage

### Running on Hadoop Cluster
```bash
python seasonal_analysis.py -r hadoop hdfs:///path/to/data/water_pollution_data.csv
```

### Running on AWS EMR
```bash
python seasonal_analysis.py -r emr s3://your-bucket/water_pollution_data.csv
```

### Custom Configuration
Create `mrjob.conf`:
```yaml
runners:
  local:
    jobconf:
      mapred.map.tasks: 4
      mapred.reduce.tasks: 2
```

## Output Interpretation

### Seasonal Patterns
- High `std_deviation`: Indicates seasonal variability
- High `max_pollution`: Identifies peak pollution periods

### Hotspots
- High `hotspot_score`: Critical areas requiring attention
- High `locations_count`: Large affected areas

### Severity Classification
- Monitor `CRITICAL` category locations
- Track trends in severity distribution

### Correlation Analysis
- Strong positive correlation: Runoff increases pollution
- Strong negative correlation: Runoff dilutes pollution

## Contributing

To extend the analysis:
1. Create new MapReduce job class
2. Follow the existing pattern: mapper → reducer
3. Add to `run_analysis.py` for integration

## License

This project is provided as-is for educational and research purposes.
