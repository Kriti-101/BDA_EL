# Water Pollution MapReduce Analysis
# Complete implementation for analyzing CSV water pollution dataset

# ===================================================================
# FILE: run_analysis.py
# MASTER EXECUTION SCRIPT
# ===================================================================

# Water Pollution MapReduce Analysis
# Modified to save results into "results/" directory

import subprocess
import sys
import os

def run_mapreduce_analysis(data_file):
    """
    Run all MapReduce analyses on the water pollution dataset
    and save outputs to results directory.
    """

    print("=== Water Pollution MapReduce Analysis ===\n")

    if not os.path.exists(data_file):
        print(f"Error: Data file '{data_file}' not found!")
        return

    # Create results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    analyses = [
        {
            'name': 'Seasonal Pollution Analysis',
            'script': 'seasonal_analysis.py',
            'description': 'Analyzing monthly pollution variations...',
            'output_file': 'seasonal_analysis.txt'
        },
        {
            'name': 'Geographical Hotspot Analysis',
            'script': 'hotspot_analysis.py',
            'args': ['--grid-size=0.1', '--pollution-threshold=0.01'],
            'description': 'Identifying pollution hotspots...',
            'output_file': 'hotspot_analysis.txt'
        },
        {
            'name': 'Pollution Severity Classification',
            'script': 'severity_classification.py',
            'description': 'Classifying pollution severity levels...',
            'output_file': 'severity_classification.txt'
        },
        {
            'name': 'Runoff-Pollution Correlation',
            'script': 'correlation_analysis.py',
            'description': 'Analyzing runoff-pollution correlations...',
            'output_file': 'correlation_analysis.txt'
        }
    ]

    for analysis in analyses:
        print(f"Running {analysis['name']}...")
        print(f"Description: {analysis['description']}")

        cmd = ['python', analysis['script']]
        if 'args' in analysis:
            cmd.extend(analysis['args'])
        cmd.append(data_file)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            output_path = os.path.join(results_dir, analysis['output_file'])

            if result.returncode == 0:
                print("✓ Analysis completed successfully")
                print(f"→ Saving output to: {output_path}")
                with open(output_path, "w") as f:
                    f.write(result.stdout)
            else:
                print("✗ Analysis failed")
                print("→ Saving error to:", output_path)
                with open(output_path, "w") as f:
                    f.write("Error:\n" + result.stderr)

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

# import subprocess
# import sys
# import os

# def run_mapreduce_analysis(data_file):
#     """
#     Run all MapReduce analyses on the water pollution dataset
#     """
    
#     print("=== Water Pollution MapReduce Analysis ===\n")
    
#     if not os.path.exists(data_file):
#         print(f"Error: Data file '{data_file}' not found!")
#         return
    
#     analyses = [
#         {
#             'name': 'Seasonal Pollution Analysis',
#             'script': 'seasonal_analysis.py',
#             'description': 'Analyzing monthly pollution variations...'
#         },
#         {
#             'name': 'Geographical Hotspot Analysis', 
#             'script': 'hotspot_analysis.py',
#             'args': ['--grid-size=0.1', '--pollution-threshold=0.01'],
#             'description': 'Identifying pollution hotspots...'
#         },
#         {
#             'name': 'Pollution Severity Classification',
#             'script': 'severity_classification.py', 
#             'description': 'Classifying pollution severity levels...'
#         },
#         {
#             'name': 'Runoff-Pollution Correlation',
#             'script': 'correlation_analysis.py',
#             'description': 'Analyzing runoff-pollution correlations...'
#         }
#     ]
    
#     for analysis in analyses:
#         print(f"Running {analysis['name']}...")
#         print(f"Description: {analysis['description']}")
        
#         # Build command
#         cmd = ['python', analysis['script']]
#         if 'args' in analysis:
#             cmd.extend(analysis['args'])
#         cmd.append(data_file)
        
#         try:
#             # Run the analysis
#             result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
#             if result.returncode == 0:
#                 print("✓ Analysis completed successfully")
#                 print("Results:")
#                 print(result.stdout)
#             else:
#                 print("✗ Analysis failed")
#                 print("Error:", result.stderr)
                
#         except subprocess.TimeoutExpired:
#             print("✗ Analysis timed out (5 minutes)")
#         except Exception as e:
#             print(f"✗ Error running analysis: {e}")
        
#         print("-" * 60)

# if __name__ == '__main__':
#     if len(sys.argv) != 2:
#         print("Usage: python run_analysis.py <data_file>")
#         print("Example: python run_analysis.py water_pollution_data.txt")
#         sys.exit(1)
    
#     data_file = sys.argv[1]
#     run_mapreduce_analysis(data_file)

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