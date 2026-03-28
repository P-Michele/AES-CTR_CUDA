import pandas as pd
import matplotlib.pyplot as plt
import os
import re

file_path = os.path.join(os.path.dirname(__file__), "data.csv")

try:
    data = []
    regex_pattern = r"test_(\d+)GB\.bin,\s*Time:\s*([\d\.]+)\s*ms"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r') as f:
        for line in f:
            if "GPU Test" or "CPU Test" in line:
                match = re.search(regex_pattern, line)
                if match:
                    size_gb = int(match.group(1))
                    execution_time = float(match.group(2))
                    data.append({
                        "Size": size_gb, 
                        "Label": f"{size_gb} GB", 
                        "Time": execution_time
                    })

    if not data:
        raise ValueError("No data found. Check if the file contains 'GPU Test' lines.")

    df = pd.DataFrame(data).sort_values('Size')

    # Calcolo crescita ideale basata sul primo test (1 GB)
    time_1gb = df.iloc[0]['Time']
    df['Ideal_Time'] = df['Size'] * time_1gb

    plt.figure(figsize=(12, 8))
    
    bars = plt.bar(df['Label'], df['Time'], color='#2ecc71', alpha=0.7, 
                   label='Actual GPU Time', edgecolor='#27ae60')

    plt.plot(df['Label'], df['Ideal_Time'], color='#e74c3c', marker='o', 
             linestyle='--', linewidth=2, label='Ideal Linear Scaling')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (yval * 0.02), 
                 f'{yval:,.1f} ms', ha='center', va='bottom', 
                 fontsize=10, fontweight='bold')

    plt.xlabel('Dataset Size', fontsize=12, fontweight='bold')
    plt.ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    plt.title('GPU Performance: Actual vs. Ideal Linear Scaling', fontsize=15, pad=20)
    plt.legend(loc='upper left')
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Error: {e}")