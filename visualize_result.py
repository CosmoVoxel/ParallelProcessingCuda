import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

folder_path = Path(f'C:\\Users\\{os.getlogin()}\\metrix_results')
final_csv_path = folder_path / 'combined_results.csv'

df = pd.read_csv(final_csv_path)

print("Available columns in data:")
print(df.columns.tolist())
print("\nUnique metrics:")
print(df['Metric Name'].unique())
print("\nSample of data:")
print(df.head(10))

metrics = [
    'flop_count_sp',
    'flop_sp_efficiency',
    'achieved_occupancy', 
    'shared_load_traactio',
    'shared_store_traactio',
    'dram_read_throughput',
    'dram_write_throughput'
]

unique_configs = df[['elements_per_thread_x', 'elements_per_thread_y', 'block_size']].drop_duplicates()
unique_configs = unique_configs.sort_values(['block_size', 'elements_per_thread_x', 'elements_per_thread_y'])

print(f"\nFound {len(unique_configs)} unique")

color_map = {16: 'steelblue', 32: 'darkorange'}

for metric in metrics:
    if metric not in df['Metric Name'].values:
        print(f"Warning: Metric '{metric}' not found in data")
        continue
        
    plt.figure(figsize=(16, 8))
    
    x_labels = []
    values = []
    colors = []
    
    for _, config in unique_configs.iterrows():
        mask = ((df['elements_per_thread_x'] == config['elements_per_thread_x']) & 
                (df['elements_per_thread_y'] == config['elements_per_thread_y']) & 
                (df['block_size'] == config['block_size']) &
                (df['Metric Name'] == metric))
        
        matching_rows = df[mask]
        
        if not matching_rows.empty:
            value = matching_rows['Avg'].iloc[0]
            
            if pd.isna(value):
                print(f"Warning: NaN value for {metric} in config {config['elements_per_thread_x']}x{config['elements_per_thread_y']}_B{config['block_size']}")
                continue
                
            x_labels.append(f"{config['elements_per_thread_x']}x{config['elements_per_thread_y']}")
            values.append(float(value))
            colors.append(color_map.get(config['block_size'], 'gray'))
    
    if not values:
        print(f"No valid data found for metric: {metric}")
        continue
        
    bars = plt.bar(range(len(x_labels)), values, color=colors)
    
    plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha='right')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'{metric.replace("_", " ").title()}')
    plt.grid(axis='y', alpha=0.3)
    
    legend_elements = []
    for block_size, color in color_map.items():
        if block_size in unique_configs['block_size'].values:
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, label=f'Block Size {block_size}'))
    
    if legend_elements:
        plt.legend(handles=legend_elements, loc='upper right')
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        if value >= 1000000:
            text = f'{value/1000000:.1f}M'
        elif value >= 1000:
            text = f'{value/1000:.1f}K'
        else:
            text = f'{value:.2f}'
            
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                text, ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    output_path = folder_path / f'{metric}_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    plt.show()

available_metrics = [m for m in metrics if m in df['Metric Name'].values]

if available_metrics:
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(available_metrics[:8]):
        ax = axes[idx]
        
        x_labels = []
        values = []
        colors = []
        
        for _, config in unique_configs.iterrows():
            mask = ((df['elements_per_thread_x'] == config['elements_per_thread_x']) & 
                    (df['elements_per_thread_y'] == config['elements_per_thread_y']) & 
                    (df['block_size'] == config['block_size']) &
                    (df['Metric Name'] == metric))
            
            matching_rows = df[mask]
            
            if not matching_rows.empty:
                value = matching_rows['Avg'].iloc[0]
                if not pd.isna(value):
                    x_labels.append(f"{config['elements_per_thread_x']}x{config['elements_per_thread_y']}")
                    values.append(float(value))
                    colors.append(color_map.get(config['block_size'], 'gray'))
        
        if values:
            bars = ax.bar(range(len(x_labels)), values, color=colors)
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=6)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
            ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12)
            ax.grid(axis='y', alpha=0.3)
    
    for i in range(len(available_metrics), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    summary_path = folder_path / 'metrics_summary.png'
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"Saved summary: {summary_path}")
    plt.show()

print("✅ Visualization complete!")
print(f"Processed metrics: {available_metrics}")