from pathlib import Path
import pandas as pd
import re
import os

folder_path = Path(f'C:\\Users\\{os.getlogin()}\\metrix_results')

csv_files = list(folder_path.glob('*.csv'))

if not csv_files:
    print("No CSV files found in the folder!")
    exit()

print(f"Found {len(csv_files)} CSV files")

def clean_numeric_data(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)
            
            df[col] = df[col].str.replace(r'%', '', regex=True)
            df[col] = df[col].str.replace(r'GB/s', '', regex=True)
            df[col] = df[col].str.replace(r'MB/s', '', regex=True)
            df[col] = df[col].str.replace(r'KB/s', '', regex=True)
            df[col] = df[col].str.replace(r'GHz', '', regex=True)
            df[col] = df[col].str.replace(r'MHz', '', regex=True)
            df[col] = df[col].str.replace(r'ms', '', regex=True)
            df[col] = df[col].str.replace(r'us', '', regex=True)
            df[col] = df[col].str.replace(r'ns', '', regex=True)
            df[col] = df[col].str.replace(r'B', '', regex=True)
            df[col] = df[col].str.replace(r',', '', regex=True)
            
            try:
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                if not numeric_col.isna().all():
                    df[col] = numeric_col
            except:
                pass
    
    return df

def extract_parameters(file_name):
    name = file_name.stem
    print(f"Processing file: {name}")
    
    params = {}
    
    if 'elements_per_thread_x_' in name:
        start = name.find('elements_per_thread_x_') + len('elements_per_thread_x_')
        end = name.find('_', start)
        if end == -1:
            end = len(name)
        params['elements_per_thread_x'] = int(name[start:end])
    
    if 'elements_per_thread_y_' in name:
        start = name.find('elements_per_thread_y_') + len('elements_per_thread_y_')
        end = name.find('_', start)
        if end == -1:
            end = len(name)
        params['elements_per_thread_y'] = int(name[start:end])
    
    if 'block_size_' in name:
        start = name.find('block_size_') + len('block_size_')
        end = name.find('_', start)
        if end == -1:
            end = len(name)
        params['block_size'] = int(name[start:end])
    
    return params

dataframes = []
for file in csv_files:
    try:
        df = pd.read_csv(file)
        
        df = clean_numeric_data(df)
        
        params = extract_parameters(file)
        
        for key, value in params.items():
            df[key] = value
            
        dataframes.append(df)
        print(f"Successfully processed: {file.name}")
        
    except Exception as e:
        print(f"Error processing {file.name}: {e}")

if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    columns_to_delete = ['Device', 'Kernel', 'Invocations', 'Metric Description']
    for col in columns_to_delete:
        if col in combined_df.columns:
            combined_df = combined_df.drop(columns=[col])
            print(f"Deleted column: {col}")
    
    param_columns = ['elements_per_thread_x', 'elements_per_thread_y', 'block_size']
    other_columns = [col for col in combined_df.columns if col not in param_columns]
    
    new_column_order = param_columns + other_columns
    combined_df = combined_df[new_column_order]
    
    output_file = folder_path / 'combined_results.csv'
    combined_df.to_csv(output_file, index=False)
    
    print(f"\n✅ Combined {len(dataframes)} CSV files")
    print(f"📊 Total rows: {len(combined_df)}")
    print(f"📁 Saved to: {output_file}")
    
    print(f"\n📋 Columns: {list(combined_df.columns)}")
    print(f"\n🔍 First 5 rows:")
    print(combined_df.head())
    
    print(f"\n🔢 Data types:")
    print(combined_df.dtypes)
    
else:
    print("❌ No valid CSV files were processed!")