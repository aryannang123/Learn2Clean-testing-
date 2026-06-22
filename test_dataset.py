#!/usr/bin/env python3
"""
Dataset Testing Script for Learn2Clean
Usage: python test_dataset.py [dataset_name]
Example: python test_dataset.py crimes.csv
"""

import sys
import pandas as pd
from pathlib import Path

def test_dataset(dataset_name):
    """Test a dataset and show its properties"""
    
    print("=" * 60)
    print(f"📊 DATASET ANALYSIS: {dataset_name}")
    print("=" * 60)
    
    # Try to load the dataset
    try:
        if dataset_name.endswith('.csv'):
            data_path = Path('data') / dataset_name
            if not data_path.exists():
                print(f"❌ File not found: {data_path}")
                return False
            
            df = pd.read_csv(data_path)
        else:
            print(f"❌ Unsupported file type: {dataset_name}")
            return False
            
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False
    
    # Basic Info
    print(f"✅ Dataset loaded successfully!")
    print(f"📏 Shape: {df.shape} (rows × columns)")
    print(f"💾 Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    print()
    
    # Column Information
    print("📋 COLUMN INFORMATION:")
    print("-" * 40)
    for i, col in enumerate(df.columns):
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        unique_count = df[col].nunique()
        
        print(f"{i+1:2d}. {col:<20} | {str(dtype):<10} | Nulls: {null_count:4d} ({null_pct:5.1f}%) | Unique: {unique_count}")
    
    print()
    
    # Missing Data Analysis
    total_missing = df.isnull().sum().sum()
    missing_pct = (total_missing / (len(df) * len(df.columns))) * 100
    
    print("🕳️  MISSING DATA ANALYSIS:")
    print("-" * 40)
    print(f"Total missing values: {total_missing:,}")
    print(f"Missing percentage: {missing_pct:.2f}%")
    
    if total_missing > 0:
        print("\nColumns with missing data:")
        missing_cols = df.isnull().sum()
        missing_cols = missing_cols[missing_cols > 0].sort_values(ascending=False)
        for col, count in missing_cols.items():
            pct = (count / len(df)) * 100
            print(f"  • {col}: {count} ({pct:.1f}%)")
    
    print()
    
    # Data Types Analysis
    print("🔢 DATA TYPES ANALYSIS:")
    print("-" * 40)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    
    print(f"Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols) if numeric_cols else 'None'}")
    print(f"Categorical columns ({len(categorical_cols)}): {', '.join(categorical_cols) if categorical_cols else 'None'}")
    print(f"DateTime columns ({len(datetime_cols)}): {', '.join(datetime_cols) if datetime_cols else 'None'}")
    
    print()
    
    # Sample Data
    print("👀 SAMPLE DATA (first 5 rows):")
    print("-" * 40)
    print(df.head())
    
    print()
    
    # Potential Target Columns
    print("🎯 POTENTIAL TARGET COLUMNS:")
    print("-" * 40)
    
    # Look for common target column names
    potential_targets = []
    target_keywords = ['target', 'label', 'class', 'category', 'survived', 'outcome', 'result', 'y']
    
    for col in df.columns:
        col_lower = col.lower()
        for keyword in target_keywords:
            if keyword in col_lower:
                potential_targets.append(col)
                break
    
    # Also check for binary/categorical columns with few unique values
    for col in categorical_cols:
        unique_vals = df[col].nunique()
        if 2 <= unique_vals <= 10:  # Good target candidate
            if col not in potential_targets:
                potential_targets.append(col)
    
    if potential_targets:
        print("Suggested target columns:")
        for col in potential_targets:
            unique_vals = df[col].nunique()
            sample_vals = df[col].dropna().unique()[:5]
            print(f"  • {col}: {unique_vals} unique values {list(sample_vals)}")
    else:
        print("No obvious target column found. Manual inspection needed.")
    
    print()
    
    # Learn2Clean Compatibility
    print("🤖 LEARN2CLEAN COMPATIBILITY:")
    print("-" * 40)
    
    issues = []
    recommendations = []
    
    # Check for common issues
    if len(df) < 100:
        issues.append("Dataset is very small (< 100 rows)")
    elif len(df) < 500:
        recommendations.append("Small dataset - consider shorter training")
    
    if missing_pct > 50:
        issues.append("High missing data percentage (> 50%)")
    elif missing_pct > 20:
        recommendations.append("Moderate missing data - good for imputation testing")
    
    if len(numeric_cols) == 0:
        issues.append("No numeric columns found")
    
    if len(potential_targets) == 0:
        issues.append("No clear target column identified")
    
    if issues:
        print("⚠️  Potential issues:")
        for issue in issues:
            print(f"  • {issue}")
    
    if recommendations:
        print("💡 Recommendations:")
        for rec in recommendations:
            print(f"  • {rec}")
    
    if not issues and not recommendations:
        print("✅ Dataset looks good for Learn2Clean!")
    
    print()
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_dataset.py <dataset_name>")
        print("Examples:")
        print("  python test_dataset.py titanic.csv")
        print("  python test_dataset.py crimes.csv")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    test_dataset(dataset_name)