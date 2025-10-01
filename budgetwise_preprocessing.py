"""
BudgetWise Personal Finance Dataset Analysis
Dataset: https://www.kaggle.com/datasets/mohammedarfathr/budgetwise-personal-finance-dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class FinanceDataProcessor:
    """Class to handle finance data cleaning and analysis"""
    
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.df = None
        self.df_original = None
        self.cleaning_report = {}
        
    def load_data(self):
        """Load dataset with error handling"""
        try:
            self.df = pd.read_csv(self.file_path)
            self.df_original = self.df.copy()
            print("Dataset Loaded Successfully!")
            print(f"Shape: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
            print("\n" + "="*60)
            return True
        except FileNotFoundError:
            print(f"Error: File not found at {self.file_path}")
            return False
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def explore_data(self):
        """Initial data exploration"""
        print("\n DATASET OVERVIEW")
        print("="*60)
        print(self.df.head())
        
        print("\n COLUMN INFORMATION")
        print("="*60)
        print(self.df.info())
        
        print("\n NUMERICAL STATISTICS")
        print("="*60)
        print(self.df.describe())
        
        print("\n DATA QUALITY CHECK")
        print("="*60)
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        quality_df = pd.DataFrame({
            'Missing_Count': missing,
            'Missing_Percent': missing_pct.round(2)
        })
        quality_df = quality_df[quality_df['Missing_Count'] > 0].sort_values(
            'Missing_Count', ascending=False
        )
        if len(quality_df) > 0:
            print(quality_df)
        else:
            print("No missing values detected!")
        
        duplicates = self.df.duplicated().sum()
        print(f"\n Duplicate rows: {duplicates:,}")
        
        self.cleaning_report['original_shape'] = self.df.shape
        self.cleaning_report['initial_missing'] = missing.sum()
        self.cleaning_report['initial_duplicates'] = duplicates
        
    def clean_data(self):
        """Comprehensive data cleaning"""
        print("\n STARTING DATA CLEANING")
        print("="*60)
        
        # 1. Remove duplicates
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = before - len(self.df)
        print(f" Removed {removed:,} duplicate rows")
        self.cleaning_report['duplicates_removed'] = removed
        
        # 2. Handle date columns
        date_cols = self.df.select_dtypes(include=['object']).columns
        date_cols = [col for col in date_cols if 'date' in col.lower()]
        
        for col in date_cols:
            try:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                print(f" Converted '{col}' to datetime")
            except Exception as e:
                print(f" Could not convert '{col}' to datetime: {str(e)}")
        
        # 3. Clean numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Remove negative values if they don't make sense (e.g., Amount)
            if 'amount' in col.lower() or 'balance' in col.lower():
                neg_count = (self.df[col] < 0).sum()
                if neg_count > 0:
                    print(f" Found {neg_count} negative values in '{col}'")
        
        # 4. Handle missing values intelligently
        print("\n Handling missing values...")
        
        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            if missing_count > 0:
                missing_pct = (missing_count / len(self.df)) * 100
                
                # Drop column if >50% missing
                if missing_pct > 50:
                    self.df = self.df.drop(columns=[col])
                    print(f" Dropped '{col}' ({missing_pct:.1f}% missing)")
                
                # Fill numeric columns with median
                elif self.df[col].dtype in ['int64', 'float64']:
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                    print(f" Filled '{col}' with median ({missing_count} values)")
                
                # Fill categorical with mode
                elif self.df[col].dtype == 'object':
                    mode_val = self.df[col].mode()
                    if len(mode_val) > 0:
                        self.df[col] = self.df[col].fillna(mode_val[0])
                        print(f" Filled '{col}' with mode ({missing_count} values)")
        
        self.cleaning_report['final_shape'] = self.df.shape
        print(f"\n Cleaning complete! Final shape: {self.df.shape}")
        
    def detect_outliers(self, columns=None, method='iqr', remove=False):
        """Detect and optionally remove outliers"""
        print("\n OUTLIER DETECTION")
        print("="*60)
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        outlier_summary = {}
        
        for col in columns:
            if col not in self.df.columns:
                continue
                
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                
                outliers = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
                outlier_pct = (outliers / len(self.df)) * 100
                
                outlier_summary[col] = {
                    'count': outliers,
                    'percent': outlier_pct,
                    'lower_bound': lower,
                    'upper_bound': upper
                }
                
                print(f"{col}:")
                print(f"  Outliers: {outliers} ({outlier_pct:.2f}%)")
                print(f"  Valid range: [{lower:.2f}, {upper:.2f}]")
                
                if remove and outliers > 0:
                    before = len(self.df)
                    self.df = self.df[(self.df[col] >= lower) & (self.df[col] <= upper)]
                    print(f"  Removed {before - len(self.df)} outliers")
        
        self.cleaning_report['outliers'] = outlier_summary
        return outlier_summary
    
    def visualize_data(self):
        """Create comprehensive visualizations"""
        print("\n GENERATING VISUALIZATIONS")
        print("="*60)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            print(" No numeric columns to visualize")
            return
        
        # 1. Distribution plots
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for idx, col in enumerate(numeric_cols):
            if idx < len(axes):
                axes[idx].hist(self.df[col].dropna(), bins=30, 
                             edgecolor='black', alpha=0.7)
                axes[idx].set_title(f'Distribution of {col}')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frequency')
        
        # Hide empty subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.file_path.parent / 'distributions.png', dpi=300, bbox_inches='tight')
        print(" Saved: distributions.png")
        plt.show()
        
        # 2. Correlation heatmap
        if len(numeric_cols) > 1:
            plt.figure(figsize=(12, 8))
            corr = self.df[numeric_cols].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            
            sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', 
                       cmap='coolwarm', center=0, square=True,
                       linewidths=1, cbar_kws={"shrink": 0.8})
            plt.title('Correlation Matrix', fontsize=16, pad=20)
            plt.tight_layout()
            plt.savefig(self.file_path.parent / 'correlation.png', dpi=300, bbox_inches='tight')
            print("✓ Saved: correlation.png")
            plt.show()
        
        # 3. Box plots for outlier visualization
        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(1, min(4, len(numeric_cols)), 
                                    figsize=(15, 5))
            if len(numeric_cols) == 1:
                axes = [axes]
            
            for idx, col in enumerate(list(numeric_cols)[:4]):
                axes[idx].boxplot(self.df[col].dropna())
                axes[idx].set_title(f'{col}')
                axes[idx].set_ylabel('Value')
            
            plt.tight_layout()
            plt.savefig(self.file_path.parent / 'boxplots.png', dpi=300, bbox_inches='tight')
            print(" Saved: boxplots.png")
            plt.show()
    
    def generate_report(self):
        """Generate cleaning report"""
        print("\n CLEANING REPORT")
        print("="*60)
        print(f"Original shape: {self.cleaning_report.get('original_shape', 'N/A')}")
        print(f"Final shape: {self.cleaning_report.get('final_shape', 'N/A')}")
        print(f"Rows removed: {self.cleaning_report['original_shape'][0] - self.cleaning_report['final_shape'][0]:,}")
        print(f"Duplicates removed: {self.cleaning_report.get('duplicates_removed', 0):,}")
        print(f"Initial missing values: {self.cleaning_report.get('initial_missing', 0):,}")
        print(f"Final missing values: {self.df.isnull().sum().sum():,}")
        print("="*60)
    
    def save_cleaned_data(self, output_name='cleaned_data.csv'):
        """Save cleaned dataset"""
        output_path = self.file_path.parent / output_name
        self.df.to_csv(output_path, index=False)
        print(f"\n Cleaned dataset saved: {output_path}")
        return output_path


# ==============================
# MAIN EXECUTION
# ==============================
if __name__ == "__main__":
    # Update this path to your file location
    FILE_PATH = r"C:\Users\ASUS\Downloads\archive\budgetwise_finance_dataset.csv"
    
    # Initialize processor
    processor = FinanceDataProcessor(FILE_PATH)
    
    # Load data
    if processor.load_data():
        # Explore data
        processor.explore_data()
        
        # Clean data
        processor.clean_data()
        
        # Detect outliers (set remove=True to remove them)
        processor.detect_outliers(remove=False)
        
        # Visualize
        processor.visualize_data()
        
        # Generate report
        processor.generate_report()
        
        # Save cleaned data
        processor.save_cleaned_data('budgetwise_finance_cleaned.csv')
        
        print("\n ALL OPERATIONS COMPLETED SUCCESSFULLY!")