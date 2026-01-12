import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Loading the Healthcare Dataset
# Week 1 focuses on data acquisition. 
# We'll use a standard diagnostic dataset (e.g., Diabetes or Heart Disease).
def load_and_inspect(diabetes.csv):
    df = pd.read_csv(diabetes.csv)
    print("--- Dataset Shape ---")
    print(f"Total Patients: {df.shape[0]}, Total Features: {df.shape[1]}")
    
    # Checking for missing values (Healthcare challenge: Data gaps)
    print("\n--- Missing Data (Diagnostic Gaps) ---")
    print(df.isnull().sum())
    return df

# 2. Pattern Discovery (Phase 2: EDA)
def discover_patterns(df):
    # Pattern A: Finding High-Risk Age Groups
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x='Age', hue='Outcome', kde=True, element="step")
    plt.title("Disease Distribution Across Age Groups")
    plt.xlabel("Age of Patient")
    plt.ylabel("Frequency")
    plt.show()

    # Pattern B: Correlation Analysis (Key Diagnostic Indicators)
    # This helps identify which parameters (like Glucose or BMI) are most linked to disease
    plt.figure(figsize=(12, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='RdYlGn', linewidths=0.2)
    plt.title("Correlation Between Medical Parameters")
    plt.show()

    # Pattern C: Gender or Feature-based Patterns
    # For example, how BMI relates to Glucose levels
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Glucose', y='BMI', hue='Outcome', data=df)
    plt.title("Glucose vs BMI: Risk Segmentation")
    plt.show()

# To execute:
# df = load_and_inspect('healthcare_data.csv')
# discover_patterns(df)