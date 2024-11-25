import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

def load_data(filepath):
    df = pd.read_csv(filepath)
    print("Initial Data Loaded:")
    print(df.head())
    print("\nData Info:")
    print(df.info())
    print("\nMissing Values per Column:")
    print(df.isnull().sum())
    return df

def rename_columns(df):
    df = df.rename(columns={
        'sex': 'gender',
        'marital-status': 'marital_status',
        'native-country': 'native_country'
    })
    print("\nColumns after renaming:")
    print(df.columns.tolist())
    return df

def preprocess_data(df, categorical_to_keep, target_column):
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)
    categorical_encode = [col for col in categorical_cols if col not in categorical_to_keep]
    categorical_keep = [col for col in categorical_cols if col in categorical_to_keep]
    print("\nCategorical Columns to Encode:", categorical_encode)
    print("Categorical Columns to Keep (Original):", categorical_keep)
    for col in categorical_encode:
        df[col] = le.fit_transform(df[col])
        print(f"Encoded '{col}'")
    df[target_column] = le.fit_transform(df[target_column])
    print(f"Encoded target variable '{target_column}'")
    return df, categorical_encode, categorical_keep

def compute_correlation(df, target_column, threshold=0.8, categorical_keep=[]):
    cols_to_drop = [target_column] + categorical_keep
    corr_matrix = df.drop(columns=cols_to_drop).corr()
    print("\nCorrelation Matrix:")
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Correlation Matrix of Numerical Features")
    plt.savefig('corr.png')
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                high_corr_pairs.append((col1, col2, corr_value))
    print(f"\nHighly Correlated Pairs (|correlation| > {threshold}):")
    for pair in high_corr_pairs:
        print(f"{pair[0]} and {pair[1]}: {pair[2]:.2f}")
    return corr_matrix, high_corr_pairs

def select_features(df, categorical_keep, initial_threshold=0.8, desired_feature_count=6):
    corr_matrix, high_corr_pairs = compute_correlation(df, target_column='income', threshold=initial_threshold, categorical_keep=categorical_keep)
    features_to_drop = set()
    for pair in high_corr_pairs:
        col1, col2, corr = pair
        if col1 in categorical_keep and col2 not in categorical_keep:
            features_to_drop.add(col2)
            print(f"Dropping '{col2}' due to high correlation with kept categorical '{col1}'")
        elif col2 in categorical_keep and col1 not in categorical_keep:
            features_to_drop.add(col1)
            print(f"Dropping '{col1}' due to high correlation with kept categorical '{col2}'")
        else:
            avg_corr_col1 = corr_matrix[col1].abs().mean()
            avg_corr_col2 = corr_matrix[col2].abs().mean()
            if avg_corr_col1 > avg_corr_col2:
                features_to_drop.add(col1)
                print(f"Dropping '{col1}' due to higher average correlation ({avg_corr_col1:.2f})")
            else:
                features_to_drop.add(col2)
                print(f"Dropping '{col2}' due to higher average correlation ({avg_corr_col2:.2f})")
    print(f"\nFeatures to Drop: {features_to_drop}")
    df_reduced = df.drop(columns=list(features_to_drop))
    print(f"\nFeatures after dropping highly correlated ones: {df_reduced.columns.tolist()}")
    numerical_features = [col for col in df_reduced.columns if col not in categorical_keep + ['income']]
    current_numerical_count = len(numerical_features)
    numerical_desired = desired_feature_count - len(categorical_keep)
    if current_numerical_count > numerical_desired:
        target_corr = df_reduced[numerical_features + ['income']].corr()['income'].abs().sort_values(ascending=False)
        selected_numerical = target_corr.drop('income').head(numerical_desired).index.tolist()
        features_to_drop_additional = [col for col in numerical_features if col not in selected_numerical]
        df_reduced = df_reduced.drop(columns=features_to_drop_additional)
        print(f"\nDropping additional numerical features to reach desired count: {features_to_drop_additional}")
        print(f"Selected Numerical Features: {selected_numerical}")
    return df_reduced

def finalize_dataset(df_reduced, original_df, categorical_keep, target_column):
    categorical_in_reduced = [col for col in categorical_keep if col in df_reduced.columns]
    print(f"\nFinal Reduced Dataset Shape: {df_reduced.shape}")
    print("\nFinal Reduced Dataset Preview:")
    print(df_reduced.head())
    return df_reduced

def save_reduced_data(df, output_filepath):
    df.to_csv(output_filepath, index=False)
    print(f"\nReduced dataset saved to '{output_filepath}'.")

def main():
    filepath = "adult.csv"
    output_filepath = "adult_reduced.csv" 
    target_column = "income" 
    categorical_to_keep = ['education', 'relationship', 'race', 'gender'] 
    desired_feature_count = 8
    df = load_data(filepath)
    df = rename_columns(df)
    df_encoded, categorical_encode, categorical_keep = preprocess_data(df, categorical_to_keep, target_column)
    df_selected = select_features(df_encoded, categorical_keep, initial_threshold=0.8, desired_feature_count=desired_feature_count)
    df_final = finalize_dataset(df_selected, df, categorical_keep, target_column)
    save_reduced_data(df_final, output_filepath)

if __name__ == "__main__":
    main()
