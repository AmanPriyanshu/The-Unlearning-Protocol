import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import sys

def create_images_directory(directory_name='images'):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
        print(f"Created directory: {directory_name}")
    else:
        print(f"Directory already exists: {directory_name}")
    return directory_name

def read_and_clean_data(filepath):
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded data from '{filepath}'.")
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{filepath}' is empty.")
        sys.exit(1)
    except pd.errors.ParserError:
        print(f"Error: The file '{filepath}' does not appear to be in CSV format.")
        sys.exit(1)
    initial_shape = df.shape
    df.dropna(inplace=True)
    final_shape = df.shape
    print(f"Removed {initial_shape[0] - final_shape[0]} rows containing NaN values.")
    print(f"Data shape after cleaning: {df.shape}")
    return df

def plot_numerical_column(df, column, directory):
    plt.figure(figsize=(8,6))
    plt.hist(df[column], bins=10, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.tight_layout()
    image_path = os.path.join(directory, f"{column}.png")
    plt.savefig(image_path)
    plt.close()
    print(f"Saved histogram for numerical column '{column}' as '{image_path}'.")

def plot_categorical_column(df, column, directory):
    plt.figure(figsize=(10,6))
    value_counts = df[column].value_counts()
    value_counts.plot(kind='bar', color='lightgreen', edgecolor='black')
    plt.title(f'Bar Chart of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    image_path = os.path.join(directory, f"{column}.png")
    plt.savefig(image_path)
    plt.close()
    print(f"Saved bar chart for categorical column '{column}' as '{image_path}'.")

def generate_histograms(df, directory):
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            plot_numerical_column(df, column, directory)
        else:
            plot_categorical_column(df, column, directory)

def main():
    np.random.seed(42)
    input_filepath = 'adult_reduced.csv'
    df = read_and_clean_data(input_filepath)
    df = df.sample(frac=0.2)
    df.to_csv("adult_subset.csv", index=False)
    print("Final df shape",df.shape)
    images_dir = create_images_directory('images')
    generate_histograms(df, images_dir)
    print("\nAll histograms have been generated and saved in the 'images' directory.")

if __name__ == "__main__":
    main()