import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import tensorflowjs as tfjs
import os
from math import log
from tqdm import tqdm

EPOCHS = 50

df = pd.read_csv("adult_subset.csv")
df.columns = df.columns.str.strip()
feature_cols = ['age', 'education', 'education-num', 'relationship', 'race', 'gender', 'hours-per-week']
target_col = 'income'

def create_feature_mappings(df, feature_cols):
    feature_maps = {}
    for col in feature_cols:
        unique_values = sorted(df[col].unique())
        feature_maps[col] = {val: idx for idx, val in enumerate(unique_values)}
    return feature_maps

def one_hot_encode_row(row, feature_maps):
    encoded_features = []
    for col, value_map in feature_maps.items():
        one_hot = [0] * len(value_map)
        one_hot[value_map[row[col]]] = 1
        encoded_features.extend(one_hot)
    return encoded_features

def create_feature_metadata(feature_maps):
    metadata = {
        "features": {},
        "total_features": 0
    }
    current_position = 0
    for feature, value_map in feature_maps.items():
        metadata["features"][feature] = {
            "start_index": current_position,
            "length": len(value_map),
            "mapping": {str(k): v for k, v in value_map.items()}
        }
        current_position += len(value_map)
    metadata["total_features"] = current_position
    return metadata

def encode_data(df, feature_cols):
    feature_maps = create_feature_mappings(df, feature_cols)
    X = np.array([one_hot_encode_row(row, feature_maps) for _, row in df.iterrows()])
    target_map = {val: idx for idx, val in enumerate(sorted(df[target_col].unique()))}
    y = np.array([target_map[val] for val in df[target_col]])
    metadata = create_feature_metadata(feature_maps)
    with open('feature_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    data_for_js = {
        "X": X.tolist(),
        "y": y.tolist(),
        "target_mapping": {str(k): v for k, v in target_map.items()}
    }
    with open('data.json', 'w') as f:
        json.dump(data_for_js, f)

def complete_model(path="data.json"):
    with open(path, "r") as f:
        d = json.load(f)
    X = np.array(d['X'])
    y = np.array(d['y'])
    print("Original training with:", X.shape[0], "samples")
    model = keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=EPOCHS, batch_size=32)
    tfjs_target_dir = 'tfjs_model'
    tfjs.converters.save_keras_model(model, tfjs_target_dir)
    print(f"Model saved for TensorFlow.js in directory: {tfjs_target_dir}/")
    y_pred = model.predict(X).flatten()
    return y_pred, y

def compute_base_performance(y_pred, y_true, df, feature_cols):
    base_performance_histogram = {}
    for col in feature_cols:
        base_performance_histogram[col] = {}
        unique_attributes = sorted(list(set(df[col].tolist())))
        for attribute in unique_attributes:
            indices = df[df[col].isin([attribute])].index.tolist()
            y_subset_pred = [i for idx,i in enumerate(y_pred) if idx in indices]
            y_subset_true = [i for idx,i in enumerate(y_true) if idx in indices]
            acc = sum(round(pred) == true for pred, true in zip(y_subset_pred, y_subset_true)) / len(y_subset_true)
            loss = -sum(true * log(pred) + (1-true) * log(1-pred) for pred, true in zip(y_subset_pred, y_subset_true)) / len(y_subset_true)
            base_performance_histogram[col][attribute] = {"loss": loss, "acc": acc}
    with open("base_performance_histogram.json", "w") as f:
        json.dump(base_performance_histogram, f, indent=4)

def retrain_without_indices(index_to_forget, path="data.json"):
    with open(path, "r") as f:
        d = json.load(f)
    d['X'] = [row for idx,row in enumerate(d['X']) if idx!=index_to_forget]
    d['y'] = [row for idx,row in enumerate(d['y']) if idx!=index_to_forget]
    X = np.array(d['X'])
    y = np.array(d['y'])
    print("Re-training with:", X.shape[0], "samples")
    model = keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=EPOCHS, batch_size=32)
    y_pred = model.predict(X).flatten()
    return y_pred, y

def compute_retrained_performance(y_pred, y_true, index_to_forget, df_og, feature_cols):
    df = df_og.copy(deep=True)
    df = df.drop(index=df.index[index_to_forget])
    performance_histogram = {}
    for col in feature_cols:
        performance_histogram[col] = {}
        unique_attributes = sorted(list(set(df[col].tolist())))
        for attribute in unique_attributes:
            indices = df[df[col].isin([attribute])].index.tolist()
            y_subset_pred = [i for idx,i in enumerate(y_pred) if idx in indices]
            y_subset_true = [i for idx,i in enumerate(y_true) if idx in indices]
            acc = sum(round(pred) == true for pred, true in zip(y_subset_pred, y_subset_true)) / len(y_subset_true)
            loss = -sum(true * log(pred) + (1-true) * log(1-pred) for pred, true in zip(y_subset_pred, y_subset_true)) / len(y_subset_true)
            performance_histogram[col][attribute] = {"loss": loss, "acc": acc}
    return performance_histogram

encode_data(df, feature_cols)
y_pred, y = complete_model()
compute_base_performance(y_pred=y_pred, y_true=y, df=df, feature_cols=feature_cols)
with open("marginalized_info.json", "r") as f:
    marginalized_indices = json.load(f)['common_indices']
for index_to_forget in tqdm(marginalized_indices, desc="re-traininig all"):
    y_pred, y_true = retrain_without_indices(index_to_forget)
    performance_histogram = compute_retrained_performance(y_pred=y_pred, y_true=y, index_to_forget=index_to_forget, df_og=df, feature_cols=feature_cols)
    os.makedirs("retrained", exist_ok=True)
    with open(os.path.join("retrained", str(index_to_forget)+".json"), "w") as f:
        json.dump(performance_histogram, f, indent=4)