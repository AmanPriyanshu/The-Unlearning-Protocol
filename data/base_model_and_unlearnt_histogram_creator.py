import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import json
import tensorflowjs as tfjs

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
    model = keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=20, batch_size=32)
    tfjs_target_dir = 'tfjs_model'
    tfjs.converters.save_keras_model(model, tfjs_target_dir)
    print(f"Model saved for TensorFlow.js in directory: {tfjs_target_dir}/")

encode_data(df, feature_cols)
complete_model()