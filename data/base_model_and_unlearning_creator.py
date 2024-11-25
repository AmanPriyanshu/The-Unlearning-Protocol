import pandas as pd
import json

def identify_majority_values(value_counts, threshold=0.5):
    total = value_counts.sum()
    marginalized_sum = total
    majority_values = []
    
    for value, count in value_counts.items():
        if (marginalized_sum - count) / total < threshold:
            majority_values.append(value)
            marginalized_sum -= count
            break
        else:
            majority_values.append(value)
            marginalized_sum -= count
    for value, count in value_counts.items():
        if value in majority_values:
            continue
        if (marginalized_sum) / total < threshold:
            break
        majority_values.append(value)
        marginalized_sum -= count
    if (marginalized_sum) / total >= threshold:
        for value, count in value_counts.items():
            if value in majority_values:
                continue
            majority_values.append(value)
            marginalized_sum -= count
            if (marginalized_sum) / total < threshold:
                break
    
    marginalized_values = [val for val in value_counts.index if val not in majority_values]
    
    return majority_values, marginalized_values

df = pd.read_csv("adult_subset.csv")
counts_unique = {}
marginalized_info = {}

for column in df.columns:
    value_counts = df[column].value_counts().sort_values(ascending=False)
    counts_unique[column] = value_counts.to_dict()
    majority_values, marginalized_values = identify_majority_values(value_counts, threshold=0.15)
    marginalized_indices = df[df[column].isin(marginalized_values)].index.tolist()
    marginalized_info[column] = {
        "marginalized_groups": marginalized_values,
        "indices": marginalized_indices
    }
    total = value_counts.sum()
    marginalized_count = len(marginalized_indices)
    proportion = marginalized_count / total
    print(f"Column: {column}")
    print(f"  Majority Values: {majority_values}")
    print(f"  Marginalized Groups Count: {marginalized_count} ({proportion:.2%})")
    print("-" * 50)

common_indices = []
for col_name in marginalized_info.keys():
    common_indices.extend(marginalized_info[col_name]['indices'])
common_indices = sorted(list(set(common_indices)))
print("Ratio of common indices for column-wise ANY marginalized groups:", round(100*len(common_indices)/df.shape[0]),'%')
marginalized_info['common_indices'] = common_indices

with open("counts_unique.json", "w") as f:
    json.dump(counts_unique, f, indent=4)
with open("marginalized_info.json", "w") as f:
    json.dump(marginalized_info, f, indent=4)

print("JSON files 'counts_unique.json' and 'marginalized_info.json' have been successfully created.")