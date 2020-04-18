import json
from math import log 


def import_dataset(filename):
    
    dataset = []
    features = []
    
    f = open(filename, 'r')
    labels = f.readline()
    labels = labels.split(',')
    for word in labels:
        word = word.strip()
        word = word.strip('()')
        features.append(word)
    features = features[:-1]
    f = f.readlines()[1:]
    for row in f:
        row = row[4:]
        row = row.strip('\n;:')
        row = row.split(',')
        row1 = []
        for word in row:
            word = word.strip()
            row1.append(word)
        dataset.append(row1)
        
    return (dataset, features)

    
def calculate_entropy(dataset):
    
    label_data = []
    for row in dataset:
        label_data.append(row[-1])
    entropy = 0
    numInstance = len(label_data)
    counts = {}
    for item in label_data:
        counts[item] = counts.get(item, 0) + 1
    for key in counts:
        p = counts[key] / numInstance
        entropy = entropy - p * log(p, 2)
    
    return entropy  


def dataset_to_subset(dataset,feature_index, value):
    
    subset = []
    for row in dataset:
        if row[feature_index] == value:
            new_row = row[:feature_index] + row[feature_index+1:]
            subset.append(new_row)
            
    return subset


def select_feature(dataset):
    
    entropy = calculate_entropy(dataset)
    IG = 0
    selected_feature_index = -1
    for feature_index in range(len(dataset[0]) - 1):
        feature_value = [row[feature_index] for row in dataset]
        unique_feature_value = set(feature_value)
        feature_entropy = 0
        for value in unique_feature_value:
            subset = dataset_to_subset(dataset,feature_index, value)
            subset_entropy = calculate_entropy(subset)
            p = len(subset) / len(dataset)
            feature_entropy = feature_entropy + p * subset_entropy
        feature_IG = entropy - feature_entropy 
        if feature_IG > IG:
            IG = feature_IG
            selected_feature_index = feature_index 
    
    return (selected_feature_index, IG)  


def build_decision_tree(dataset, features):
    
    label_value = [row[-1] for row in dataset]
    if len(set(label_value)) == 1:    
        return label_value[0]  
    selected_feature_index = select_feature(dataset)[0]
    select_feature_IG = select_feature(dataset)[1]
    if select_feature_IG == 0:
        counts = {}
        for item in label_value:
            counts[item] = counts.get(item, 0) + 1
        mode = max(counts, key = counts.get)
        return mode
    else:
        features1 = features[:]
        selected_feature = features[selected_feature_index]
        decision_tree = {}
        decision_tree[selected_feature] = {}
        features1.remove(selected_feature)
        feature_value = [row[selected_feature_index] for row in dataset]
        unique_feature_value = set(feature_value)
        for value in unique_feature_value:
            subset = dataset_to_subset(dataset,selected_feature_index, value)
            sub_features = features1[:]
            decision_tree[selected_feature][value] = build_decision_tree(subset, sub_features)

        return decision_tree 

        
def predict(tree, feature_labels, test):
    
    best_feature = list(tree)[0]
    best_feature_value_tree = tree[best_feature]
    feature_index = feature_labels.index(best_feature)
    test_value = test[feature_index]
    value_tree = best_feature_value_tree[test_value]
    if isinstance(value_tree, dict): 
        outcome = predict(value_tree, feature_labels, test)
    else: 
        outcome = value_tree
        
    return outcome


result = import_dataset('dt_data.txt')
dataset = result[0]
features = result[1]
tree = build_decision_tree(dataset, features)
print('Tree: \n{}'.format(json.dumps(tree, indent=4)))
# Make prediction
outcome = predict(tree, features, ["Moderate", "Cheap", "Loud", "City-Center", "No", "No"])
print('The prediction result is {}.'.format(outcome))

