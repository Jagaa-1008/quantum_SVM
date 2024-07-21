
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import ast
import json

def  plot_figure(clf, X, y, class_num, X_test, X_train, y_test, y_train, filename = None):
    def make_meshgrid(x, y):
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1    
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        return xx, yy

    # Function to plot decision boundaries
    def plot_contours(clf, xx, yy, **params):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, **params)

    # Set up plot figures
    fig, sub = plt.subplots(figsize=(5, 5))

    # Create meshgrid
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    # Define colormap for plotting
    coolwarm = plt.cm.coolwarm
    colors = coolwarm(np.linspace(0, 1, class_num))
    cm = ListedColormap(colors)

    # Plot decision boundaries and data points for clf
    plot_contours(clf, xx, yy, cmap=cm, alpha=0.6)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm, marker='^', edgecolors='k', s=20)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm, marker='o', edgecolors='k', s=20)

    # Compute and plot centroids
    centroids = np.array([X[y == i].mean(axis=0) for i in np.unique(y)])
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=169, linewidths=3, color='g', zorder=10)

    # Predict
    predict = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predict)*100

    # plt.title(f"Decision Boundary Accuracy: {accuracy:.2f}%")

    # Add labels
    labels = ['Test data', 'Train data']
    handles = [plt.Line2D([0], [0], marker='^', color='white', markerfacecolor='k', markersize=10, linestyle=''),
            plt.Line2D([0], [0], marker='o', color='white', markerfacecolor='k', markersize=10, linestyle='')]
    plt.legend(handles, labels)

    if filename:
        plt.savefig(f'{filename}.jpg')

def compute_metrics(SVM,data,t,b,N,validation_pts):
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(N, N+validation_pts):
        predicted_cls = SVM.predict(data[i], b)
        y_i = t[i]
        if(y_i == 1):
            if(predicted_cls > 0):
                tp += 1
            else:
                fp += 1
        else:
            if(predicted_cls < 0):
                tn += 1
            else:
                fn += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = tp/(tp + 1/2*(fp+fn))
    accuracy = (tp + tn)/(tp+tn+fp+fn)

    return precision,recall,f_score,accuracy

def convert_tuple_keys_to_str(obj):
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            new_key = str(k) if isinstance(k, tuple) else k
            new_obj[new_key] = convert_tuple_keys_to_str(v)
        return new_obj
    elif isinstance(obj, list):
        return [convert_tuple_keys_to_str(e) for e in obj]
    else:
        return obj

def save_embedding(embedding, name):
    filename = f"data/{name}_embedding.json"
    # Convert tuple keys to string keys
    converted_embedding = [convert_tuple_keys_to_str(qubo) for qubo in embedding]
    
    # Save to JSON file
    with open(filename, 'w') as f:
        json.dump(converted_embedding, f)

# Function to load the embedding dictionary from a JSON file
def load_embedding(name):
    filename = f"data/{name}_embedding.json"
    with open(filename, 'r') as f:
        embedding = json.load(f)
    return convert_str_to_int(embedding)

def convert_str_to_int(obj):
    return_obj = []
    max_tmp = 0
    for i in range(len(obj)):
        new_obj = {}
        for k, v in obj[i].items():
            new_key = int(k) - max_tmp
            new_obj[new_key] = v
        
        max_tmp += len(new_obj)
        return_obj.append(new_obj)
    return return_obj
    
def save_TotalQubo(TotalQubo, name):    
    # Convert tuple keys to strings
    TotalQubo_str_keys = {str(k): v for k, v in TotalQubo.items()}
    
    filename = f"data/{name}_TotalQubo.json"

    with open(filename, 'w') as f:
        json.dump(TotalQubo_str_keys, f)

def load_TotalQubo(name):
    filename = f"data/{name}_TotalQubo.json"
    
    with open(filename, 'r') as f:
        TotalQubo_str_keys = json.load(f)
    
    # Convert string keys back to tuples
    return {tuple(map(int, k.strip('()').split(', '))): v for k, v in TotalQubo_str_keys.items()}

def save_qubo_list(Qubo_list, name):
    filename = f"data/{name}_Qubo_list.json"
    # Convert tuple keys to string keys
    converted_qubo_list = [convert_tuple_keys_to_str(qubo) for qubo in Qubo_list]
    
    # Save to JSON file
    with open(filename, 'w') as f:
        json.dump(converted_qubo_list, f)

def load_qubo_list(name):
    filename = f"data/{name}_Qubo_list.json"
    # Load from JSON file
    with open(filename, 'r') as f:
        loaded_qubo_list = json.load(f)
    
    # Convert string keys back to tuple keys
    return_dict = []
    for qubo in loaded_qubo_list:
        return_dicts = convert_str_keys_to_tuple(qubo)
        return_dict.append(return_dicts)
    
    return return_dict

def convert_str_keys_to_tuple(obj):
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            new_key = ast.literal_eval(k) if "(" in k and ")" in k else k
            new_obj[new_key] = convert_str_keys_to_tuple(v)
        return new_obj
    elif isinstance(obj, list):
        return [convert_str_keys_to_tuple(e) for e in obj]
    else:
        return obj