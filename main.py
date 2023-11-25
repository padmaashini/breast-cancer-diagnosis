import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

class KNN: 
    def __init__(self, k):
        self.k = k

    def fit(self, train_features, train_labels):
        self.train_features = train_features
        self.train_labels = train_labels
    
    def predict_multiple_points(self, points):
        labels = [self.predict_single_point(x) for x in points]
        return labels
    
    def predict_single_point(self, x):
        # Calculate the distance between x and all the datapoints in the train dataset
        distances = [np.sqrt(np.sum((trained - x)**2)) for trained in self.train_features]

        # Constraint - only classify based on certain radius


        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.train_labels[i] for i in k_indices]
        
        # Return the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

def setup():
    dataset = datasets.load_breast_cancer()

    # Write the dataset to an Excel file
    df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
    df['target'] = dataset.target
    
    excel_file_path = "breast_cancer_dataset.xlsx"
    df.to_excel(excel_file_path, index=False)

    # Standardize the features of the dataset
    features = StandardScaler().fit_transform(dataset.data)
    labels = dataset.target

    num_features = features.shape[1] # outputs 30

    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.3, stratify=labels, random_state=42
    )

    return train_features, test_features, train_labels, test_labels

def plot_single_prediction(train_features, test_features, train_labels, prediction): 
    # Apply PCA to reduce dimensions to 2 for visualization purposes
    pca = PCA(n_components=2)

    # Reduce dimensionality --> transforms data into a new coordinate system
    train_features_reduced_dim = pca.fit_transform(train_features)
    test_features_reduced_dim = pca.transform(test_features)

    # Plotting
    plt.figure(figsize=(10, 6))
    colors = ['navy', 'turquoise']
    lw = 2

    for color, i in zip(colors, [0, 1]):
        plt.scatter(train_features_reduced_dim[train_labels == i, 0], train_features_reduced_dim[train_labels == i, 1], color=color, alpha=.8, lw=lw,
                    label=f'Class {i}')

    # Highlighting the test point 'x' in a different color
    test_point_x = test_features_reduced_dim[0][0]
    test_point_y = test_features_reduced_dim[0][1]
    plt.scatter(test_point_x, test_point_y, color='red', edgecolor='k', lw=lw, label='Test Point X', s=100)

    prediction_label = "Malignant" if prediction == 1 else "Benign"
    plt.text(test_point_x, test_point_y, f'{prediction_label} - {prediction}', color='black', fontsize=12)

    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of Breast Cancer dataset with Test Point X')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

def main():
    train_features, test_features, train_labels, test_labels = setup()

    # Parameters to change: k
    k = 20
    datapoint_to_test = test_features[0]

    classifier = KNN(k)
    classifier.fit(train_features, train_labels)

    # The Wisconsin dataset labels benign as 0 and 1 as malignant cancer 
    prediction = classifier.predict_single_point(datapoint_to_test)
    print(f"Classification for point {datapoint_to_test}: {prediction}")

    plot_single_prediction(train_features, test_features, train_labels, prediction)
    
    

    

if __name__ == '__main__':
    main()
