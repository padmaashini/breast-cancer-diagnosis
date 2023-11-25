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
from sklearn.neighbors import KNeighborsClassifier

class KNN: 
    def __init__(self, k):
        self.k = k

    def fit(self, train_features, train_labels):
        self.train_features = train_features
        self.train_labels = train_labels
    
    def predict(self, x):
        # Calculate the distance between x and all the datapoints in the train dataset
        distances = [np.sqrt(np.sum(trained - x ) ** 2) for trained in self.train_features]
        print('distances', distances)

        # Constraint - only classify based on certain radius


        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.train_labels[i] for i in k_indices]
        
        # Return the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

def main():
    dataset = datasets.load_breast_cancer()

    # Write the dataset to an Excel file
    df = pd.DataFrame(data=dataset, columns=dataset.feature_names)
    df['target'] = dataset.target
    
    excel_file_path = "breast_cancer_dataset.xlsx"
    df.to_excel(excel_file_path, index=False)

    features = StandardScaler().fit_transform(dataset.data)
    labels = dataset.target

    num_features = features.shape[1]
    print('features: ', features)
    print('labels', labels)
    print('num features', num_features)

    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.3, stratify=labels
    )
    classifier = KNN(k=3)
    classifier.fit(train_features, train_labels)
    prediction = classifier.predict(test_features[0])
    print('prediction', prediction)

    # Apply PCA to reduce dimensions to 2
    pca = PCA(n_components=2)
    X_r = pca.fit_transform(train_features)

    # Initialize and train KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_r, train_labels)

    # Predict using the reduced dataset
    predictions = knn.predict(pca.transform(test_features))

    # Plotting
    plt.figure(figsize=(10, 6))
    colors = ['navy', 'turquoise']
    lw = 2

    for color, i in zip(colors, [0, 1]):
        plt.scatter(X_r[train_labels == i, 0], X_r[train_labels == i, 1], color=color, alpha=.8, lw=lw,
                    label=f'Class {i}')
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of Breast Cancer dataset')
    plt.show()


if __name__ == '__main__':
    main()
