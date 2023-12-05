import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
import concurrent.futures
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
    
    def predict_multiple_points(self, points, search_radius):
        labels = [self.predict_single_point(x, search_radius) for x in points]
        return labels
    
    def predict_single_point(self, x, search_radius):
        # Calculate the distance between x and all the datapoints in the train dataset
        distances = [np.sqrt(np.sum((trained - x)**2)) for trained in self.train_features]

        # print('distances', sum(distances)/len(distances))
        # Constraint - only classify based on certain radius
        # Some notes for us: ideally I wanted to do a distance of 6, but it didn't work due to outliers
        distances = list(filter(lambda d: d <= search_radius, distances))

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

def evaluate_knn(k, search_radius, train_features, train_labels, test_features, test_labels):
    classifier = KNN(k)
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict_multiple_points(test_features, search_radius)
    tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
    accuracy = (tp + tn) / len(test_features)
    return accuracy, k, search_radius

def main():
    train_features, test_features, train_labels, test_labels = setup()
    results = {}

    # Use ProcessPoolExecutor to run the tasks in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        
        k_end = len(train_features) # 398
        for k in range(1, k_end + 1, 25):
            for search_radius in range(15, 50):
                # Submit each task to be executed in a separate process
                futures.append(executor.submit(evaluate_knn, k, search_radius, train_features, train_labels, test_features, test_labels))

        for future in concurrent.futures.as_completed(futures):
            accuracy, k, search_radius = future.result()
            print('accuracy', accuracy)
            results[accuracy] = (k, search_radius)

    max_accuracy = max(results.keys())
    best_k, best_search_radius = results[max_accuracy]
    print(f"{max_accuracy}: k - {best_k}, search_radius - {best_search_radius}")

if __name__ == '__main__':
    main()
