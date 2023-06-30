import numpy as np
from scipy.spatial import distance

from sklearn.base import BaseEstimator

def euclidian_distances_not_vectorized(test, train):
    """
    Compute Euclidean distances between a test sample and all training samples
    (not vectorized version)

    Parameters:
    - test: The test sample
    - train: Array of training samples

    Returns:
    - distances: List of Euclidean distances
    """
    distances = []
    for row in train:
        d = distance.euclidean(row, test)
        distances.append(d)
    return distances

def euclidian_distances(test, train):
    """
    Compute Euclidean distances between a test sample and all training samples
    (vectorized version)

    Parameters:
    - test: The test sample
    - train: Array of training samples

    Returns:
    - distances: Array of Euclidean distances
    """
    return np.sqrt(np.sum((test - train) ** 2, axis=1))

def chebyshev_distances_not_vectorized(test, train):
    """
    Compute Chebyshev distances between a test sample and all training samples
    (not vectorized version)

    Parameters:
    - test: The test sample
    - train: Array of training samples

    Returns:
    - distances: List of Chebyshev distances
    """
    distances = []
    for row in train:
        d = distance.chebyshev(row, test)
        distances.append(d)
    return distances

def chebyshev_distances(test, train):
    """
    Compute Chebyshev distances between a test sample and all training samples
    (vectorized version)

    Parameters:
    - test: The test sample
    - train: Array of training samples

    Returns:
    - distances: Array of Chebyshev distances
    """
    return np.max(np.abs(train - test), axis=1)

def cosine_distances_not_vectorized(test, train):
    """
    Compute cosine distances between a test sample and all training samples
    (not vectorized version)

    Parameters:
    - test: The test sample
    - train: Array of training samples

    Returns:
    - distances: List of cosine distances
    """
    distances = []
    for row in train:
        d = distance.cosine(row, test)
        distances.append(d)
    return distances

def cosine_distances(test, train):
    """
    Compute cosine distances between a test sample and all training samples
    (vectorized version)

    Parameters:
    - test: The test sample
    - train: Array of training samples

    Returns:
    - distances: Array of cosine distances
    """
    t = np.repeat(test.reshape(1, -1), train.shape[0], axis=0)
    return 1.0 - np.divide(np.sum(t * train, axis=1), (np.linalg.norm(test) * np.linalg.norm(train, axis=1)))


class KNN(BaseEstimator):
    """
    KNN algorithm implementation based on https://towardsdatascience.com/k-nearest-neighbors-classification-from-scratch-with-numpy-cb222ecfeac1
    """
    def __init__(self, n_neighbors=5, weights='uniform', metric='euclidean', n_classes=10):
        """
        Initialize the KNN classifier with specified parameters

        Parameters:
        - n_neighbors: Number of neighbors to consider
        - weights: Weighting scheme for prediction ('uniform' or 'distance')
        - metric: Distance metric to use ('euclidean', 'chebyshev', or 'cosine')
        - n_classes: Number of classes in the classification problem
        """
        self.n_neighbors = n_neighbors
        self.weights = weights

        self.metric = metric

        if self.metric == 'euclidean':
            self.distance_fn = euclidian_distances
        elif self.metric == 'chebyshev':
            self.distance_fn = chebyshev_distances
        elif self.metric == 'cosine':
            self.distance_fn = cosine_distances
        else:
            raise AttributeError("distance_fn is not initialized")
        
        self.n_classes = n_classes

    def fit(self, X_train, y_train):
        """
        Save the training data and labels

        Parameters:
        - X_train: Array of training samples
        - y_train: Array of training labels
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test, return_distance=False):
        """
        Return array of indices of n_neighbors and their distances to X_test

        Parameters:
        - X_test: Array of test samples
        - return_distance: Whether to return distances along with indices

        Returns:
        - dist: Array of distances to nearest neighbors
        - neigh_ind: Array of indices of nearest neighbors
        """
        dist = []
        neigh_ind = []

        point_dist = [self.distance_fn(x_test, self.X_train) for x_test in X_test]

        for row in point_dist:
            enum_neigh = enumerate(row)
            sorted_neigh = sorted(enum_neigh, key=lambda x: x[1])[:self.n_neighbors]

            ind_list = [tup[0] for tup in sorted_neigh]
            dist_list = [tup[1] for tup in sorted_neigh]

            dist.append(dist_list)
            neigh_ind.append(ind_list)

        if return_distance:
            return np.array(dist), np.array(neigh_ind)

        return np.array(neigh_ind)

    def predict(self, X_test):
        """
        Make predictions for the test data

        Parameters:
        - X_test: Array of test samples

        Returns:
        - y_pred: Array of predicted labels
        """
        if self.weights == 'uniform':
            neighbors = self.kneighbors(X_test)
            y_pred = np.array([
                np.argmax(np.bincount(self.y_train[neighbor])) for neighbor in neighbors
            ])

            return y_pred

        if self.weights == 'distance':
            dist, neigh_ind = self.kneighbors(X_test, return_distance=True)

            inv_dist = 1 / dist

            mean_inv_dist = inv_dist / np.sum(inv_dist, axis=1)[:, np.newaxis]

            proba = []

            for i, row in enumerate(mean_inv_dist):
                row_pred = self.y_train[neigh_ind[i]]

                for k in range(self.n_classes):
                    indices = np.where(row_pred == k)
                    prob_ind = np.sum(row[indices])
                    proba.append(np.array(prob_ind))

            predict_proba = np.array(proba).reshape(X_test.shape[0], self.n_classes)

            y_pred = np.array([np.argmax(item) for item in predict_proba])

            return y_pred

    def score(self, X_test, y_test):
        """
        Compute the accuracy of the classifier on the test data

        Parameters:
        - X_test: Array of test samples
        - y_test: Array of true labels for the test samples

        Returns:
        - accuracy: Accuracy of the classifier
        """
        y_pred = self.predict(X_test)
        accuracy = float(sum(y_pred == y_test)) / float(len(y_test))
        return accuracy
    

if __name__ == '__main__':
    from sklearn.datasets import make_classification

    # Generate a synthetic classification dataset
    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2,
                                n_clusters_per_class=1, n_classes=3, random_state=21)

    # Standardize the features
    mu = np.mean(X, 0)
    sigma = np.std(X, 0)
    X = (X - mu) / sigma

    # Combine the features and labels
    data = np.hstack((X, y[:, np.newaxis]))
        
    np.random.seed(21)
    np.random.shuffle(data)

    split_rate = 0.7

    # Split the data into training and test sets
    train, test = np.split(data, [int(split_rate * (data.shape[0]))])

    X_train = train[:, :-1]
    y_train = train[:, -1]

    X_test = test[:, :-1]
    y_test = test[:, -1]

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # Create an instance of the KNN classifier
    our_classifier = KNN(n_neighbors=3, weights='distance', metric='cosine')

    # Train the classifier
    our_classifier.fit(X_train, y_train)

    # Evaluate the classifier on the test data
    our_accuracy = our_classifier.score(X_test, y_test)

    print(our_accuracy)