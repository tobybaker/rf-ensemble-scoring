import numpy as np
from sklearn.ensemble import RandomForestClassifier
import foreststats


# class to store training and test data along with weights
class Dataset:
    def __init__(self,  data, labels, test_percentage, shuffle_indicies, weights=None):
        data = data[shuffle_indicies]
        labels = labels[shuffle_indicies]
        sample_split = int(data.shape[0]*test_percentage)

        self.test_data = data[:sample_split]
        self.test_labels = labels[:sample_split]

        self.training_data = data[sample_split:]
        self.trainingLabels = labels[sample_split:]

        if weights is not None:
            weights = weights[shuffle_indicies]
            self.test_weights = weights[:sample_split]
            self.training_weights = weights[sample_split:]
        else:
            self.test_weights = None
            self.training_weights = None


# sequentially label the data with an optional 50% offset
def get_labels(length, run_length, half_offset):
    start_range = np.arange(length)
    if half_offset:
        start_range = start_range + run_length/2
    binned = np.floor(start_range/run_length)
    return binned.astype(int)


# compute probabilies for class
def forest_probabilities(dataset, n_jobs=1, min_samples_leaf=100):
    forest = RandomForestClassifier(n_estimators=20, n_jobs=n_jobs, min_samples_leaf=min_samples_leaf)
    forest.fit(dataset.training_data, dataset.trainingLabels, sample_weight=dataset.training_weights)
    probabilities = forest.predict_proba(dataset.test_data)
    return probabilities,  forest


# score the forest
def forest_score(ca_distances, run_length, test_split, weights=None, n_jobs=1, min_samples_leaf=100):
    # get sequential offsetted and non-offsetted labels
    labels = get_labels(ca_distances.shape[0], run_length, False)
    offsetLabels = get_labels(ca_distances.shape[0], run_length, True)

    # provide a set of indicies to randomly shuffle the data between test and trained
    # but ensure the shuffling is the same for the data with offsetted labels
    indicies = np.array(range(0, ca_distances.shape[0]))
    np.random.shuffle(indicies)

    dataset = Dataset(ca_distances, labels, 0.2, indicies, weights=weights)
    offset_dataset = Dataset(ca_distances, offsetLabels, 0.2, indicies, weights=weights)

    probabilities,  offset_forest = forest_probabilities(dataset, n_jobs, min_samples_leaf)
    offset_probabilities, offsetForest = forest_probabilities(offset_dataset, n_jobs, min_samples_leaf)

    probability_scores = []
    for i in range(probabilities.shape[0]):
        # find probability assigned to the correct label
        p1 = probabilities[i, int(dataset.test_labels[i])]
        p2 = offset_probabilities[i, int(offset_dataset.test_labels[i])]

        # max probability of the two labelling regimes is used
        probability_scores.append(max(p1, p1))

    # if weights exist use the weighted average
    score = np.average(probability_scores, weights=dataset.test_weights)

    return score

