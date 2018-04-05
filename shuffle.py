from sklearn.model_selection import ShuffleSplit
import pandas as pd

def shuffle(X, y, n_splits=3, test_size=0.25):
	X = list(X)				# convert ndarray to list
	y = y.values.tolist()	# convert pandas dataframes to values to list
	# print(len(X), len(y))
	shuffled_X = {"test": [], "train": []}
	shuffled_y = {"test": [], "train": []}
	rs = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0)
	rs.get_n_splits(X)
	for train, test in rs.split(X):
		temp_x = []
		temp_y = []
		for x in train: 
			temp_x.append(X[x])
			temp_y.append(y[x])
		shuffled_X["train"].append(temp_x)
		shuffled_y["train"].append(temp_y)
		temp_x = []
		temp_y = []
		for x in test: 
			temp_x.append(X[x])
			temp_y.append(y[x])
		shuffled_X["test"].append(temp_x)
		shuffled_y["test"].append(temp_y)

	return {"X": shuffled_X, "y": shuffled_y}
