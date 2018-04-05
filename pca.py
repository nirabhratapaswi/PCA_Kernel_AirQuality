print("Importing libraries...")
import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn import preprocessing
from sklearn.model_selection import ShuffleSplit
import shuffle
import matplotlib.pyplot as plt
from kernel_regression import KernelRegression
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

print("Reading data...")
data = pd.read_csv("AirQualityUCI_clean.csv")
pop_array = ["Date", "Time", "AH", "RH", "T"]
popped_array = {}
for x in pop_array:
	popped_array[x] = pd.DataFrame(data.pop(x))

for i in range(1, 11):
	print("Forming PCA...")
	# pca = PCA(n_components=i)
	ipca = IncrementalPCA(n_components=i, batch_size=100)
	# X = pca.fit_transform(data)
	X = ipca.fit_transform(data)

	print("Shuffling & splitting data...")
	shuffled_data = shuffle.shuffle(X, popped_array["RH"], 10, 0.33)
	# X_train, X_test, y_train, y_test = train_test_split(X, popped_array["T"], test_size=0.2, random_state=42)
	for  j in range(0, len(shuffled_data["X"]["train"])):
		# X_train = preprocessing.scale(X_train)
		# X_test = preprocessing.scale(X_test)
		# y_train = preprocessing.scale(y_train)
		# y_test = preprocessing.scale(y_test)
		X_train = preprocessing.scale(shuffled_data["X"]["train"][j])
		X_test = preprocessing.scale(shuffled_data["X"]["test"][j])
		y_train = preprocessing.scale(shuffled_data["y"]["train"][j])
		y_test = preprocessing.scale(shuffled_data["y"]["test"][j])
		print("Fitting data for shuffle: ", j, " ...")
		# clf = KernelRidge(alpha=1.0)
		# clf.fit(X_train, y_train)
		# print(clf.score(X_test, y_test))
		neigh = KNeighborsRegressor(n_neighbors=i)
		neigh.fit(X_train, y_train)
		print(neigh.score(X_test, y_test))
