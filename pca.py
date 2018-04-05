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
import sys

regression_y = "T"
if len(sys.argv) > 1:
	regression_y = sys.argv[1]

print("Reading data...")
data = pd.read_csv("AirQualityUCI_clean.csv")
pop_array = ["AH", "RH", "T"]
popped_array = {}
for x in pop_array:
	popped_array[x] = pd.DataFrame(data.pop(x))

plt.xlabel("components")
plt.ylabel("accuracy")
scores = []
dims = []

for i in range(1, 13):
	# pca = PCA(n_components=i)
	ipca = IncrementalPCA(n_components=i, batch_size=100)
	# X = pca.fit_transform(data)
	X = ipca.fit_transform(data)

	gScore = 0.0
	print("Shuffling & splitting data...")
	shuffled_data = shuffle.shuffle(X, popped_array[regression_y], 10, 0.33)
	# X_train, X_test, y_train, y_test = train_test_split(X, popped_array["T"], test_size=0.2, random_state=42)
	for  j in range(0, len(shuffled_data["X"]["train"])):
		X_train = preprocessing.scale(shuffled_data["X"]["train"][j])
		X_test = preprocessing.scale(shuffled_data["X"]["test"][j])
		y_train = preprocessing.scale(shuffled_data["y"]["train"][j])
		y_test = preprocessing.scale(shuffled_data["y"]["test"][j])
		print("Fitting data for ", i, "th dimensional PCA -- shuffle: ", j, " for target: ", regression_y, "...")
		knn = KNeighborsRegressor(n_neighbors=i, weights='distance', algorithm='auto', leaf_size=30, p=2, metric='minkowski')
		knn.fit(X_train, y_train)
		print(knn.score(X_test, y_test))
		score = knn.score(X_test, y_test)
		if gScore<score:
			gScore=score

	dims.append(i)
	scores.append(gScore)
	

plt.plot(dims, scores, 'b-')
plt.ylabel("accuracy")
plt.xlabel("components")
plt.show()
