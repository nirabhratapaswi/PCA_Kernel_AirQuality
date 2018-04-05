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

	print("\nDIMENSION ",i)
	ipca = IncrementalPCA(n_components=i, batch_size=100)
	X = ipca.fit_transform(data)

	gScore = 0.0
	shuffled_data = shuffle.shuffle(X, popped_array[regression_y], 10, 0.33)
	shuffles = len(shuffled_data["X"]["train"])
	for  j in range(0, shuffles):
		print("Shuffle :", j)
		X_train = preprocessing.scale(shuffled_data["X"]["train"][j])
		X_test = preprocessing.scale(shuffled_data["X"]["test"][j])
		y_train = preprocessing.scale(shuffled_data["y"]["train"][j])
		y_test = preprocessing.scale(shuffled_data["y"]["test"][j])

		knn = KNeighborsRegressor(n_neighbors=i)
		knn.fit(X_train, y_train)
		print(knn.score(X_test, y_test))
		score = knn.score(X_test, y_test)
		gScore = gScore+score

	gScore = gScore/shuffles
	dims.append(i)
	scores.append(gScore)
	print("->Average score for dim ",i, ": ", gScore)
	

plt.plot(dims, scores, 'b-')
plt.ylabel("accuracy")
plt.xlabel("components")
plt.show()
