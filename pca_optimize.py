import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn import preprocessing
from sklearn.model_selection import ShuffleSplit
import shuffle
import matplotlib.pyplot as plt
import matplotlib.pyplot as plot_1
from mpl_toolkits.mplot3d import Axes3D
from kernel_regression import KernelRegression
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import sys
import time
import bias_variance as bv
import math

regression_y = "T"
dim_range = {"min": 2, "max": 3}
if len(sys.argv) > 1:
	regression_y = sys.argv[1]
if len(sys.argv) > 3:
	dim_range["min"] = int(sys.argv[2])
	dim_range["max"] = int(sys.argv[3])

print("Reading data...")
data = pd.read_csv("AirQualityUCI_clean.csv")#[1:1000]
pop_array = ["AH", "RH", "T"]
popped_array = {}
for x in pop_array:
	popped_array[x] = pd.DataFrame(data.pop(x))

def find_nearest(array, value):
	item = array[0]
	closest = abs(array[0]-value)
	index = 0
	counter = 0
	for x in array:
		counter += 1
		new_closest = min(abs(x-value), closest)
		if new_closest < closest:
			closest = new_closest
			index = counter

	return index

def gaussian_distance(dist, sigma):
	# print(dist.tolist())
	dist = dist.tolist()
	count = 0
	for x in dist:
		dist[count] = (1/(math.sqrt(2*math.pi)*sigma))*math.exp(-pow(x/sigma, 2)/2)
		# if np.isnan(dist[count]):
		# 	dist[count] = 0
		# if np.isinf(dist[count]):
		# 	dist[count] = 0
		if dist[count] == 0:
			dist[count] = 100
		count += 1

	mean_dist = np.mean(np.array(dist))
	count = 0
	for x in dist:
		if dist[count] == 0:
			dist[count] = mean_dist	
		count += 1
	return np.array(dist)

def my_distance(weights):
	count = 0
	for x in weights:
		# sigma = math.sqrt(np.var(np.array(x)))
		sigma = np.mean(np.array(x))# 0.050
		weights[count, :] = gaussian_distance(x, sigma)
		# print(weights[count, :])
		count += 1

	return weights

scores = []
dims = []
neighbor_range = {"start": 2, "end": 30, "gap": 2}

for i in range(dim_range["min"], dim_range["max"]):

	print("Dimension ", i)
	ipca = IncrementalPCA(n_components=i, batch_size=100)
	X = ipca.fit_transform(data)
	print("Transformed data X shape: ", np.shape(X), ", y shape: ", np.shape(popped_array[regression_y]))
	bias = list()
	variance = list()

	acc_var_arr = list()
	for n_neighbors in range(neighbor_range["start"], neighbor_range["end"], neighbor_range["gap"]):
		print("Neighbors: ", n_neighbors)
		cumScore = 0.0
		shuffled_data = shuffle.shuffle(X, popped_array[regression_y], 10, 0.33)
		shuffles = len(shuffled_data["X"]["train"])
		y_predict = list()
		for  j in range(0, shuffles):
			X_train = preprocessing.scale(shuffled_data["X"]["train"][j])
			X_test = preprocessing.scale(shuffled_data["X"]["test"][j])
			y_train = preprocessing.scale(shuffled_data["y"]["train"][j])
			y_test = preprocessing.scale(shuffled_data["y"]["test"][j])
			# knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance', algorithm='auto', leaf_size=30, p=2, metric='minkowski', n_jobs=-1)
			knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights=my_distance, algorithm='auto', leaf_size=30, p=2, metric='minkowski', n_jobs=-1)
			knn.fit(X_train, y_train)
			y_fit = knn.predict(X_test)
			y_predict.append(y_fit)
			# print("Shuffle: ", j, " for Target: ", regression_y, " -- Accuracy: ", knn.score(X_test, y_test))
			score = knn.score(X_test, y_test)
			cumScore = cumScore + score

		# print("Shape of y_predict: ", np.shape(y_predict), ", of y_test: ", np.shape(y_test))
		avgScore = cumScore / shuffles
		dims.append(i)
		scores.append(avgScore)
		bias.append(pow(bv.bias(y_test, np.swapaxes(y_predict, 0, 1)), 2))
		variance.append(bv.variance(y_predict))
		print("Variance in y_predict: ", variance[int((n_neighbors-neighbor_range["start"])/neighbor_range["gap"])], ", bias: ", bias[int((n_neighbors-neighbor_range["start"])/neighbor_range["gap"])], ", average score: ", avgScore)
		# acc_var_arr.append(gScore**2/mean_variance)
		# print("Accuracy^2 / Variance: ", gScore**2/mean_variance)

	plot_1.xlabel("n_neighbors")
	plot_1.ylabel("Percentage fraction")
	plot_1.plot(range(neighbor_range["start"], neighbor_range["end"], neighbor_range["gap"]), bias, '-', label="bias")
	plot_1.plot(range(neighbor_range["start"], neighbor_range["end"], neighbor_range["gap"]), variance, 'r-', label="variance")
	
print("Optimum dimension: ", find_nearest(scores, 0.85*max(scores)))
plot_1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
plot_1.title("For dimension starting from " + str(dim_range["min"]))
plot_1.show()
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
plt.title("For dimension starting from " + str(dim_range["min"]))
plot_1.xlabel("n_neighbors")
plot_1.ylabel("Accuracy")
plot_1.plot(range(neighbor_range["start"], neighbor_range["end"], neighbor_range["gap"]), scores, '-', label="accuracy")
plt.show()
# time.sleep(30)
