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

regression_y = "T"
if len(sys.argv) > 1:
	regression_y = sys.argv[1]

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

def my_distance(weights):
    return weights

scores = []
dims = []
# plot_1 = plt.figure()
# plot_1_3d = plot_1.add_subplot(111, projection='3d')
# plot_1_3d = Axes3D(plot_1)

for i in range(1, 10):

	print("Dimension ", i)
	ipca = IncrementalPCA(n_components=i, batch_size=100)
	X = ipca.fit_transform(data)
	print("Transformed data X shape: ", np.shape(X), ", y shape: ", np.shape(popped_array[regression_y]))
	# plot_1.xlabel("X1")
	# plot_1.ylabel("X2")
	# plot_1_3d.set_xlabel("X1")
	# plot_1_3d.set_ylabel("X2")
	# plot_1_3d.set_zlabel("X3")
	# plot_1_3d.set_xlim(-200, 200)
	# plot_1_3d.set_ylim(-200, 200)
	# plot_1_3d.set_zlim(-50, 50)
	# plot_1.plot(X[:, 1], X[:, 1], 'r.')
	# plot_1_3d.scatter(X[:, 1], X[:, 1], X[:, 2], '.')

	gScore = 0.0
	shuffled_data = shuffle.shuffle(X, popped_array[regression_y], 6, 0.33)
	shuffles = len(shuffled_data["X"]["train"])
	y_predict = list()
	for  j in range(0, shuffles):
		X_train = preprocessing.scale(shuffled_data["X"]["train"][j])
		X_test = preprocessing.scale(shuffled_data["X"]["test"][j])
		y_train = preprocessing.scale(shuffled_data["y"]["train"][j])
		y_test = preprocessing.scale(shuffled_data["y"]["test"][j])
		knn = KNeighborsRegressor(n_neighbors=int(pow(i, 2.5))+1, weights='distance', algorithm='auto', leaf_size=30, p=i, metric='minkowski', n_jobs=-1)
		knn.fit(X_train, y_train)
		y_fit = knn.predict(X_test)
		y_predict.append(y_fit)
		# print(len(y_fit), len(X_test))
		# plot_1.xlabel("X_test")
		# plot_1.ylabel("y_fit")
		# plot_1.plot(X_test, y_fit, '.')
		# plot_1.show()
		print("Shuffle: ", j, " for Target: ", regression_y, " -- Accuracy: ", knn.score(X_test, y_test))
		score = knn.score(X_test, y_test)
		gScore = gScore+score

	print("Variance in predicted y: ", np.var(np.array(y_predict)))
	gScore = gScore/shuffles
	dims.append(i)
	scores.append(gScore)
	print("->Average score for dim ",i, ": ", gScore)
	
print("Optimum dimension: ", find_nearest(scores, 0.85*max(scores)))
# plot_1.xlabel("components")
# plot_1.ylabel("accuracy")
# plot_1.plot(dims, scores, 'b-')
# plot_1.ylabel("accuracy")
# plot_1.xlabel("components")
# plot_1.show()
# time.sleep(30)
