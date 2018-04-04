print("Importing libraries...")
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import kernel_reg
from sklearn.kernel_ridge import KernelRidge
from sklearn import preprocessing
from sklearn.model_selection import ShuffleSplit

print("Reading data...")
data = pd.read_csv("AirQualityUCI_clean.csv")
pop_array = ["Date", "Time", "AH", "RH", "T"]
popped_array = {}
for x in pop_array:
	popped_array[x] = pd.DataFrame(data.pop(x))

for i in range(1, 6):
	print("Forming PCA...")
	pca = PCA(n_components=i)
	X = pca.fit_transform(data)
	rs = ShuffleSplit(n_splits=3, test_size=.25, random_state=0)
	rs.get_n_splits(X)

	print("Splitting data...")
	X_train, X_test, y_train, y_test = train_test_split(X, popped_array["T"], test_size=0.2, random_state=42)
	X_train = preprocessing.scale(X_train)
	X_test = preprocessing.scale(X_test)
	y_train = preprocessing.scale(y_train)
	y_test = preprocessing.scale(y_test)
	print("Fitting data...")
	# result = kernel_reg.fit(X_test, X_train, y_train, 0.3, 'gs')
	# print("Hooray motherfuckers")
	clf = KernelRidge(alpha=1.0)
	clf.fit(X_train, y_train)
	print(clf.score(X_test, y_test))