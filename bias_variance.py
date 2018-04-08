import numpy as np

def bias(y_test, y_predict_shuffled):
	# y_noise = np.var(y_test, axis=1)
	if len(np.shape(y_predict_shuffled)) > 1:
		y_predict_avg = np.mean(y_predict_shuffled, axis=1)
		return np.mean(np.power(np.subtract(y_test, y_predict_avg), np.array(2)))
	else:
		y_predict_avg = np.sum(y_predict_shuffled)
		return np.mean(np.power(np.subtract(y_test, y_predict_avg), np.array(2)))

def variance(y_predict_shuffled):
	if len(np.shape(y_predict_shuffled)) > 1:
		return np.mean(np.var(y_predict_shuffled, axis=1))
	else:
		return np.var(y_predict_shuffled)
