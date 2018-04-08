import numpy as np

def bias(y_test, y_predict_shuffled):
	# y_noise = np.var(y_test, axis=1)
	if len(np.shape(y_predict_shuffled)) > 1:
		y_predict_avg = np.mean(y_predict_shuffled, axis=1)
		reshape_y_list = list()
		for x in y_predict_avg:
			reshape_y_list.append([x])
		reshaped_predicted_y = np.array(reshape_y_list)
		# print("Shape of y_predict_avg: ", np.shape(y_predict_avg), ", reshaped_predicted_y: ", np.shape(reshaped_predicted_y), ", subtracted: ", np.shape(np.subtract(y_test, reshaped_predicted_y)), ", powered: ", np.shape(np.power(np.subtract(y_test, reshaped_predicted_y), np.array(2))))
		# print("Shape of y_predict_avg: ", np.shape(y_predict_avg.tolist()), ", y_test: ", np.shape(y_test.tolist()))
		print("Shape of y_predict_avg: ", np.shape(y_predict_avg.tolist()), ", y_test: ", np.shape(y_test.tolist()), ", subtracted: ", np.shape(np.subtract(y_test.tolist(), y_predict_avg.tolist())), ", powered: ", np.shape(np.power(np.subtract(y_test.tolist(), y_predict_avg.tolist()), np.array(2))))
		# return np.mean(np.power(np.subtract(y_test, reshaped_predicted_y), np.array(2)))
		return np.mean(np.power(np.subtract(y_test.tolist(), y_predict_avg.tolist()), np.array(2)))
	else:
		y_predict_avg = np.sum(y_predict_shuffled)
		return np.mean(np.power(np.subtract(y_test, y_predict_avg), np.array(2)))

def variance(y_predict_shuffled):
	if len(np.shape(y_predict_shuffled)) > 1:
		return np.mean(np.var(y_predict_shuffled, axis=1))
	else:
		return np.var(y_predict_shuffled)
