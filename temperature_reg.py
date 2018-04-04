import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
import matplotlib.pyplot as plt

filename = "AirQualityUCI.csv"
data = np.loadtxt(filename, delimiter=";")
print(data.shape)