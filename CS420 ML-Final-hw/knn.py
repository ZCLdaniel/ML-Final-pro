import numpy as np
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

#get the data
data = np.fromfile("mnist_train_data",dtype=np.uint8)
label = np.fromfile("mnist_train_label",dtype=np.uint8)

testdata = np.fromfile("../mnist_test/mnist_test_data",dtype=np.uint8)
testlabel = np.fromfile("../mnist_test/mnist_test_label",dtype=np.uint8) 

#reshape the matrix
data = data.reshape(data_num,2025)
testdata = testdata.reshape(10000,2025)
print data[1]

#pca
pca = PCA(n_components=50)
pca.fit(data)
data = pca.transform(data)
testdata = pca.transform(testdata)

print "start train"
#KNN
neighbors = KNeighborsClassifier(n_neighbors=10)
neighbors.fit(data,label)
result = neighbors.predict(testdata)
print(classification_report(testlabel, result))
