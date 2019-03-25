from functools import reduce
import numpy as np
import pandas as pd
import os
import cv2
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

FILEPATH = './NewData/'

class Model:
    def __init__(self, file_path, n_components):
        self.pca = PCA(n_components=n_components, whiten=True)
        self.logistic_regression = LogisticRegression()
        self.file_path = file_path
        self.category_list = self.getCategories()
        self.data = self.getData()
        self.data_ravell = self.generate_ravell()
        self.labels = self.generate_labels()

    def getCategories(self):
        return os.walk(self.file_path).__next__()[1]

    def getData(self):
        data = {}
        for folder in self.category_list:
            folder_path = os.path.join(FILEPATH, folder)
            images = []
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                # read as grayscale image
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                images.append(img)
            data[folder] = images
        return data

    def generate_ravell(self):
        def ravell(img):
            return img.ravel()
        data_ravell = list()
        for cat in self.category_list:
            data_ravell.extend(list(map(ravell, self.data[cat])))
        return np.array(data_ravell)

    def perform_pca(self, show=True):
        self.pca.fit(self.data_ravell)
        self.pca_X = self.pca.transform(self.data_ravell)
        if show:
            eigen_faces = list(map(lambda x : x.reshape(256, 256), self.pca.components_))
            mean_eigen_face = reduce(lambda x, y : x + y, eigen_faces)
            plt.imshow(mean_eigen_face, 'gray')
            plt.show()
            for i, face in enumerate(eigen_faces):
                plt.subplot(5, 5, 1+i)
                plt.imshow(face, 'gray')
                plt.xticks(list())
                plt.yticks(list())
            plt.show()

    def generate_labels(self):
        labels = list()
        for cat in self.category_list:
            labels += [cat]*len(self.data[cat])
        return labels

    def predict_logistic(self):
        scores = cross_val_score(self.logistic_regression, self.pca_X, self.labels, cv=5, scoring='accuracy')
        print(scores)

model = Model(file_path=FILEPATH, n_components=25)
model.perform_pca(show=False)
model.predict_logistic()
