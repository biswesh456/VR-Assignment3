from functools import reduce
import numpy as np
import pandas as pd
import os
import cv2
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

FILEPATH = './NewData/'

class Model:
    def __init__(self, file_path, n_components):
        self.pca = PCA(n_components=n_components, whiten=True)
        self.file_path = file_path
        self.category_list = self.getCategories()
        self.data = self.getData()
        self.data_ravell = self.generate_ravell()
        self.data_list = self.generate_data_list()
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
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                images.append(img)
            data[folder] = images
        return data

    def generate_data_list(self):
        data_list = list()
        for cat in self.category_list:
            data_list.extend(self.data[cat])
        return np.array(data_list)

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
                plt.subplot(10, 20, 1+i)
                plt.imshow(face, 'gray')
                plt.xticks(list())
                plt.yticks(list())
            plt.show()

    def generate_labels(self):
        labels = list()
        for cat in self.category_list:
            labels += [0]*len(self.data[cat])
        return np.array(labels).reshape(-1, 1)

    def create_gender_labels(self):
        girls = ["Anagha", "Deepika", "Deepti", "Devyani", "Juhi", "Nehal", "Prachi", "Pragya", "Shiloni", "Sowmya", "Sravya", "Tripti"]
        labels = list()
        for cat in self.category_list:
            labels += [1 if cat in girls else 0]*len(self.data[cat])

        return np.array(labels).reshape(-1, 1)

    def model_predict(self, models=list()):
        for model in models:
            scores = cross_val_score(model, self.pca_X, self.labels, cv=5, scoring='accuracy')
            print(model.__class__.__name__, np.mean(scores))

    def perform_lda(self):
        clf = LinearDiscriminantAnalysis()
        clf.fit(self.data_ravell, self.labels)
        scores = cross_val_score(clf, self.data_ravell, self.labels, cv=5, scoring='accuracy')
        print(clf.__class__.__name__, np.mean(scores))

    def perform_lbp(self):
        lbp = cv2.face.LBPHFaceRecognizer_create()
        print(len(self.data_list))
        print(len(self.labels))
        lbp.train(self.data_list, self.labels)
        lbp.predict(self.data_list)

model = Model(file_path=FILEPATH, n_components=200)
# model.perform_pca(show=True)
# models = list()
# models.append(LogisticRegression())
# models.append(SVC(kernel='linear'))
# model.model_predict(models=models)
# model.perform_lda()
model.perform_lbp()
