

from functools import reduce
import numpy as np
# import pandas as pd
import os
import cv2
import warnings
import matplotlib
# matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
# plt.plot(range(20), range(20))
# plt.show()
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

FILEPATH = './NewData/'
warnings.filterwarnings('ignore')

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
            plt.savefig('screenshots/meaneigenface.png')
            plt.show()

            for i, face in enumerate(eigen_faces):
                plt.subplot(10, 20, 1+i)
                plt.imshow(face, 'gray')
                plt.xticks(list())
                plt.yticks(list())
            plt.savefig('screenshots/eigenfaces.png')
            plt.show()

            for i in range(10):
                original_face = self.data_ravell[i].reshape(256, 256)
                eigen_coefficients = list(self.pca_X[i])
                eigen_face = sum(map(lambda coeff: eigen_faces[coeff[0]]*coeff[1], list(enumerate(eigen_coefficients))))
                plt.subplot(2, 10, i+1)
                plt.imshow(original_face, 'gray') , plt.xticks(list()), plt.yticks(list())
                plt.subplot(2, 10, i+11)
                plt.imshow(eigen_face, 'gray') , plt.xticks(list()), plt.yticks(list())
            plt.savefig('screenshots/eigenfaces_comparison.png')
            plt.show()

    def generate_labels(self):
        labels = list()
        for i, cat in enumerate(self.category_list):
            labels += [i]*len(self.data[cat])
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
            print('PCA - ', model.__class__.__name__, np.mean(scores))

    def perform_lda(self):
        clf = LinearDiscriminantAnalysis()
        clf.fit(self.data_ravell, self.labels)
        scores = cross_val_score(clf, self.data_ravell, self.labels, cv=5, scoring='accuracy')
        print(clf.__class__.__name__, np.mean(scores))

    def perform_face_recognition(self, model):
        print("Running ", model.__class__.__name__)
        train_data, test_data, train_labels, test_labels = train_test_split(self.data_list, self.labels, test_size=0.2, random_state=42)
        model.train(train_data, train_labels)
        # score = 0
        # fail = 0
        # for i in range(test_data.shape[0]):
        #     if(model.predict(test_data[i, :, :])[0] == test_labels.reshape(-1)[i]):
        #         score += 1
        #     else:
        #         fail += 1
        # print(score/(score+fail))

        def eval(n=1):
            score = 0
            for i in range(len(test_data)):
                num = 0
                id = model.predict(test_data[i, :, :])[0]
                model.predict(test_data[i])[0]
                model.predict_collect(test_data[i], collector)

                results = collector.getResults(sorted=True)

                for (label, dist) in results:
                    if num < n:
                        if label == test_labels[i]:
                            score += 1
                            break
                    else:
                        break
                    num += 1
            return score


        total = test_data.shape[0]
        collector = cv2.face.StandardCollector_create()
        top_1_score = eval(1)
        top_3_score = eval(3)
        top_10_score = eval(10)

        print("Top 1 accuracy", top_1_score/total)
        print("Top 3 accuracy", top_3_score/total)
        print("Top 10 accuracy", top_10_score/total)


if __name__ == '__main__':
    model = Model(file_path=FILEPATH, n_components=200)
    model.labels = model.create_gender_labels()
    model.perform_pca(show=False)
    models = list()
    models.append(LogisticRegression())
    models.append(SVC(kernel='linear'))
    model.model_predict(models=models)
    # model.perform_lda()
    # eigen_model = cv2.face.EigenFaceRecognizer_create()
    # fisher_model = cv2.face.FisherFaceRecognizer_create()
    # lbph_model = cv2.face.LBPHFaceRecognizer_create()
    # model.perform_face_recognition(eigen_model)
    # model.perform_face_recognition(fisher_model)
    # model.perform_face_recognition(lbph_model)
