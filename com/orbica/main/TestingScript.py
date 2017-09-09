import numpy as np

np.random.seed(1337)
import pandas
from sklearn.preprocessing import Normalizer

from keras.layers import Dense
from keras.models import Sequential, model_from_json
from keras.wrappers.scikit_learn import KerasRegressor
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from keras.models import model_from_json

# create model function and configure input and output neurons (single fully connected hidden layer)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support

inputFilePath = "/media/sagar/DATA/Orbica/Work/Dataset/OL017 ECan Waterbodies Classification/Merge_vectorLayers_Training/Model/Test_Data/EcanWaterBodies_Online/Nodes_EcanWaterBodies_Online_Model_7Sep.csv"

#modelWeights="/media/sagar/DATA/Orbica/Work/Dataset/ECanWaterBodies/Test_Data_ECAN/Model/Final/model_7Sep_Theano.h5"

modelWeights="/media/sagar/DATA/Orbica/Work/Dataset/OL017 ECan Waterbodies Classification/Merge_vectorLayers_Training/Model/Train_Data/model_Theano_5000_8Sep.h5"

outputPath = "/media/sagar/DATA/Orbica/Work/Dataset/OL017 ECan Waterbodies Classification/Merge_vectorLayers_Training/Model/Test_Data/EcanWaterBodies_Online/Nodes_EcanWaterBodies_Online_ModelPredictions_8Sep.csv"

# fix random seed for reproducibility
# from keras.models import load_model
# loaded_model = load_model(modelPath)

# create model
model = Sequential()
model.add(Dense(500, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(100, kernel_initializer='uniform', activation='relu'))
model.add(Dense(50, kernel_initializer='uniform', activation='relu'))
model.add(Dense(20, kernel_initializer='uniform', activation='relu'))
model.add(Dense(6, activation='softmax'))

# load weights into new model
model.load_weights(modelWeights)

print(model.weights)

print("Loaded model from disk")

# load dataset
dataframe = pandas.read_csv(inputFilePath, sep=",", header=None)
dataframe = dataframe._values
# split into input (X) and output (Y) variables
X_Feature_Vector = dataframe[:, 1:9]

X_trainOriginal = dataframe[:,:]

scaler = Normalizer().fit(X_Feature_Vector)
rescaledX = scaler.transform(X_Feature_Vector)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

Y_Predicted = model.predict(rescaledX)
y_classes = Y_Predicted.argmax(axis=-1)

labels = np.zeros(Y_Predicted.shape)

prob = model.predict_proba(rescaledX)

flag = True

ActualSet=dataframe[:,9]
ActualLabelsInt=[]
f = open(outputPath, 'w')
f.write("Test_ID,Min,Max,Elev_diff,Area,Perimeter,AreaLengthRatio,NoofNodes,AvgPolygonWidth,Actual,classLabel1,Score1,classLabel2,Score2,classLabel3,Score3,Difference\n")
for x_data, prob,Actual in zip(X_trainOriginal, prob,ActualSet):
    for i in x_data:
        if (flag):
            f.write("%d" % i + ",")
            flag = False
        else:
            data = str((i))
            f.write(data + ",")
    flag = True
    prob_s = np.around(prob, decimals=5)
    predictions = prob_s.argsort()[-3:][::-1]
    scoreAll = -np.sort(-prob_s)
    scoreAll = scoreAll[:3]
    allresults = ""
    LabelStart=True
    diff=0
    for predicted, score in zip(predictions, scoreAll):
        if (predicted == 0):
            predicted = "RIVER"
        elif (predicted == 1):
            predicted = "CANAL"
        elif (predicted == 2):
            predicted = "LAKE"
        elif (predicted == 3):
            predicted = "POND"
        elif (predicted == 4):
            predicted = "LAGOON"
        elif (predicted == 5):
            predicted = "RESERVOIR"
        if(LabelStart):
            if(Actual==predicted):
                diff=1
            LabelStart= False

        scoreInt = (int)(score * 100)
        allresults += predicted + "," + (("%d" % scoreInt)) + ","

    allresults = allresults[0:allresults.__len__() - 1]
    f.write(str(allresults)+","+"%d" % diff + "\n")



for value in ActualSet:
    if (value == "RIVER"):
        value = 0
    elif(value == "CANAL"):
        value = 1
    elif(value == "LAKE"):
        value = 2
    elif(value =="POND"):
        value = 3
    elif(value =="LAGOON"):
        value = 4
    elif(value == "RESERVOIR"):
        value = 5
    ActualLabelsInt.append(value)

#ActualLabelsInt=np.array(ActualLabelsInt)
Y_Output_Encode_test=np_utils.to_categorical(ActualLabelsInt)

scores = model.evaluate(rescaledX,Y_Output_Encode_test)
print("\nModel Prediction Accuracy For test Data: %.2f%%" % (scores[1] * 100))
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

PredictedSet=[]

print("Confusion Matrix")
matrix = confusion_matrix(ActualLabelsInt, y_classes)
print(matrix)

# print(f1_score(ActualLabelsInt, y_classes, average="macro"))
# print(precision_score(ActualLabelsInt, y_classes, average="macro"))
# print(recall_score(ActualLabelsInt, y_classes, average="macro"))

    #
    # labels = ['business', 'health']
    # cm = confusion_matrix(y_test, pred, labels)
    # print(cm)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # cax = ax.matshow(cm)
    # pl.title('Confusion matrix of the classifier')
    # fig.colorbar(cax)
    # ax.set_xticklabels([''] + labels)
    # ax.set_yticklabels([''] + labels)
    # pl.xlabel('Predicted')
    # pl.ylabel('True')
    # pl.show()

# import numpy as np
#
# np.random.seed(1337)
# import pandas
# from sklearn.preprocessing import Normalizer
#
# from keras.layers import Dense
# from keras.models import Sequential, model_from_json
# from keras.wrappers.scikit_learn import KerasRegressor
# from keras.wrappers.scikit_learn import KerasRegressor
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# from keras.utils import np_utils
# from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
# from keras.models import model_from_json
#
# # create model function and configure input and output neurons (single fully connected hidden layer)
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import precision_recall_fscore_support
#
# inputFilePath = "/media/sagar/DATA/Orbica/Work/Dataset/ECanWaterBodies/Test_Data_ECAN/NZTM_DIRTY_WATEROUTLINES/NZTM_DIRTY_WATEROUTLINES_Model.csv"
#
# modelWeights="/media/sagar/DATA/Orbica/Work/Dataset/ECanWaterBodies/Test_Data_ECAN/Model/Final/model_7Sep_Theano.h5"
#
# outputPath = "/media/sagar/DATA/Orbica/Work/Dataset/ECanWaterBodies/Test_Data_ECAN/Model/Final/NZTM_ModelPredictions.csv"
#
# # fix random seed for reproducibility
# # from keras.models import load_model
# # loaded_model = load_model(modelPath)
#
# # create model
# model = Sequential()
# model.add(Dense(500, input_dim=8, kernel_initializer='uniform', activation='relu'))
# model.add(Dense(100, kernel_initializer='uniform', activation='relu'))
# model.add(Dense(50, kernel_initializer='uniform', activation='relu'))
# model.add(Dense(20, kernel_initializer='uniform', activation='relu'))
# model.add(Dense(6, activation='softmax'))
#
# # load weights into new model
# model.load_weights(modelWeights)
#
# print(model.weights)
#
# print("Loaded model from disk")
#
# # load dataset
# dataframe = pandas.read_csv(inputFilePath, sep=",", header=None)
# dataframe = dataframe._values
# # split into input (X) and output (Y) variables
# X_Feature_Vector = dataframe[:, 1:9]
#
# X_trainOriginal = dataframe[:,:]
#
# scaler = Normalizer().fit(X_Feature_Vector)
# rescaledX = scaler.transform(X_Feature_Vector)
#
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# Y_Predicted = model.predict(rescaledX)
# y_classes = Y_Predicted.argmax(axis=-1)
#
# labels = np.zeros(Y_Predicted.shape)
#
# prob = model.predict_proba(rescaledX)
#
# flag = True
#
#
#
#
# f = open(outputPath, 'w')
# f.write(
#     "Test_ID,Min,Max,Elev_diff,Area,Perimeter,AreaLengthRatio,NoofNodes,AvgPolygonWidth,Actual,classLabel1,Score1,classLabel2,Score2,classLabel3,Score3\n")
# for x_data, prob in zip(X_trainOriginal, prob):
#     for i in x_data:
#         if (flag):
#             f.write("%d" % i + ",")
#             flag = False
#         else:
#             data = str((i))
#             f.write(data + ",")
#     flag = True
#     prob_s = np.around(prob, decimals=5)
#     predictions = prob_s.argsort()[-3:][::-1]
#     scoreAll = -np.sort(-prob_s)
#     scoreAll = scoreAll[:3]
#     allresults = ""
#     LabelStart=True
#     for predicted, score in zip(predictions, scoreAll):
#         if (predicted == 0):
#             predicted = "RIVER"
#         elif (predicted == 1):
#             predicted = "CANAL"
#         elif (predicted == 2):
#             predicted = "LAKE"
#         elif (predicted == 3):
#             predicted = "POND"
#         elif (predicted == 4):
#             predicted = "LAGOON"
#         elif (predicted == 5):
#             predicted = "RESERVOIR"
#         # if(LabelStart):
#         #     PredictedSet.append(predicted)
#         #     LabelStart= False
#
#         scoreInt = (int)(score * 100)
#         allresults += predicted + "," + (("%d" % scoreInt)) + ","
#
#     allresults = allresults[0:allresults.__len__() - 1]
#     f.write(str(allresults) + "\n")
#
#
# ActualSet=dataframe[:,9]
# ActualLabelsInt=[]
#
# for value in ActualSet:
#     if (value == "RIVER"):
#         value = 0
#     elif(value == "CANAL"):
#         value = 1
#     elif(value == "LAKE"):
#         value = 2
#     elif(value =="POND"):
#         value = 3
#     elif(value =="LAGOON"):
#         value = 4
#     elif(value == "RESERVOIR"):
#         value = 5
#     ActualLabelsInt.append(value)
#
# #ActualLabelsInt=np.array(ActualLabelsInt)
# Y_Output_Encode_test=np_utils.to_categorical(ActualLabelsInt)
#
# scores = model.evaluate(rescaledX,Y_Output_Encode_test)
# print("\nModel Prediction Accuracy For test Data: %.2f%%" % (scores[1] * 100))
# print("Baseline Error: %.2f%%" % (100-scores[1]*100))
#
# PredictedSet=[]
#
# print("Confusion Matrix")
# matrix = confusion_matrix(ActualLabelsInt, y_classes)
# print(matrix)
#
# # print(f1_score(ActualLabelsInt, y_classes, average="macro"))
# # print(precision_score(ActualLabelsInt, y_classes, average="macro"))
# # print(recall_score(ActualLabelsInt, y_classes, average="macro"))
#
#     #
#     # labels = ['business', 'health']
#     # cm = confusion_matrix(y_test, pred, labels)
#     # print(cm)
#     # fig = plt.figure()
#     # ax = fig.add_subplot(111)
#     # cax = ax.matshow(cm)
#     # pl.title('Confusion matrix of the classifier')
#     # fig.colorbar(cax)
#     # ax.set_xticklabels([''] + labels)
#     # ax.set_yticklabels([''] + labels)
#     # pl.xlabel('Predicted')
#     # pl.ylabel('True')
#     # pl.show()
# # import numpy as np
# #
# # np.random.seed(1337)
# # import pandas
# # from sklearn.preprocessing import Normalizer
# #
# # from keras.layers import Dense
# # from keras.models import Sequential, model_from_json
# # from keras.wrappers.scikit_learn import KerasRegressor
# # from keras.wrappers.scikit_learn import KerasRegressor
# # from sklearn.model_selection import cross_val_score
# # from sklearn.model_selection import KFold
# # from keras.utils import np_utils
# # from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
# # from keras.models import model_from_json
# #
# # # create model function and configure input and output neurons (single fully connected hidden layer)
# # from sklearn.pipeline import Pipeline
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.metrics import precision_recall_fscore_support
# #
# # inputFilePath = "/media/sagar/DATA/Orbica/Work/Dataset/ECanWaterBodies/Test_Data_ECAN/NZTM_DIRTY_WATEROUTLINES/NZTM_DIRTY_WATEROUTLINES_Model.csv"
# #
# # modelWeights="/media/sagar/DATA/Orbica/Work/Dataset/ECanWaterBodies/Test_Data_ECAN/Model/Final/model_7Sep_Theano.h5"
# #
# # outputPath = "/media/sagar/DATA/Orbica/Work/Dataset/ECanWaterBodies/Test_Data_ECAN/Model/Final/NZTM_ModelPredictions.csv"
# #
# # # fix random seed for reproducibility
# # # from keras.models import load_model
# # # loaded_model = load_model(modelPath)
# #
# # # create model
# # model = Sequential()
# # model.add(Dense(500, input_dim=8, kernel_initializer='uniform', activation='relu'))
# # model.add(Dense(100, kernel_initializer='uniform', activation='relu'))
# # model.add(Dense(50, kernel_initializer='uniform', activation='relu'))
# # model.add(Dense(20, kernel_initializer='uniform', activation='relu'))
# # model.add(Dense(6, activation='softmax'))
# #
# # # load weights into new model
# # model.load_weights(modelWeights)
# #
# # print(model.weights)
# #
# # print("Loaded model from disk")
# #
# # # load dataset
# # dataframe = pandas.read_csv(inputFilePath, sep=",", header=None)
# # dataframe = dataframe._values
# # # split into input (X) and output (Y) variables
# # X_Feature_Vector = dataframe[:, 1:9]
# #
# # X_trainOriginal = dataframe[:,:]
# #
# # scaler = Normalizer().fit(X_Feature_Vector)
# # rescaledX = scaler.transform(X_Feature_Vector)
# #
# # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# #
# # Y_Predicted = model.predict(rescaledX)
# # y_classes = Y_Predicted.argmax(axis=-1)
# #
# # labels = np.zeros(Y_Predicted.shape)
# #
# # prob = model.predict_proba(rescaledX)
# #
# # flag = True
# #
# #
# #
# #
# # f = open(outputPath, 'w')
# # f.write(
# #     "Test_ID,Min,Max,Elev_diff,Area,Perimeter,AreaLengthRatio,NoofNodes,AvgPolygonWidth,Actual,classLabel1,Score1,classLabel2,Score2,classLabel3,Score3\n")
# # for x_data, prob in zip(X_trainOriginal, prob):
# #     for i in x_data:
# #         if (flag):
# #             f.write("%d" % i + ",")
# #             flag = False
# #         else:
# #             data = str((i))
# #             f.write(data + ",")
# #     flag = True
# #     prob_s = np.around(prob, decimals=5)
# #     predictions = prob_s.argsort()[-3:][::-1]
# #     scoreAll = -np.sort(-prob_s)
# #     scoreAll = scoreAll[:3]
# #     allresults = ""
# #     LabelStart=True
# #     for predicted, score in zip(predictions, scoreAll):
# #         if (predicted == 0):
# #             predicted = "RIVER"
# #         elif (predicted == 1):
# #             predicted = "CANAL"
# #         elif (predicted == 2):
# #             predicted = "LAKE"
# #         elif (predicted == 3):
# #             predicted = "POND"
# #         elif (predicted == 4):
# #             predicted = "LAGOON"
# #         elif (predicted == 5):
# #             predicted = "RESERVOIR"
# #         # if(LabelStart):
# #         #     PredictedSet.append(predicted)
# #         #     LabelStart= False
# #
# #         scoreInt = (int)(score * 100)
# #         allresults += predicted + "," + (("%d" % scoreInt)) + ","
# #
# #     allresults = allresults[0:allresults.__len__() - 1]
# #     f.write(str(allresults) + "\n")
# #
# #
# # ActualSet=dataframe[:,9]
# # ActualLabelsInt=[]
# #
# # for value in ActualSet:
# #     if (value == "RIVER"):
# #         value = 0
# #     elif(value == "CANAL"):
# #         value = 1
# #     elif(value == "LAKE"):
# #         value = 2
# #     elif(value =="POND"):
# #         value = 3
# #     elif(value =="LAGOON"):
# #         value = 4
# #     elif(value == "RESERVOIR"):
# #         value = 5
# #     ActualLabelsInt.append(value)
# #
# # #ActualLabelsInt=np.array(ActualLabelsInt)
# # Y_Output_Encode_test=np_utils.to_categorical(ActualLabelsInt)
# #
# # scores = model.evaluate(rescaledX,Y_Output_Encode_test)
# # print("\nModel Prediction Accuracy For test Data: %.2f%%" % (scores[1] * 100))
# # print("Baseline Error: %.2f%%" % (100-scores[1]*100))
# #
# # PredictedSet=[]
# #
# # print("Confusion Matrix")
# # matrix = confusion_matrix(ActualLabelsInt, y_classes)
# # print(matrix)
# #
# # # print(f1_score(ActualLabelsInt, y_classes, average="macro"))
# # # print(precision_score(ActualLabelsInt, y_classes, average="macro"))
# # # print(recall_score(ActualLabelsInt, y_classes, average="macro"))
# #
# #     #
# #     # labels = ['business', 'health']
# #     # cm = confusion_matrix(y_test, pred, labels)
# #     # print(cm)
# #     # fig = plt.figure()
# #     # ax = fig.add_subplot(111)
# #     # cax = ax.matshow(cm)
# #     # pl.title('Confusion matrix of the classifier')
# #     # fig.colorbar(cax)
# #     # ax.set_xticklabels([''] + labels)
# #     # ax.set_yticklabels([''] + labels)
# #     # pl.xlabel('Predicted')
# #     # pl.ylabel('True')
# #     # pl.show()
