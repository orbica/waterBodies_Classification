import tensorflow
import numpy as np
np.random.seed(1337)
import pandas
from sklearn.preprocessing import Normalizer

from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from keras.utils import np_utils
from keras.layers import Dropout
import matplotlib.pyplot as plt
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

import matplotlib.cm as cm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

inputFilePath = "/media/sagar/DATA/Orbica/Work/Dataset/Analysis/ML_Data/1Sep_NewFeature/Featurefile_6Labels_1Sep (copy).csv"

modelWeights="/media/sagar/DATA/Orbica/Work/Dataset/Analysis/Model_Data/1Sep_6Classlabels/BestModel/Best_Model_2/model_7Sep_Theano_20000.h5"

outputPath="/media/sagar/DATA/Orbica/Work/Dataset/Analysis/Model_Data/1Sep_6Classlabels/BestModel/Best_Model_2/Actual_Predicted_7Sep_Theano_20000.csv"

# fix random seed for reproducibility


# load dataset from CSV into pandas object
dataframe = pandas.read_csv(inputFilePath, sep=",", header=0)
dataframe = dataframe._values


X_Feature_Vector = dataframe[:, 1:9]
Y_Output_Vector = dataframe[:, 9]

X_trainOriginal= dataframe[:, 0:9]
Y_trainOriginal=Y_Output_Vector

scaler = Normalizer().fit(X_Feature_Vector)
rescaledX = scaler.transform(X_Feature_Vector)

Y_Output_Encode_train=np_utils.to_categorical(Y_Output_Vector)
numClasses=Y_Output_Encode_train.shape[1]

# create model
model = Sequential()
model.add(Dense(500,input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(100, kernel_initializer='uniform', activation='relu'))
model.add(Dense(50, kernel_initializer='uniform', activation='relu'))
model.add(Dense(20, kernel_initializer='uniform', activation='relu'))
model.add(Dense(numClasses, activation='softmax'))

print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(rescaledX, Y_Output_Encode_train, nb_epoch=20000, batch_size=100)

model.save_weights(modelWeights)

scores = model.evaluate(rescaledX,Y_Output_Encode_train)

# serialize weights to HDF5
print("Saved model to disk")


print("\nModel Prediction Accuracy For Train Data: %.2f%%" % (scores[1] * 100))
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

Y_Predicted =model.predict(rescaledX)
y_classes = Y_Predicted.argmax(axis=-1)


print ("Confusion Matrix")
matrix=confusion_matrix(Y_trainOriginal, y_classes)
print(matrix)


f = open(outputPath, 'w')
f.write("t50_fid,Min,Max,Elev_diff,Area,Perimeter,AreaLengthRatio,NoofNodes,AvgPolygonWidth,classLabel\n")
for x_data,original,predicted in zip(X_trainOriginal,Y_trainOriginal,y_classes):

    for i in x_data:
            data = str((i))
            f.write(data + ",")

    actual=(int)(original)
    predicted= (int)(predicted)

# Convert Numbers to Unique Class labels Actual
    if(actual==0):
        actual ="River"
    elif(actual==1):
        actual="Canal"
    elif (actual == 2):
        actual = "Lake"
    elif (actual == 3):
        actual = "Pond"
    elif (actual == 4):
        actual = "Lagoon"
    elif (actual == 5):
        actual = "Reservoir"


# Convert Numbers to Unique Class labels Predicted
    if(predicted==0):
        predicted ="River"
    elif(predicted==1):
        predicted="Canal"
    elif (predicted == 2):
        predicted = "Lake"
    elif (predicted == 3):
        predicted = "Pond"
    elif (predicted == 4):
        predicted = "Lagoon"
    elif (predicted == 5):
        predicted = "Reservoir"

    f.write(str((actual))+","+str((predicted))+"\n")

print(model.weights)


plt.matshow(matrix)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# import tensorflow
# import numpy as np
# import pandas
# from keras.models import Sequential
# from keras.layers import Dense
# from sklearn.preprocessing import StandardScaler
# from keras.utils import np_utils
# from keras.layers import Dropout
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
#
# import matplotlib.cm as cm
# from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
#
# inputFilePath = "/media/sagar/DATA/Orbica/Work/Dataset/Analysis/ML_Data/Featurefile_10Labels_Elevations.csv"
# modelPath = "/media/sagar/DATA/Orbica/Work/Dataset/Analysis/Datasets_ML/1Sep_5Features/model.hdf5"
# modelJsonPath = "/media/sagar/DATA/Orbica/Work/Dataset/Analysis/Datasets_ML/1Sep_5Features/model.json"
# outputPath="/media/sagar/DATA/Orbica/Work/Dataset/Analysis/Datasets_ML/1Sep_5Features/ActualPredicted.csv"
#
#
# def train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=None):
#     np.random.seed(seed)
#     perm = np.random.permutation(df.index)
#     m = len(df)
#     train_end = int(train_percent * m)
#     validate_end = int(validate_percent * m) + train_end
#     train = df.ix[perm[:train_end]]
#     validate = df.ix[perm[train_end:validate_end]]
#     test = df.ix[perm[validate_end:]]
#     return train, validate, test
#
# # fix random seed for reproducibility
# seed = 7
# np.random.seed(seed)
#
# # load dataset from CSV into pandas object
# dataframe = pandas.read_csv(inputFilePath, sep=",", header=0)
# dataframe = dataframe._values
#
#
# X_Feature_Vector = dataframe[:, 1:6]
# Y_Output_Vector = dataframe[:, 9]
#
#
#
# #train,val,test=train_validate_test_split(dataframe)
# #train = train._values
# #val = val._values
#
#
# # split into input (X) and output (Y) variables
# # X_FeatureValue=train[:,0:9]
# #
# # X_Feature_Vector = train[:, 1:9]
# # Y_Output_Vector = train[:, 9]
# # scaler = StandardScaler().fit(X_Feature_Vector)
# # X_Feature_Vector = scaler.fit_transform(X_Feature_Vector)
# #
# # Y_Output_Encode=np_utils.to_categorical(Y_Output_Vector)
# # numClasses=Y_Output_Encode.shape[1]
# #
# # X_Feature_Vector_val = val[:, 1:9]
# # Y_Output_Vector_val = val[:, 9]
# #scaler1 = StandardScaler().fit(X_Feature_Vector_val)
# #X_Feature_Vector_val = scaler1.fit_transform(X_Feature_Vector_val)
#
# X_train, X_test, y_train, y_test = train_test_split(X_Feature_Vector, Y_Output_Vector, test_size=0.40, random_state=seed)
#
# X_testOriginal=X_test
# Y_testOriginal=y_test
#
# Y_trainOriginal=y_train
#
#
# scaler = StandardScaler().fit(X_train)
# X_train = scaler.fit_transform(X_train)
#
# scaler1 = StandardScaler().fit(X_test)
# X_test = scaler1.fit_transform(X_test)
#
# Y_Output_Encode_train=np_utils.to_categorical(y_train)
# numClasses=Y_Output_Encode_train.shape[1]
#
# Y_Output_Encode_test=np_utils.to_categorical(y_test)
# numClasses1=Y_Output_Encode_test.shape[1]
#
#
#
# # create model
# model = Sequential()
# model.add(Dense(5,input_dim=5, kernel_initializer='normal', activation='relu'))
#
# #model.add(Dense(15,kernel_initializer='normal', activation='relu'))
# model.add(Dense(numClasses, activation='softmax'))
#
# print(model.summary())
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# model.fit(X_train, Y_Output_Encode_train, validation_data=(X_test,Y_Output_Encode_test), nb_epoch=500, batch_size=50)
#
# scores = model.evaluate(X_train,Y_Output_Encode_train)
# print("\nModel Prediction Accuracy For Train Data: %.2f%%" % (scores[1] * 100))
# print("Baseline Error: %.2f%%" % (100-scores[1]*100))
#
# Y_Predicted =model.predict(X_train)
# y_classes = Y_Predicted.argmax(axis=-1)
# print ("Confusion Matrix")
# matrix=confusion_matrix(Y_trainOriginal, y_classes)
# print(matrix)
#
# scores1 = model.evaluate(X_test,Y_Output_Encode_test)
# print("\nModel Prediction Accuracy For Test Data: %.2f%%" % (scores1[1] * 100))
# print("Baseline Error: %.2f%%" % (100-scores1[1]*100))
#
# Y_Predicted_test =model.predict(X_test)
# y_classes_test = Y_Predicted_test.argmax(axis=-1)
# print ("Confusion Matrix")
# matrix_test=confusion_matrix(Y_testOriginal, y_classes_test)
# print(matrix_test)
#
# f = open(outputPath, 'w')
# f.write("t50_fid,Min,Max,Elev_diff,Area,Perimeter,AreaLengthRatio,NoofNodes,AvgPolygonWidth,classLabel\n")
# for x_data,original,predicted in zip(X_testOriginal,Y_testOriginal,y_classes_test):
#
#     for i in x_data:
#             data = str((i))
#             f.write(data + ",")
#
#     actual=(int)(original)
#     predicted= (int)(predicted)
#
# # Convert Numbers to Unique Class labels Actual
#     if(actual==0):
#         actual ="River"
#     elif(actual==1):
#         actual="Canal"
#     elif (actual == 2):
#         actual = "Lake"
#     elif (actual == 3):
#         actual = "Pond"
#     elif (actual == 4):
#         actual = "ICE"
#     elif (actual == 5):
#         actual = "IsLand"
#     elif (actual == 6):
#         actual = "Lagoon"
#     elif (actual == 7):
#         actual = "Swamp"
#     elif (actual == 8):
#         actual = "Rapid"
#     elif (actual == 9):
#         actual = "Reservoir"
#
# # Convert Numbers to Unique Class labels Predicted
#     if(predicted==0):
#         predicted ="River"
#     elif(predicted==1):
#         predicted="Canal"
#     elif (predicted == 2):
#         predicted = "Lake"
#     elif (predicted == 3):
#         predicted = "Pond"
#     elif (predicted == 4):
#         predicted = "ICE"
#     elif (predicted == 5):
#         predicted = "IsLand"
#     elif (predicted == 6):
#         predicted = "Lagoon"
#     elif (predicted == 7):
#         predicted = "Swamp"
#     elif (predicted == 8):
#         predicted = "Rapid"
#     elif (predicted == 9):
#         predicted = "Reservoir"
#
#     f.write(str((actual))+","+str((predicted))+"\n")
#
#
#
# #The h5py package is a Pythonic interface to the HDF5 binary data format.
# model.save(modelPath)
#
# # #serialize model to JSON
# model_json = model.to_json()
# with open(modelJsonPath, "w") as json_file:
#     json_file.write(model_json)
#
# plt.matshow(matrix)
# plt.title('Confusion matrix')
# plt.colorbar()
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.show()
