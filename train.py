import os 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC



# training data loading from data1
pick_in=open('data.pickle','rb')
data=pickle.load(pick_in)
pick_in.close()

random.shuffle(data)
features=[]
labels=[]

for feature,label in data:
    features.append(feature)
    labels.append(label)


# training data loading from data2
pick_in1=open('data2.pickle','rb')
data2=pickle.load(pick_in1)
pick_in1.close()

random.shuffle(data2)
features1=[]
labels1=[]
for feature,label in data2:
    features1.append(feature)
    labels1.append(label)


# train and test data splitting
xtrain,ytrain=features,labels
xtest,ytest=features1,labels1
model =SVC(C=1,kernel='poly',gamma='auto')
model.fit(xtrain,ytrain)

pick=open('model4.sav','wb')
pickle.dump(model,pick)
pick.close()
# categories=['angry','disgusted','fearful','happy','neutral','sad','surprised']
# prediction=model.predict(xtest)
# accuracy=model.score(xtest,ytest)

# print('Accuracy:',accuracy)

# print('Prediction is : ',categories[prediction[0]])

# emotion=xtest[0].reshape(50,50)
# plt.show(emotion,cmap='gray')
# plt.show()
#3h 35m 17s
