import os 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
dir="F:\\mini_project_ml\\train" #converting training data into pickle file
#dir="F:\\mini_project_ml\\test" # converting test data to pickle file

categories=['angry','disgusted','fearful','happy','neutral','sad','surprised']
data=[]
for category in categories:
    path=os.path.join(dir,category)
    label=categories.index(category)

    for img in os.listdir(path):
        imgpath = os.path.join(path,img)
        emotion_img=cv2.imread(imgpath,0)
        # cv2.imshow('image',emotion_img)
        emotion_img=cv2.resize(emotion_img,(50,50))
        image=np.array(emotion_img).flatten()
        data.append([image,label])
       
pick_in =open('data.pickle','wb')
pickle.dump(data,pick_in)
pick_in.close()

print(len(data))       
# cv2.waitKey(0)
# cv2.destroyAllWindows()