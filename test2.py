from flask import Flask,render_template,url_for,redirect,request
from deepface import DeepFace
import time
import dlib
import pyautogui
import cv2
import numpy as np
import os
from collections import Counter
import glob

def save(img,name, bbox, width=180,height=227):
    x, y, w, h = bbox
    imgCrop = img[y:h, x: w]
    imgCrop = cv2.resize(imgCrop, (width, height))#we need this line to reshape the images
    cv2.imwrite(name+".jpg", imgCrop)
def MyRec(rgb,x,y,w,h,v=20,color=(200,0,0),thikness =2):
    cv2.line(rgb, (x,y),(x+v,y), color, thikness)
    cv2.line(rgb, (x,y),(x,y+v), color, thikness)
    cv2.line(rgb, (x+w,y),(x+w-v,y), color, thikness)
    cv2.line(rgb, (x+w,y),(x+w,y+v), color, thikness)
    cv2.line(rgb, (x,y+h),(x,y+h-v), color, thikness)
    cv2.line(rgb, (x,y+h),(x+v,y+h), color, thikness)
    cv2.line(rgb, (x+w,y+h),(x+w,y+h-v), color, thikness)
    cv2.line(rgb, (x+w,y+h),(x+w-v,y+h), color, thikness)
def faces(image,new_path):
    face_array=[]
    frame =cv2.imread(image)
    gray =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    faces = detector(gray)
    fit =40
    for counter,face in enumerate(faces):
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        cv2.rectangle(frame,(x1,y1),(x2,y2),(220,255,220),1)
        MyRec(frame, x1, y1, x2 - x1, y2 - y1, 10, (0,250,0), 3)
        face_array.append(str(counter)+".jpg")
        save(gray,new_path+str(counter),(x1,y1,x2,y2))
    print("done saving")
    return face_array
app=Flask(__name__)

@app.after_request
def add_header(response):
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response
@app.route('/')
def data():
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.config["CACHE_TYPE"] = "null"
    detector = dlib.get_frontal_face_detector()
    new_path ='F:/mini_project_ml/static/image/'
    files = glob.glob('F:/mini_project_ml/static/image/*')
    for f in files:
        os.remove(f)
    print("Taking photo")
    time.sleep(10)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    images1 = pyautogui.screenshot()
    print("Taken screenshort")
    images1 = cv2.cvtColor(np.array(images1), cv2.COLOR_RGB2BGR)
    k="image"+str(timestr)+".png"
    cv2.imwrite(k, images1)
    face_array=faces(k,new_path)
    path="F:\\mini_project_ml\\static\\image"
    f_data={}
    facial_data=[]
    for img1 in os.listdir(path):
        imgpath = os.path.join(path,img1)
        print(imgpath,img1)
        obj = DeepFace.analyze(img_path = imgpath, actions = ['emotion'],enforce_detection=False)
        c=obj["dominant_emotion"]
        print(c)
        f_data[img1]=c
        facial_data.append(c)
    duplicate_dict = dict(Counter(facial_data))
    return render_template('data.html',form_data=f_data,duplicate_dict=duplicate_dict)
    #return render_template('popup.html')
if __name__=='__main__':
    app.run()