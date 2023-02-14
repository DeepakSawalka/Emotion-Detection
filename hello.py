import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten,Dense,Dropout,BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from tensorflow.keras.applications import VGG16, InceptionResNetV2
from keras import regularizers
from tensorflow.keras.optimizers import Adam,RMSprop,SGD,Adamax
import tensorflow as tf
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import cv2
import pyautogui
import dlib
from flask import Flask,render_template,url_for,redirect,request
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from collections import Counter
opt = Options()
opt.add_argument('--disable-blink-features=AutomationControlled')
opt.add_argument('--start-maximized')
opt.add_experimental_option("prefs", {
    "profile.default_content_setting_values.media_stream_mic": 1,
    "profile.default_content_setting_values.media_stream_camera": 1,
    "profile.default_content_setting_values.geolocation": 0,
    "profile.default_content_setting_values.notifications": 1
        })
driver = webdriver.Chrome(options=opt)
new_path ='F:/mini_project_ml/image/'
def Glogin(mail_address, password):
    # Login Page
    driver.get(
        'https://accounts.google.com/ServiceLogin?hl=en&passive=true&continue=https://www.google.com/&ec=GAZAAQ')
  
    # input Gmail
    driver.find_element_by_id("identifierId").send_keys(mail_address)
    driver.find_element_by_id("identifierNext").click()
    driver.implicitly_wait(10)
  
    # input Password
    driver.find_element_by_xpath(
        '//*[@id="password"]/div[1]/div/div[1]/input').send_keys(password)
    driver.implicitly_wait(10)
    driver.find_element_by_id("passwordNext").click()
    driver.implicitly_wait(10)
  
    # go to google home page
    driver.get('https://google.com/')
    driver.implicitly_wait(100)
def turnOffMicCam():
    # turn off Microphone
    time.sleep(2)
    driver.find_element_by_xpath(
        '//*[@id="yDmH0d"]/c-wiz/div/div/div[8]/div[3]/div/div/div[2]/div/div[1]/div[1]/div[1]/div/div[4]/div[1]/div/div/div').click()
    driver.implicitly_wait(3000)
  
    # turn off camera
    time.sleep(1)
    driver.find_element_by_xpath('//*[@id="yDmH0d"]/c-wiz/div/div/div[8]/div[3]/div/div/div[2]/div/div[1]/div[1]/div[1]/div/div[4]/div[2]/div/div').click()
    driver.implicitly_wait(3000)
def joinNow():
    # Join meet
    print(1)
    time.sleep(5)
    driver.implicitly_wait(2000)
    driver.find_element_by_css_selector(
        'div.uArJ5e.UQuaGc.Y5sE8d.uyXBBb.xKiqt').click()
    print(1)
def AskToJoin():
    # Ask to Join meet
    time.sleep(20)
    driver.implicitly_wait(2000)
    driver.find_element_by_css_selector(
        'div.uArJ5e.UQuaGc.Y5sE8d.uyXBBb.xKiqt').click()
def MyRec(rgb,x,y,w,h,v=20,color=(200,0,0),thikness =2):
    """To draw stylish rectangle around the objects"""
    cv2.line(rgb, (x,y),(x+v,y), color, thikness)
    cv2.line(rgb, (x,y),(x,y+v), color, thikness)

    cv2.line(rgb, (x+w,y),(x+w-v,y), color, thikness)
    cv2.line(rgb, (x+w,y),(x+w,y+v), color, thikness)

    cv2.line(rgb, (x,y+h),(x,y+h-v), color, thikness)
    cv2.line(rgb, (x,y+h),(x+v,y+h), color, thikness)

    cv2.line(rgb, (x+w,y+h),(x+w,y+h-v), color, thikness)
    cv2.line(rgb, (x+w,y+h),(x+w-v,y+h), color, thikness)

def save(img,name, bbox, width=180,height=227):
    x, y, w, h = bbox
    imgCrop = img[y:h, x: w]
    imgCrop = cv2.resize(imgCrop, (width, height))#we need this line to reshape the images
    cv2.imwrite(name+".jpg", imgCrop)
def faces(image,new_path):
    face_array=[]
    frame =cv2.imread(image)
    gray =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    faces = detector(gray)
    fit =40
    for counter,face in enumerate(faces):
        print(counter)
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        cv2.rectangle(frame,(x1,y1),(x2,y2),(220,255,220),1)
        MyRec(frame, x1, y1, x2 - x1, y2 - y1, 10, (0,250,0), 3)
        face_array.append(str(counter)+".jpg")
        save(gray,new_path+str(counter),(x1,y1,x2,y2))
    print("done saving")
    return face_array
app=Flask(__name__)

@app.route('/')
def form():
    return render_template('index.html')
@app.route('/data/',methods=['POST','GET'])
def data():
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/form' to submit form"
    if request.method=='POST':
        form_data = request.form
        detector = dlib.get_frontal_face_detector()
        new_path ='F:/mini_project_ml/static/image/'
        mail_address=form_data["email_id"]
        password=form_data["password_id"]
        meet=form_data["meet_link"]
        print(mail_address,password,meet)
        Glogin(mail_address, password)
        driver.get(meet)
        joinNow()
        print("Taking photo")
        time.sleep(10)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        images1 = pyautogui.screenshot()
        print("Taken screenshort")
        images1 = cv2.cvtColor(np.array(images1), cv2.COLOR_RGB2BGR)
        k="image"+str(timestr)+".png"
        cv2.imwrite(k, images1)
        face_array=faces(k,new_path)
        driver.quit()
        label_dict = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}
        model = tf.keras.models.load_model('model_optimal.h5')
        model.load_weights('model_weights.h5')
        path="F:\\mini_project_ml\\static\\image"
        facial_data=[]
        for img1 in os.listdir(path):
            imgpath = os.path.join(path,img1)
            img = image.load_img(imgpath,target_size = (48,48),color_mode = "grayscale")
            img = np.array(img)
            #plt.imshow(img)
            #print(img.shape) #prints (48,48) that is the shape of our image
            img = np.expand_dims(img,axis = 0) #makes image shape (1,48,48)
            img = img.reshape(1,48,48,1)
            result = model.predict(img)
            result = list(result[0])
            #print(result)
            img_index = result.index(max(result))
            facial_data.append(label_dict[img_index])
            print(label_dict[img_index])
            #plt.show()
        f_data={}
        for i in range(len(facial_data)):
            f_data[face_array[i]]=facial_data[i]
        duplicate_dict = dict(Counter(facial_data))
        return render_template('data.html',form_data=f_data,duplicate_dict=duplicate_dict)
    #return render_template('popup.html')
if __name__=='__main__':
    app.run()
