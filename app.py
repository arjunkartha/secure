import cv2
import io
from PIL import Image
import numpy as np 
import tensorflow as tf 
from re import DEBUG,sub
from flask import Flask,render_template,request,redirect,send_file,url_for,Response
from werkzeug.utils import secure_filename,send_from_directory
import os
from subprocess import Popen

from ultralytics import YOLO

app=Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/",methods=["GET","POST"])
def predict_img():
    if request.method =="POST":
        f=request.files['file']
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,'uploads',f.filename)
        print("upload folder is",filepath)
        f.save(filepath)
        global imgpath
        predict_img.imgpath=f.filename
        print("printing predict_img",predict_img)

        file_extension=f.filename.rsplit('.',1)[1].lower()

        if file_extension=='jpeg':
            img=cv2.imread(filepath)
            frame=cv2.imencode('.jpeg',cv2.UMat(img))[1].tobytes()

            image=Image.open(io.BytesIO(frame))

            yolo=YOLO("best.pt")
            detections=yolo.predict(image,save=True)
            print (detections)
    return "Uploaded Sucessfully"
        
if __name__ =='__main__':
    app.run(host='0.0.0.0',port=8000)


