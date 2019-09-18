import os
import numpy as np
from PIL import Image
import cv2
import pickle


def train():
    BASE_DIR=os.path.dirname(os.path.abspath(__file__))
    IMG_DIR=os.path.join(BASE_DIR,'images')
    cascPath = r'haarcascade_frontalface_alt2.xml'
    faceCascade = cv2.CascadeClassifier(cascPath)
    recogniser=cv2.face.LBPHFaceRecognizer_create()
    x_img=[]
    y_labels=[]
    label_id={}
    count=-1
    for root,dirs,files in os.walk(IMG_DIR):
        for file in files:
            if file.endswith('png') or file.endswith('jpeg') or file.endswith('jpg'):
                path=os.path.join(root,file)
                label=os.path.basename(root)
                if label not in label_id:
                    count += 1
                    label_id[label]=count
                img=Image.open(path).convert("L")
                img_arr=np.array(img,"uint8")
                face=faceCascade.detectMultiScale(img_arr,scaleFactor=1.5,minNeighbors=5)
                for x,y,w,h in face:
                    roi=img_arr[y:y+h,x:x+w]
                    x_img.append(roi)
                    print(path)
                    y_labels.append(count)
    print(label_id)
    with open(r"labels.pickle",'wb') as f:
        pickle.dump(label_id,f)
    recogniser.train(x_img,np.array(y_labels))
    recogniser.save(r"trainer.yml")
train()
