import face_detect
import face_train
import os
import cv2
import time


vf=cv2.VideoCapture(0)
flag=0


print("WELCOME To the Face Recognition AI Development Programme\n")
ch=int(input('1. Set up Face AI\n2. Test the Face AI\nEnter your choice: '))


if ch==1:
    name=input('Enter your name: ')
    print('Hello {0}, Click Your 10 photos By pressing c'.format(name))
    clk=0
    i=0
    try:
        os.mkdir("images//"+name)
    except FileExistsError:
        print("Directory Already exists")
        flag=1
    while True:
        ret,frame=vf.read()
        path="images//"+name+"//"+str(i)+".png"
        cv2.imshow('Press c',frame)
        key=cv2.waitKey(1)
        if  key == ord('c'):
            cv2.imwrite(path, frame)
            clk+=1
        elif key == ord('q') or clk>=5:
            break
        i+=1
    vf.release()
    cv2.destroyAllWindows()
    face_train.train()

elif ch==2:
    face_detect.run()

else:
    print('Invalid choice')