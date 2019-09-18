def run():
    import cv2
    import pickle
    labels={}
    with open(r'labels.pickle','rb') as f:
        l=pickle.load(f)
        labels={v:k for k,v in l.items()}
    cascPath = r'haarcascade_frontalface_alt2.xml'
    faceCascade = cv2.CascadeClassifier(cascPath)
    recogniser=cv2.face.LBPHFaceRecognizer_create()
    recogniser.read(r"trainer.yml")
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
        )

        # Draw a rectangle around the faces
        # print(faces)
        for (x, y, w, h) in faces:
            roi_g=gray[y:y+h,x:x+w]
            roi=frame[y:y+h,x:x+w]
            id_,conf=recogniser.predict(roi_g)
            if conf>=45 and conf<=85:
                print(labels[id_],conf)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


        #Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
run()
