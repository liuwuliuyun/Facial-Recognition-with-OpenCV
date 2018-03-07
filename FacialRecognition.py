import cv2
import numpy as np
import os

subjects = ["","Ramiz Raja","Elvis Prisly","Amy Anchor"]

def FacialDetection(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier('XMLFiles/lbpcascade_frontalface.xml')
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.25,minNeighbors=5)
    if len(faces) == 0:
        return None, None
    # assume only one face in one picture
    (x,y,w,h) = faces[0]
    return gray[y:y+w,x:x+h],faces[0]

def PrepareTrainningData(dataFolderPath):
    dirs = os.listdir(dataFolderPath)
    faces = []
    lables = []
    for dirName in dirs:
        if not dirName.startswith('s'):
            continue
        lable = int(dirName.replace('s',''))
        subDirPath = dataFolderPath + '/' + dirName
        subDir = os.listdir(subDirPath)
        for imgName in subDir:
            if imgName.startswith('.'):
                continue
            imgPath = subDirPath + '/' + imgName
            img = cv2.imread(imgPath)
            face, rect = FacialDetection(img)
            if face is not None:
                faces.append(face)
                lables.append(lable)
    return faces, lables

faces, lables = PrepareTrainningData('TrainningData')

faceRecognizer = cv2.face.LBPHFaceRecognizer_create()

faceRecognizer.train(faces,np.array(lables))

def DrawRectangle(img,rect):
    (x,y,w,h) = rect
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

def DrawText(img,text,x,y):
    cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_PLAIN,1.5,(255,0,0),2)

def PredictFace(img):
    image = img.copy()
    face, rect = FacialDetection(image)
    if face is None:
        return None
    lable, confidance= faceRecognizer.predict(face)
    lableText = subjects[lable]
    DrawRectangle(image,rect)
    DrawText(image,lableText,rect[0],rect[1]-6)
    return image

def StartPrediction(testDataFolder,resultDataFolder):
    testImgs = os.listdir(testDataFolder)
    for testImg in testImgs:
        if testImg.startswith('.'):
            continue
        testImgPath = testDataFolder + '/' + testImg
        img = cv2.imread(testImgPath)
        predictImg = PredictFace(img)
        if predictImg is None:
            print("Warning: No face is detected in: "+testImgPath)
            continue
        predictImgPath = resultDataFolder +'/'+'rst_'+testImg
        cv2.imwrite(predictImgPath, predictImg)
    print("Note: Predicting Complete Successfully!")

StartPrediction('TestData','RecognitionResults')