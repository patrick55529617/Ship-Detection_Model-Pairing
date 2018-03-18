#!/usr/bin/env python
import numpy as np
import cv2
from keras.models import Sequential, load_model
from keras.preprocessing import image

# Turn Image To Gray Level
def Gray_Scale(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_image

# Get Interesting Area
def InterestROI(img):
    roiImg = img[150:190,300:380]
    return roiImg

def ModelPairing(imgroi):
    images = []
    img_array = image.img_to_array(imgroi)
    images.append(img_array)
    data = np.array(images)
    
    prediction = mymodel.predict_classes(data)
    return prediction

if __name__ == '__main__' :
        
    mymodel = load_model('model_CatAndDog_Test.h5')
    
    capture = cv2.VideoCapture('D://testdata//sky02.mp4')
    #cv2.namedWindow('GroundTruth', cv2.WINDOW_NORMAL)
    
    if capture.isOpened():
        while True:   
            ret, img = capture.read()
            if (ret==True):
                
                
                
                imgroi = InterestROI(img)
                Gray = Gray_Scale(img)
                
                prediction = ModelPairing(imgroi)
                
                if (prediction[0]==0):
                    cv2.rectangle(img,(300,150),(380,190), (0,255,0), 3)
                    #cv2.imwrite('ImgTest.jpg',img)
                cv2.imshow('GroundTruth', img)
                cv2.imshow('imgroi', imgroi)
                print(prediction)
            else:
                break
            if cv2.waitKey(20)==27:
                break
        # Save Image
        cv2.imwrite('ImgTest.jpg',img)
    cv2.destroyAllWindows() 
