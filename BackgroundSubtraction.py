# coding:utf8
import cv2
import numpy as np
from keras.models import Sequential, load_model
from keras.preprocessing import image

def detect_video(video):
    
    Module = load_model('ship_model.h5')
    
    camera = cv2.VideoCapture(video)
    history = 20

    frames = 0

    while True:
        res, frame = camera.read()

        if not res:
            break
        imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
          
        if frames < history:
            frames += 1
            continue

        #th = cv2.threshold(imgray, 200, 255, cv2.THRESH_BINARY)[1] # OTSU
        th = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=3)
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=3)
        
        imaged, contours, hier = cv2.findContours(th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            print(area)
            if 5000 <area < 100000:
                roiImg = frame[y:(y+h+40),x:(x+w+40)]
                res=cv2.resize(roiImg,(120,60))
                #print(roiImg)
                images = []
                img_array = image.img_to_array(res)
                images.append(img_array)
                data = np.array(images)
                
                
                #prediction = Module.predict_classes(data)
                #print(prediction)
                #if(prediction==1):
                #    cv2.imwrite('ImgTest.jpg',roiImg)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                
                
        if cv2.waitKey(110) & 0xff == 27:
            break
        cv2.imshow("detection", frame)
        cv2.imshow("back", dilated)
        
    camera.release()
    cv2.destroyAllWindows() 

if __name__ == '__main__':
    video = 'D://testdata//sky01.mp4'
    detect_video(video)
