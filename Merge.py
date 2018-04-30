# coding:utf8
import cv2
import numpy as np
from keras.models import Sequential, load_model
from keras.preprocessing import image

def detect_video(video):
    
    Module = load_model('D://user//Documents//Model//ship_model_0430_rotate.h5')
    
    delay = 13
    
    a=0
    
    camera = cv2.VideoCapture(video)
#    fps = camera.get(cv2.CAP_PROP_FPS)
    history = 20

    frames = 0

    while True:
        res, frame = camera.read()
        
#        print(fps)

        st = 'D://user//Documents//Save//Ship_' + str(a) + '.jpg'

        if not res:
            break
        imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
          
        if frames < history:
            frames += 1
            continue

        th = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=3)
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)
        
        imaged, contours, hier = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            
            area = cv2.contourArea(c)
            
            if 5000 < area < 50000:
                roiImg = frame[y:(y+h+40),x:(x+w+40)]
                res=cv2.resize(roiImg,(60,60))
                
                
                images = []
                img_array = image.img_to_array(res)
                images.append(img_array)
                data = np.array(images)
            
                prediction1 = Module.predict_classes(data)
                
                if(prediction1==1):
                    print(str(x)+' '+str(y)+' '+str(w)+' '+str(h))
                    print(prediction1)
                    cv2.imwrite(st,roiImg)
                    a+=1
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if(delay>=0 and cv2.waitKey (delay)>=0):  
            cv2.waitKey(0)       
        if cv2.waitKey(110) & 0xff == 27:
            break
        cv2.imshow("detection", frame)
        cv2.imshow("back", dilated)
        

        
    camera.release()
    cv2.destroyAllWindows() 

if __name__ == '__main__':
    video = 'D://testdata//sky01.mp4'
    detect_video(video)