# coding:utf8
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from sklearn.cluster import KMeans


Module = load_model('D://user//Documents//Model//ship_model_0430_rotate.h5')
video = 'D://testdata//test2.mp4'
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
    criteria_RGB = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    if not res:
        break
    imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    Z = frame.reshape((-1,3))
    Z1 = np.float32(Z)
    
    height , width , channel = frame.shape
    
    if(frames==0):
        
        K = 3
        ret,label,center=cv2.kmeans(Z1,K,None,criteria_RGB,3,cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        rest = center[label.flatten()]
        rest2 = rest.reshape((frame.shape))
    
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K1 = 3
    K2 = 2
    ret,label,center=cv2.kmeans(Z1,K1,None,criteria,3,cv2.KMEANS_RANDOM_CENTERS)
    
    center = np.uint8(center)
    rest = center[label.flatten()]
    rest2 = rest.reshape((frame.shape))
    
    location0 = np.where(label==0)
    location1 = np.where(label==1)
    location2 = np.where(label==2)
        
    l0 = np.empty([location0[0].shape[0],2])
    l1 = np.empty([location1[0].shape[0],2])
    l2 = np.empty([location2[0].shape[0],2])
    
    k0=0
    k1=0
    k2=0
    
    for i in location0[0]:
#        print(i)
        l0[k0][0] = i/width
        l0[k0][1] = i%width
        k0+=1
    k0=0
    
    l0 = np.float32(l0)
    ret0,label0,center0=cv2.kmeans(l0,K2,None,criteria,3,cv2.KMEANS_RANDOM_CENTERS)
    
    for i in location1[0]:
#        print(j)
        l1[k1][0] = int(i/width)
        l1[k1][1] = int(i%width)
        k1+=1
    k1=0
    
    for i in location2[0]:
#        print(j)
        l2[k2][0] = int(i/width)
        l2[k2][1] = int(i%width)
        k2+=1
    k2=0
#    
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
            print(prediction1)
            if(prediction1==1):
                print(str(x)+' '+str(y)+' '+str(w)+' '+str(h))
                
                #cv2.imwrite(st,roiImg)
                a+=1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
    if(delay>=0 and cv2.waitKey (delay)>=0):  
        cv2.waitKey(0)       
    if cv2.waitKey(110) & 0xff == 27:
        break
    cv2.imshow("detection", frame)
#    cv2.imshow("back", dilated)
    cv2.imshow("K", rest2)
    
camera.release()
cv2.destroyAllWindows() 

