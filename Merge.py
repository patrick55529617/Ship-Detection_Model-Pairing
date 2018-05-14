# coding:utf8
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from sklearn.cluster import KMeans,DBSCAN


#Module = load_model('D://user//Documents//Model//ship_model_0430_rotate.h5')
video = 'D://testdata//test3.mp4'
delay = 13

a=0

camera = cv2.VideoCapture(video)
fps = camera.get(cv2.CAP_PROP_FPS)
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
    
    K=3
    
    height , width , channel = frame.shape
    
    if (frames==0):
        kmeans_RGB = KMeans(n_clusters=K,random_state=0).fit(Z1)
        center_RGB = kmeans_RGB.cluster_centers_
        center_RGB = np.uint8(center_RGB)
        label = kmeans_RGB.labels_
        res = center_RGB[label.flatten()]
        res2 = res.reshape((frame.shape))
        res3 = cv2.erode(res2, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)
        res3 = cv2.dilate(res3, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
       
        
#        score = list[K]
        
        k0=0
        for i in range(K):
            location = np.where(label==i)
            l = np.empty([location[0].shape[0],2])
            for x in location[0]:
                l[k0][0] = x/width
                l[k0][1] = x%width
                k0+=1
            k0=0
            l=l.astype(np.int32)
            kmeans_Coordinate = KMeans(n_clusters=2,random_state=0).fit(l)
            center_Coordinate = np.uint8(kmeans_Coordinate.cluster_centers_)
            label_Coordinate = kmeans_Coordinate.labels_
            c0 = np.where(label_Coordinate==1)
            
#            cv2.rectangle(frame, center_Coordinate, (x + w, y + h), (0, 255, 0), 2)
            min_X = height+1
            min_Y = width+1
            Max_X = 0
            Max_Y = 0
            for y in c0[0]:
                if(min_X>l[y][0]):min_X=l[y][0]
                if(Max_X<l[y][0]):Max_X=l[y][0]
                if(min_Y>l[y][1]):min_Y=l[y][1]
                if(Max_Y<l[y][1]):Max_Y=l[y][1]

            print(min_X)
            print(min_Y)
            print(Max_X)
            print(Max_Y)
            
            c1 = np.where(label_Coordinate==1)
            
    else:
        label = kmeans_RGB.predict(Z1)
        res = center_RGB[label.flatten()]
        res2 = res.reshape((frame.shape))
    
#    if frames < history:
#        frames += 1
#        continue
#
#    th = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
#    th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=3)
#    dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)
#
#    imaged, contours, hier = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#
#    for c in contours:
#        x, y, w, h = cv2.boundingRect(c)
#        
#        area = cv2.contourArea(c)
#        
#        if 5000 < area < 50000:
#            roiImg = frame[y:(y+h+40),x:(x+w+40)]
#            res=cv2.resize(roiImg,(60,60))
#            
#            images = []
#            img_array = image.img_to_array(res)
#            images.append(img_array)
#            data = np.array(images)
#        
#            prediction1 = Module.predict_classes(data)
#            print(prediction1)
#            if(prediction1==1):
#                print(str(x)+' '+str(y)+' '+str(w)+' '+str(h))
#                
#                #cv2.imwrite(st,roiImg)
#                a+=1
#                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if(delay>=0 and cv2.waitKey (delay)>=0):  
        cv2.waitKey(0)       
    if cv2.waitKey(110) & 0xff == 27:
        break
    cv2.imshow("detection", frame)
    cv2.imshow("back", res2)
    if(frames==0):
        cv2.imwrite("D://user//Documents//Save//Kmeans.jpg",res2)
    frames+=1
camera.release()
cv2.destroyAllWindows() 

def return_Texture_Complexity():
    return 0

