# coding:utf8
import cv2
import numpy as np
#from keras.models import load_model
#from keras.preprocessing import image
from sklearn.cluster import KMeans,DBSCAN
from sklearn import preprocessing




def return_Texture_Complexity(img):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("jjj")
    print(np.sum(imgray))
    print(img.shape[0])
    print(img.shape[1])
    mean = np.sum(imgray)/(img.shape[0]*img.shape[1])
    print("kkk")
    print(mean)
    square = imgray-mean
    square = square**2
    variance = np.sum(square)/(img.shape[0]*img.shape[1])
    variance/=100
    
    
    
    return variance

#Module = load_model('D://user//Documents//Model//ship_model_0430_rotate.h5')
video = 'D://testdata//test5.mp4'
delay = 13



camera = cv2.VideoCapture(video)
fps = camera.get(cv2.CAP_PROP_FPS)
history = 20

frames = 0
a=0
while True:
    
    res, frame = camera.read()
#        print(fps)
    criteria_RGB = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    if not res:
        break
    imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    Z = frame.reshape((-1,3))
    Z1 = np.float32(Z)
    
    K=3
    
    height , width , channel = frame.shape
    
    if (frames==0):
        a=0
        kmeans_RGB = KMeans(n_clusters=K,random_state=0).fit(Z1)
        center_RGB = kmeans_RGB.cluster_centers_
        center_RGB = np.uint8(center_RGB)
        label = kmeans_RGB.labels_
        res = center_RGB[label.flatten()]
        res2 = res.reshape((frame.shape))
        
    else:
        label = kmeans_RGB.predict(Z1)
        res = center_RGB[label.flatten()]
        res2 = res.reshape((frame.shape))
    
    res2 = cv2.erode(res2, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=3)
    res2 = cv2.dilate(res2, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=3)
    
    x = cv2.Sobel(res2,cv2.CV_16S,1,0)  
    y = cv2.Sobel(res2,cv2.CV_16S,0,1)  
      
    absX = cv2.convertScaleAbs(x)   # 转回uint8  
    absY = cv2.convertScaleAbs(y)  
      
    edges = cv2.addWeighted(absX,0.5,absY,0.5,0)  
    edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY) 
    imaged, contours, hier = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#    if frames < history:
#        frames += 1
#        continue
#
#    th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=3)
#    dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)
#
#    imaged, contours, hier = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
#        
        area = cv2.contourArea(c)
        print(area)
        if 5000 < area < 50000:
#            roiImg = frame[y:(y+h+40),x:(x+w+40)]
#            res=cv2.resize(roiImg,(60,60))
            
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
            cv2.rectangle(frame, (x, y), (x + w, y + h),   (0, 255, 0), 2)
    if(delay>=0 and cv2.waitKey (delay)>=0):  
        cv2.waitKey(0)       
    if cv2.waitKey(110) & 0xff == 27:
        break
    cv2.imshow("detection", frame)
    cv2.imshow("back", res2)
    cv2.imshow("gray",edges)
    if(frames==0):
        cv2.imwrite("D://user//Documents//Save//Kmeans.jpg",res2)
    frames+=1
camera.release()
cv2.destroyAllWindows() 



