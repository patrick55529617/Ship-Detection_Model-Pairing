# coding:utf8
import cv2
import numpy as np
#from skimage.filters import threshold_otsu
from keras.models import load_model
import time
from keras.preprocessing import image

Module = load_model('D://user//Documents//ShipDetection//ship_model (1).h5')
video = 'D://testdata//test12.mp4'
delay = 13

camera = cv2.VideoCapture(video)
fps = camera.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('D://Training Data//Experiment//test12//Result_Brown.avi',fourcc, fps, (640,360))
out1 = cv2.VideoWriter('D://Training Data//Experiment//test12//Result.avi',fourcc, fps, (640,360))

# test3: 1 3 3 1 5
# test9: 1 3 2 1 5
# test10: 3 3 3 1 5
# test11: 2 4 3 1 4
# test12: 1 3 3 1 4


positive = 0
negative = 0

print(Module.summary())
tStart = time.time()
frames = 0
if (camera.isOpened()):    
    
    while True:
        res, frame = camera.read()
        
        if (res==False): break
        frame = cv2.resize(frame, (640, 360))
        height , width , channel = frame.shape
        frame1 = cv2.medianBlur(frame,5)
        aa = frame.copy()
        imgray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        
        imgray = cv2.erode(imgray, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        imgray = cv2.dilate(imgray, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=3)
    
        th2 = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY_INV,11,3)
        img = cv2.erode(th2, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        th2 = cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=4)
        ret, labels = cv2.connectedComponents(th2)
        
    
#        label_hue = np.uint8(179*labels/np.max(labels))
#        blank_ch = 255*np.ones_like(label_hue)
#        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
#        
#        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
#    
#    # set bg label to black
#        labeled_img[label_hue==0] = 0
    #    
        for index in range(1,np.max(labels)+1):
            N = np.where(labels==index)
    
            if(N[0].shape[0]<1200): continue
            else:
                x = np.min(N[0]) ## 框框最左上角的x座標
                y = np.min(N[1]) ## 框框最左上角的y座標
                w = np.max(N[0]) ## 框框最右下角的x座標
                h = np.max(N[1]) ## 框框最右下角的y座標
                
            ##////////////////////////////////////////////    
            
                area = (w-x)*(h-y)



            ##////////////////////////////////////////////                




#                if(w-x==height-1 or h-y==width-1): continue
#                cv2.rectangle(frame,(y,x),(h,w),(255,255,0),2)
    #            print(str(x)+' '+str(y)+' '+str(w)+' '+str(h))
                
                roiImg = frame[x:w,y:h]
                
                res=cv2.resize(roiImg,(80,80))
                
                images = []
                img_array = image.img_to_array(res)
                images.append(img_array)
                data = np.array(images)
#    #        
                prediction1 = Module.predict_classes(data)
#                
    #            print(prediction1)
                if(prediction1[0]==1):
                    cv2.rectangle(aa,(y,x),(h,w),(0,255,0),2)
                    cv2.rectangle(frame,(y,x),(h,w),(0,255,0),2)
                    positive+=1
#                    st = 'D://Training Data//P//' + str(positive) + '.jpg'
#                    cv2.imwrite(st,roiImg)
                else:
                    cv2.rectangle(frame,(y,x),(h,w),(140,180,210),2)
                    negative+=1
#                    st = 'D://Training Data//N//' + str(negative) + '.jpg'
#                    cv2.imwrite(st,roiImg)
                
    
#        if not res:
#            break
        cv2.imshow("original", frame)
#        if(frames>=43):
        out.write(frame)
        out1.write(aa)
        cv2.imshow("GAUSSIAN_C", th2)
#        cv2.imshow("IMGRAY", imgray)
        st = 'D://Training Data//Experiment//test12//' + str(frames) + '.jpg'
        cv2.imwrite(st,frame)
#        st = 'D://Training Data//Experiment//test12//LABELING_' + str(frames) + '.jpg'
#        cv2.imwrite(st,labeled_img)

        if cv2.waitKey(1)==27:
            break
        frames+=1
        
tEnd = time.time()
print (tEnd - tStart)

print(positive)
print(negative)
camera.release()
out.release()
out1.release()
cv2.destroyAllWindows()
    
