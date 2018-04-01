
import numpy as np
import cv2 as cv

# built-in module
import sys


if __name__ == '__main__':
    print (__doc__)

    try:
        fn = sys.argv[1]
    except:
        fn = 0

    def nothing(*arg):
        pass

    cv.namedWindow('SEEDS')
    cv.createTrackbar('Number of Superpixels', 'SEEDS', 200, 1000, nothing)
    cv.createTrackbar('Iterations', 'SEEDS', 4, 12, nothing)

    seeds = None
    display_mode = 0
    num_superpixels = 200
    prior = 2
    num_levels = 4
    num_histogram_bins = 5

    cap =cv.VideoCapture('D://testdata//sky01.mp4')
    while True:
        flag, img = cap.read()
        converted_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        height,width,channels = converted_img.shape
        num_superpixels_new = cv.getTrackbarPos('Number of Superpixels', 'SEEDS')
        num_iterations = cv.getTrackbarPos('Iterations', 'SEEDS')

        if not seeds or num_superpixels_new != num_superpixels:
            num_superpixels = num_superpixels_new
            seeds = cv.ximgproc.createSuperpixelSEEDS(width, height, channels,
                    num_superpixels, num_levels, prior, num_histogram_bins)
            color_img = np.zeros((height,width,3), np.uint8)
            color_img[:] = (0, 0, 255)

        seeds.iterate(converted_img, num_iterations)
        
        # retrieve the segmentation result
        seedNumber = seeds.getNumberOfSuperpixels()
        
        labels = seeds.getLabels()
        
        mask = seeds.getLabelContourMask(False)
        SuperPixel_img = img
        
        #seedTable[i,0] represents pixel value of the ith seeds
        seedTable = np.zeros((seedNumber, 2), dtype=np.int) 
        
        #VoteTable_SeedNumber[i,0] represents the amount of pixel value 'i' 
        #VoteTable_SeedNumber[i,1] represents the amount of 
        
        VoteTable_SeedNumber = np.zeros((256,2), dtype=np.int)
        
        
        thee = 0 
        for i in range(SuperPixel_img.shape[0]):
            for j in range(SuperPixel_img.shape[1]):
                thee+=1
                if thee%40==0:
                    thee=0
                    temp = labels[i,j]
                    seedTable[temp,0] += converted_img[i,j,0]
                    seedTable[temp,1] += 1
        
        for i in range(seedNumber):
            seedTable[i,0] /= seedTable[i,1]
        
        
        for i in range(height):
            for j in range(width):
                VoteTable_SeedNumber[seedTable[labels[i,j],0],1]+=1
         
        for i in range(seedNumber):
            VoteTable_SeedNumber[seedTable[i,0],0]+=1 


        # stitch foreground & background together
        mask_inv = cv.bitwise_not(mask)
        result_bg = cv.bitwise_and(img, img, mask=mask_inv)
        result_fg = cv.bitwise_and(color_img, color_img, mask=mask)
        result = cv.add(result_bg, result_fg)

        if display_mode == 0:
            cv.imshow('SEEDS', result)
        elif display_mode == 1:
            cv.imshow('SEEDS', mask)
#        else:
#            cv.imshow('SEEDS', labels)

        ch = cv.waitKey(1)
        if ch == 27:
            break
        elif ch == 13: # Press 'Enter' to Stop.
            cv.waitKey(0)
        elif ch & 0xff == ord(' '):
            display_mode = (display_mode + 1) % 3
    cv.destroyAllWindows()