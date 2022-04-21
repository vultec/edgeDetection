import cv2
import numpy as np
import time
from matplotlib import pyplot as plt
from numba import jit
from collections import deque

@jit(parallel = True)
def pythagoras(horizontal,vertical,h,w):
    h_m = (horizontal[:,:,0] + horizontal[:,:,1] + horizontal[:,:,2] ) / 3
    h_m = h_m.reshape((h,w,1)) / 255
    v_m = (vertical[:,:,0] + vertical[:,:,1] + vertical[:,:,2] ) /3
    v_m = v_m.reshape((h,w,1)) / 255
    diagonal = np.sqrt(np.square(h_m)+np.square(v_m))
    return diagonal


height = 1080
width = 1920

a = np.random.rand(height,width,3)*255
b = np.random.rand(height,width,3)*255

t0 = time.time()
pythagoras(a,b,height,width)
t1 = time.time()
print(t1-t0)

vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH,width)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
length = 540

t0 = time.time()
frame_count = 0

running = True
foundSample = False

lookback = 10
pastGradientSquares = deque([np.zeros((length,length))]*lookback,maxlen=lookback)
decay = np.pow(10**(1/lookback),np.arange(lookback)+1)

while running:
    ret, frame = vid.read()
    h,w = frame.shape[:2]

    horizontal_gradient = (cv2.Sobel(frame, cv2.CV_32F, 1, 0, ksize= - 1) + cv2.Sobel(frame, cv2.CV_32F, 1, 0, ksize= 3)) / 2
    vertical_gradient = (cv2.Sobel(frame, cv2.CV_32F, 0, 1, ksize= - 1) + cv2.Sobel(frame, cv2.CV_32F, 0, 1, ksize= 3)) / 2
    gradient = pythagoras(horizontal_gradient,vertical_gradient,h,w)

    ret,thresh = cv2.threshold(gradient,1,255,cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(image=thresh.astype(np.uint8), mode = cv2.RETR_LIST,method=cv2.CHAIN_APPROX_NONE)
    image_copy = frame.copy()
    cv2.drawContours(image = image_copy, contours = contours, contourIdx = -1, color = (0,255,0), thickness = 2, lineType=cv2.LINE_AA)
    cv2.imshow('None approximation', image_copy)


    maxArea = 32400
    foundQuad = False
    for contour in contours:
        poly = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
        if len(poly) == 4:
            quadArea = cv2.contourArea(poly)
            if quadArea > maxArea:
                maxArea = quadArea
                quad =  poly[:,0,:].astype('float32')
                foundQuad = True

    if foundQuad:
        if not foundSample:
            inds = np.arange(4)
            topTwoIndices = np.argsort(quad[:,0])[:2]
            topLeftIndex = topTwoIndices[np.argmin(quad[topTwoIndices][:,1])]

            sources = quad[(inds+topLeftIndex)%4]
            targets = np.array([[0,0],[0,length],[length,length],[length,0]]).astype('float32')

            perspectiveMatrix = cv2.getPerspectiveTransform(sources,targets)
            square = cv2.warpPerspective(frame,perspectiveMatrix,(length,length))

            foundSample = True

        else:
            indCycles = [inds, (inds+1)%4, (inds+2)%4, (inds+3)%4]
            newSquares = []
            newSquareDeltas = []

            for indCycle in indCycles:
                sources = quad[indCycle]
                perspectiveMatrix = cv2.getPerspectiveTransform(sources,targets)
                newSquare = cv2.warpPerspective(frame,perspectiveMatrix,(length,length))

                kernel = np.ones((11,11),np.float32)/121
                smoothedPreviousSquare = cv2.filter2D(np.mean(previousSquare,axis=2),-1,kernel)
                smoothedPreviousSquare = smoothedPreviousSquare / np.sum(smoothedPreviousSquare**2)
                smoothedNewSquare = cv2.filter2D(np.mean(newSquare,axis=2),-1,kernel)
                smoothedNewSquare = smoothedNewSquare / np.sum(smoothedNewSquare**2)
                newSquareDelta = np.sum(smoothedPreviousSquare*smoothedNewSquare)

                newSquares.append(newSquare)
                newSquareDeltas.append(newSquareDelta)

            square = newSquares[np.argmax(newSquareDeltas)]

            perspectiveMatrix = cv2.getPerspectiveTransform(quad[indCycles[np.argmax(newSquareDeltas)]],targets)

        gradientSquare = cv2.warpPerspective(gradient,perspectiveMatrix,(length,length))
        previousSquare = square

        print(gradientSquare.shape)

        cv2.imshow('Gradient Square',gradientSquare)
        cv2.imshow('Square',square)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False


t1 = time.time()
