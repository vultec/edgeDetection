import cv2
import numpy as np
import time
from matplotlib import pyplot as plt
from numba import jit



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

a = []
b = []
for i in range(1):
        a.append(np.random.rand(height,width,3)*255)
        b.append(np.random.rand(height,width,3)*255)


times = [time.time()]
for i in range(1):
        pythagoras(a[i],b[i],height,width)
        times.append(time.time())
times = np.array(times)
times = times[1:] - times[:-1]
print(times)






vid = cv2.VideoCapture(0)

t0 = time.time()
frame_count = 0

while(True):
	
	# Capture the video frame by frame
	ret, frame = vid.read()

	h,w = frame.shape[:2]

	# Canny filter
	#canny = cv2.Canny(frame,200,100)

	# Sobel+Scharr filter
	horizontal = (cv2.Sobel(frame, cv2.CV_32F, 1, 0, ksize= - 1) + cv2.Sobel(frame, cv2.CV_32F, 1, 0, ksize= 3)) / 2
	vertical = (cv2.Sobel(frame, cv2.CV_32F, 0, 1, ksize= - 1) + cv2.Sobel(frame, cv2.CV_32F, 0, 1, ksize= 3)) / 2

	#h_m = np.mean(horizontal,axis=2).reshape((h,w,1)) / 255
	#v_m = np.mean(vertical,axis=2).reshape((h,w,1)) / 255
	#sobel = np.sqrt(np.square(h_m)+np.square(v_m))

	sobel = pythagoras(horizontal,vertical,h,w)

	# Display
	cv2.imshow('sobel', sobel)
	#cv2.imshow('canny', c)
	#cv2.imshow('frame', frame)


	frame_count += 1

        # Press q to end
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

t1 = time.time()

print('FPS: ' + str(frame_count/(t1-t0)))
print('ms/frame:' + str(1000*(t1-t0)/frame_count))

vid.release()
cv2.destroyAllWindows()

