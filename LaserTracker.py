import cv2
import numpy as np
import math
import time
import pyautogui


cam = cv2.VideoCapture(0)
flag = True
def nothing(x):
    pass

		
# TACKBAR CREATION
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 600, 400)
# create trackbars for color change

cv2.createTrackbar('Hmin','image',0,255,nothing)
cv2.createTrackbar('Smin','image',0,255,nothing)
cv2.createTrackbar('Vmin','image',0,255,nothing)
cv2.createTrackbar('Hmax','image',0,255,nothing)
cv2.createTrackbar('Smax','image',0,255,nothing)
cv2.createTrackbar('Vmax','image',0,255,nothing)
cv2.createTrackbar('kernelEro','image',0,20,nothing)
cv2.createTrackbar('kernelDia','image',0,20,nothing)
#cv2.createTrackbar('kernelDst','image',0,20,nothing)
cv2.createTrackbar('median','image',0,20,nothing)
cv2.createTrackbar('edgeT1','image',0,400,nothing)
cv2.createTrackbar('edgeT2','image',0,400,nothing)

'''
cv2.createTrackbar('Hmin2','image',0,255,nothing)
cv2.createTrackbar('Smin2','image',0,255,nothing)
cv2.createTrackbar('Vmin2','image',0,255,nothing)
cv2.createTrackbar('Hmax2','image',0,255,nothing)
cv2.createTrackbar('Smax2','image',0,255,nothing)
cv2.createTrackbar('Vmax2','image',0,255,nothing)'''
#cv2.createTrackbar('ero_type','image',0,2,nothing)
#cv2.createTrackbar('ero_size','image',0,100,nothing)
#cv2.createTrackbar('ero_ON/OFF','image',0,1,nothing)
#cv2.createTrackbar('Krnl_size','image',0,50,nothing)
#cv2.createTrackbar('dia_elmt','image',0,255,nothing)
#cv2.createTrackbar('dia_max','image',0,255,nothing)
#cv2.createTrackbar('dia_ON/OFF','image',0,255,nothing)
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'image',0,1,nothing)
# 31,47,62 for green min
#83,255,255 green max
s, init = cam.read()
size = init.shape[0],init.shape[1]
print size
frame_count = 0
start = 0
flag = False
#fgbg = cv2.createBackgroundSubtractorGMG()
cx_prev, cy_prev, dcx, dcy = 0,0,0,0
screenWidth, screenHeight = pyautogui.size()
flag_start = False
	
while(1):

	s, img1 = cam.read()

	#img1 = np.asmatrix(img2)
	#img1.convertTo(img1,CV_8U)
	hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV_FULL)
	img1 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
	hls = cv2.cvtColor(img1, cv2.COLOR_BGR2HLS)
	
	
	lower_red = np.array([cv2.getTrackbarPos('Hmin','image'), cv2.getTrackbarPos('Smin','image'), cv2.getTrackbarPos('Vmin','image')])
	upper_red = np.array([cv2.getTrackbarPos('Hmax','image'), cv2.getTrackbarPos('Smax','image'), cv2.getTrackbarPos('Vmax','image')])
	'''
	lower_red = np.array([234,196,0])
	upper_red = np.array([255,255,255])
	'''
	gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

	gray = np.float32(gray)
	dst = cv2.cornerHarris(gray,2,3,0.04)
	cv2.imshow('dst',dst)
	mask = cv2.inRange(hsv,lower_red,upper_red)

#	res = img1
	'''
	if flag:
		print str(hsv.size)
		flag = False
	size = 960,960
	contrast = np.zeros(size,dtype=np.uint8)
	for y in range(0,img1.rows):
		for x in range(0,img1.cols):
			for c in range(0,2):
				sharpen.at<Vec3b>(y,x)[c] = saturate_cast<uchar>( alpha*( hsv.at<Vec3b>(y,x)[c] ) + beta );
	'''
	'''	
	erosion_type = 0;
	if cv2.getTrackbarPos('ero_type','image') == 0 :
		erosion_type = cv2.MORPH_RECT
	elif cv2.getTrackbarPos('ero_type','image') == 1 :
		erosion_type = cv2.MORPH_CROSS
	else:
		erosion_type = cv2.MORPH_ELLIPSE
	
	
	
	erosion_size = cv2.getTrackbarPos('ero_size','image')
	ele_size = size
	ele_point = erosion_size, erosion_size
	'''
	element = cv2.getStructuringElement(cv2.MORPH_RECT,size)
	'''
	kernel_ero = np.ones((cv2.getTrackbarPos('kernelEro','image'),cv2.getTrackbarPos('kernelEro','image')),dtype=np.uint8)
	kernel_dialate = np.ones((cv2.getTrackbarPos('kernelDia','image'),cv2.getTrackbarPos('kernelDia','image')),dtype=np.uint8)
	'''
	
	kernel_ero = np.ones((1,1),dtype=np.uint8)
	kernel_dialate = np.ones((10,10),dtype=np.uint8)
	
	res = cv2.erode( mask, kernel_ero, element)
	res = cv2.dilate(res, kernel_dialate, element)
	
	
	if cv2.getTrackbarPos('median','image') == 0:
		dst_Size  = 5
	elif cv2.getTrackbarPos('median','image')%2 != 1:
		dst_Size = cv2.getTrackbarPos('median','image')+1
	else:
		dst_Size = cv2.getTrackbarPos('median','image')
	median = cv2.medianBlur(res,dst_Size)
	''' BLUR
	kernel_dst = np.ones(dst_Size,np.float32)/25
	dst_2d = cv2.filter2D(res,-1,kernel_dst)
	'''
	
	
	edge = cv2.Canny(res,190,200)
	
	image, contours, hierarchy = cv2.findContours(edge,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	if flag_start:
		
		if len(contours)!=0: 
			cnt = contours[0]
			M = cv2.moments(cnt)
			if int(M['m00'])!= 0:
				cx = int(M['m10']/M['m00'])
				cy = int(M['m01']/M['m00'])
				dcx = cx - cx_prev
				dcy = cy - cy_prev
				#print "centeroid: %s, %s"%(dcx, dcy)
				cx_prev = cx
				cy_prev = cy
				a=dcy*3
				b=-(dcx*3)
				
				pyautogui.moveRel(b, a)

			
	img1 = cv2.drawContours(img1, contours, -1, (0,255,0), 3)
	#ret,thresh1 = cv2.threshold(hsv,220,255,cv2.THRESH_BINARY)#thresh to light. can try laser 
	#ret,thresh1 = cv2.threshold(hsv,127,255,cv2.THRESH_TRUNC)
	#ret,thresh2 = cv2.threshold(hsv,127,255,cv2.THRESH_TRIANGLE)
	#ret,thresh3 = cv2.threshold(hsv,127,255,cv2.THRESH_OTSU)
	#thresh2 = cv2.adaptiveThreshold(hsv2,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
	#thresh3 = cv2.adaptiveThreshold(hsv2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	#cv2.imshow('img',img1)
	#cv2.imshow('hsv',hsv)
	#cv2.imshow('mask',mask)
	#cv2.imshow('res',res)	
	#cv2.imshow('blur',median)
	#cv2.imshow('thresh1',thresh1)
	#cv2.imshow('thresh2',thresh2)
	#cv2.imshow('edge',edge)
	cv2.imshow('HLS',hls)
	#cv2.imshow('frame',fgmask)	
	#cv2.imshow('contour', contaur)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	elif cv2.waitKey(1) & 0xFF == ord('s'):
		flag_start = True
	
	'''
	elif cv2.waitKey(1) & 0xFF == ord('y'):
		start = time.time()
		time.clock()
		flag = True
		print "started timer"
	if flag:
		if (time.time()-start) < 1:
			frame_count+=1
		else:
			print "frame_count: %s" %(frame_count)
			break
	'''
cam.release()
cv2.destroyAllWindows()








