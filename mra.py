from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import pywt
import cv2

%matplotlib inline 

# function that generates Haar filters for different scales

def generateHaar(scale):
    return np.ones((scale, scale))/float(scale*scale)

def cellDecomp(A, size, power, s_0, s_1):

	# Generates various images of different resolutions

	mraImages = np.zeros((size, size, power+1), dtype=np.uint8)

	for i in range(0, power+1):
	    kernel = generateHaar(2**i)
	    mraImages[:, :, i] = cv2.filter2D(A[:, :], -1, kernel)

	maxRight = size-1
	maxLeft = 0
	maxTop = 0
	maxBottom = size-1

	minResolution = (power + 1)/2

	finalImage = np.zeros((size, size), dtype=np.uint8)
	nodesImage = np.zeros((size, size, 2), dtype=np.uint8)
	Dict = {}
	s_0 = 100
	s_1 = 25

	# Layer 1
	left = max(maxLeft, s_1-1)
	right = min(maxRight, s_1+2)
	bottom = min(maxBottom, s_0+1)
	top = max(maxTop, s_0-2)

	print ("\n left: ", left, "\n right: ", right, "\n bottom: ", bottom, "\n top: ", top)

	nodeNum = 0

	for x in range(left, right+1):
	    for y in range(top, bottom+1):
	        nodeNum += 1
	        Dict[nodeNum] = [y, x, 1]
	        finalImage[y][x] = mraImages[y][x][0]
	        nodesImage[y][x][0] = mraImages[y][x][0]
	        nodesImage[y][x][1] = nodeNum
	        

	leftInner = left
	rightInner = right
	topInner = top
	bottomInner = bottom 

	plt.imshow(finalImage, interpolation='nearest')
	plt.show()

	# Further Layers

	currResolution = 1
	leftInnerTrue = s_1 - 1
	rightInnerTrue = s_1 + 2
	bottomInnerTrue = s_0 + 1
	topInnerTrue = s_0 - 2

	leftOuter = max(maxLeft, leftInner - (2**currResolution))
	rightOuter = min(maxRight, rightInner + (2**currResolution))
	topOuter = max(maxTop, topInner - (2**currResolution))
	bottomOuter = min(maxBottom, bottomInner + (2**currResolution))

	leftOuterTrue = leftInnerTrue - (2**currResolution)
	rightOuterTrue = rightInnerTrue + (2**currResolution)
	topOuterTrue = topInnerTrue - (2**currResolution)
	bottomOuterTrue = bottomInnerTrue + (2**currResolution)

	lefti = righti = bottomi = topi = 0

	direction = 0

	x = leftOuter
	y = topOuter

	k = 0

	while (leftInner != maxLeft or rightInner != maxRight or topInner != maxTop or bottomInner != maxBottom):
	    
	    if (direction == 4): 
	        
	        k += 1
	        
	        # Transferring Outer to Inner in order to create a new Outer
	        leftInner = leftOuter
	        rightInner = rightOuter
	        bottomInner = bottomOuter
	        topInner = topOuter
	        
	        leftInnerTrue = leftOuterTrue
	        rightInnerTrue = rightOuterTrue
	        bottomInnerTrue = bottomOuterTrue
	        topInnerTrue = topOuterTrue
	                
	        # Increase or clip the currentResolution 
	        currResolution = min(int(minResolution), currResolution + 1)
	        
	        leftOuter = max(maxLeft, leftInner - (2**currResolution))
	        rightOuter = min(maxRight, rightInner + (2**currResolution))
	        topOuter = max(maxTop, topInner - (2**currResolution))
	        bottomOuter = min(maxBottom, bottomInner + (2**currResolution))
	        
	        leftOuterTrue = leftInnerTrue - (2**currResolution)
	        rightOuterTrue = rightInnerTrue + (2**currResolution)
	        topOuterTrue = topInnerTrue - (2**currResolution)
	        bottomOuterTrue = bottomInnerTrue + (2**currResolution)
	        
	        for i in range(0, size):
	            if (leftOuterTrue + i*(2**currResolution) > 0):
	                lefti = i
	                break
	                
	        for i in range(0, size):
	            if (rightOuterTrue - i*(2**currResolution) < size-1):
	                righti = i
	                break
	                
	        for i in range(0, size):
	            if (bottomOuterTrue - i*(2**currResolution) < size-1):
	                bottomi = i
	                break
	                
	        for i in range(0, size):
	            if (topOuterTrue + i*(2**currResolution) > 0):
	                topi = i
	                break
	        
	        print ("\n bottomOuter: ", bottomOuter)
	        
	        direction = 0
	        
	        x = leftOuter
	        y = topOuter
	        
	    elif (direction == 0): 
	        
	        if (topInner == topOuter or x >= rightOuter):
	            x = rightOuter
	            y = topOuter
	            direction = 1
	            print ("New direction")
	            continue
	        
	        if (x == leftOuterTrue and x != 0):
	            lefti += 1
	        
	        a = min(rightOuter, leftOuterTrue + lefti*(2**currResolution) - 1)
	        print (a)
	        xRep = int((x+a)/2)
	        distX = float(x+a)/2
	        b = topInner - 1 
	        distY = float(y+b)/2
	        yRep = int((y+b)/2)
	        
	        pixelSize = (a+1-x)*(b+1-y)
	        
	        nodeNum += 1
	        
	        # copy the values from mra currResolution
	        for p in range(int(x), int(a+1)):
	            for q in range(int(y), int(b+1)):
	#                 print (p, q)
	                Dict[nodeNum] = [distY, distX, size]
	                nodesImage[q][p][0] = mraImages[yRep][xRep][currResolution]
	                nodesImage[q][p][1] = nodeNum
	                finalImage[q][p] = mraImages[yRep][xRep][currResolution]
	        
	        lefti += 1
	        
	        x = a + 1
	        
	    elif (direction == 1): 
	                
	        if (rightInner == rightOuter or y >= bottomOuter):
	            x = rightOuter
	            y = bottomOuter
	            direction = 2
	            print ("New direction")
	            continue
	        
	        if (y == topOuterTrue and y != 0):
	            topi += 1
	        
	        a = rightInner + 1
	        distX = float(x+a)/2
	        xRep = int((x+a)/2)
	        b = min(bottomOuter, topOuterTrue + topi*(2**currResolution) - 1)
	        distY = float(y+b)/2
	        yRep = int((y+b)/2)
	        
	        pixelSize = (x+1-a)*(b+1-y)
	        
	        nodeNum += 1
	        
	        # copy the values from mra currResolution
	        for p in range(int(a), int(x+1)):
	            for q in range(int(y), int(b+1)):
	#                 print (p, q)
	                Dict[nodeNum] = [distY, distX, size]
	                nodesImage[q][p][0] = mraImages[yRep][xRep][currResolution]
	                nodesImage[q][p][1] = nodeNum
	                finalImage[q][p] = mraImages[yRep][xRep][currResolution]
	        
	        topi += 1
	        
	        y = b + 1
	        
	    elif (direction == 2): 
	                
	        if (bottomInner == bottomOuter or x <= leftOuter):
	            x = leftOuter
	            y = bottomOuter
	            direction = 3
	            print ("New direction")
	            continue
	        
	        if (x == rightOuterTrue and x != size-1):
	            righti += 1
	        
	        a = max(leftOuter, rightOuterTrue - righti*(2**currResolution) + 1)
	        distX = float(x+a)/2
	        xRep = int((x+a)/2)
	        b = bottomInner + 1
	        distY = float(y+b)/2
	        yRep = int((y+b)/2)
	        
	        pixelSize = (x+1-a)*(y+1-b)
	        
	        nodeNum += 1
	        
	        # copy the values from mra currResolution
	        for p in range(int(a), int(x+1)):
	            for q in range(int(b), int(y+1)):
	#                 print (p, q)
	                Dict[nodeNum] = [distY, distX, size]
	                nodesImage[q][p][0] = mraImages[yRep][xRep][currResolution]
	                nodesImage[q][p][1] = nodeNum
	                finalImage[q][p] = mraImages[yRep][xRep][currResolution]

	        righti += 1
	        
	        x = a - 1
	        
	    elif (direction == 3): 
	#         print ("Done 1")
	                
	        if (leftInner == leftOuter or y <= topOuter):
	            direction = 4
	            print ("New layer")
	            continue
	        
	        if (y == bottomOuterTrue and y != size-1):
	            bottomi += 1
	        
	        a = leftInner - 1
	        distX = float(x+a)/2
	        xRep = int((x+a)/2)
	        b = max(topOuter, bottomOuterTrue - bottomi*(2**currResolution) + 1)
	        distY = float(y+b)/2
	        yRep = int((y+b)/2)
	        
	        pixelSize = (a+1-x)*(y+1-b)
	        
	        nodeNum += 1
	        
	        print (y)
	        
	        # copy the values from mra currResolution
	        for p in range(int(x), int(a+1)):
	            for q in range(int(b), int(y+1)):
	#                 print (p, q)
	                Dict[nodeNum] = [distY, distX, size]
	                nodesImage[q][p][0] = mraImages[yRep][xRep][currResolution]
	                nodesImage[q][p][1] = nodeNum
	                finalImage[q][p] = mraImages[yRep][xRep][currResolution]
	        
	        bottomi += 1
	        
	        y = b - 1
	        
	#         plt.imshow(finalImage, interpolation='nearest')
	#         plt.show()


	plt.imshow(finalImage, interpolation='nearest')
	plt.show()

	return nodesImage, Dict

if __name__ == '__main__':
	
	size = 128
	power = 7
	w,h = size, size
	t = (w,h)
	A = np.random.randint(0, 255, t, dtype=np.uint8)
	plt.imshow(A, interpolation='nearest')
	plt.show()

	start_point = np.random.randint(1, size-1, (1, 2))
	print ("Start Point: \n", start_point)

	s_0 = start_point[0][0]
	s_1 = start_point[0][1]

	# a, b = cellDecomp(A, size, power, s_0, s_1)


