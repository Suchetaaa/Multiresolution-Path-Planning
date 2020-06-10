"""
Iterative multiresolution map generation

Author: Sucheta Ravikanti
Date: 07 June, 2020
"""
# Import libraries
from __future__ import division, print_function, absolute_import
import numpy as np
from matplotlib import pyplot as plt
import cv2
from numpy.dual import eig
from scipy.special import comb
from scipy import linspace, pi, exp
from scipy.signal import convolve
import scipy.stats as st
import math


def Daubechies(points):
    v1 = np.array([0.2854])
    v2 = np.array([0.6142, -0.0436])
    v3 = np.array([0.7834, 0.4455, -0.0865, -0.0008])
    v4 = np.array([0.5353, 1.0320, 0.8207, 0.0681, -
                   0.1900, 0.0174, -0.0015, 0])
    v5 = np.array([0.3657, 0.7049, 0.9372, 1.1267, 1.1519, 0.4895, 0.1467, -
                   0.0105, -0.2238, -0.1561, 0.0034, 0.0313, -0.0033, 0.0003, 0, 0])

    if (points == 2):
        v2 = v2.reshape((2, 1))
        out = np.matmul(v2, np.transpose(v2))
        return out / out.sum()
    elif (points == 4):
        v3 = v3.reshape((4, 1))
        out = np.matmul(v3, np.transpose(v3))
        return out / out.sum()
    elif (points == 8):
        v4 = v4.reshape((8, 1))
        out = np.matmul(v4, np.transpose(v4))
        return out / out.sum()
    elif (points == 16):
        v5 = v5.reshape((16, 1))
        out = np.matmul(v5, np.transpose(v5))
        return out / out.sum()
    else:
        return np.ones((points, points))


def Gaussian(points):

    nsig = 1.0
    x = np.linspace(-nsig, nsig, points + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    g = kern2d / kern2d.sum()
    # plt.imshow(g, interpolation='nearest')
    # plt.show()
    return g


def generateHaar(scale):
    return np.ones((scale, scale)) / float(scale * scale)


def ricker(points, a):

    A = 2 / (np.sqrt(3 * a) * (np.pi**0.25))
    wsq = a**2
    vec = np.arange(0, points) - (points - 1.0) / 2
    tsq = vec**2
    mod = (1 - tsq / wsq)
    gauss = np.exp(-tsq / (2 * wsq))
    total = A * mod * gauss
    total = np.array(total)
    outerProduct = np.matmul(total, np.transpose(total))
    return float(1 / np.sum(outerProduct)) * outerProduct


def genWavelets(wavelet, size, scale):
    if (wavelet == 'haar'):
        return generateHaar(scale)
    elif (wavelet == 'mexicanhat'):
        return ricker(size, scale)
    elif (wavelet == 'gaussian'):
        return Gaussian(size)
    elif (wavelet == 'daubechies'):
        return Daubechies(size)


def cellDecomp(A, s_0, s_1):

    # Generates various images of different resolutions
    size, _ = np.shape(A)
    power = int(math.log(size, 2))

    # Haar

    # mraImages = np.zeros((size, size, power+1), dtype=np.uint8)
    # print (np.shape(mraImages))
    # for i in range(0, power+1):
    #     kernel = genWavelets('haar', 2**i, 2**i)
    #     mraImages[:, :, i] = cv2.filter2D(A[:, :], -1, kernel)

    # Daubechies Wavelet

    # mraImages = np.zeros((size, size, power+1), dtype=np.uint8)
    # mraImages[:, :, 0] = A[:, :]
    # # print (np.shape(mraImages))
    # for i in range(1, power+1):
    #     kernel = genWavelets('daubechies', 2**i, 2**i)
    #     mraImages[:, :, i] = cv2.filter2D(A[:, :], -1, kernel)

    # Mexican Hat

    # mraImages = np.zeros((size, size, power+1), dtype=np.uint8)
    # mraImages[:, :, 0] = A[:, :]
    # print (np.shape(mraImages))
    # for i in range(1, power+1):
    #     kernel = genWavelets('mexicanhat', 2**i, i)
    #     mraImages[:, :, i] = cv2.filter2D(A[:, :], -1, kernel)

    # Gaussian Wavelet

    mraImages = np.zeros((size, size, power + 1), dtype=np.uint8)
    mraImages[:, :, 0] = A[:, :]
    # print (np.shape(mraImages))
    for i in range(1, power + 1):
        kernel = genWavelets('gaussian', 2**i, 2**i)
        mraImages[:, :, i] = cv2.filter2D(A[:, :], -1, kernel)

    maxRight = size - 1
    maxLeft = 0
    maxTop = 0
    maxBottom = size - 1

    minResolution = (power + 1) / 2

    finalImage = np.zeros((size, size), dtype=np.uint8)
    nodesImage = np.zeros((size, size, 2), dtype=np.uint32)
    Dict = {}

    # Layer 1
    leftInnerTrue = s_1 - 3
    rightInnerTrue = s_1 + 4
    bottomInnerTrue = s_0 + 3
    topInnerTrue = s_0 - 4

    left = max(maxLeft, s_1 - 3)
    right = min(maxRight, s_1 + 4)
    bottom = min(maxBottom, s_0 + 3)
    top = max(maxTop, s_0 - 4)

    # print ("\n left: ", left, "\n right: ", right, "\n bottom: ", bottom, "\n top: ", top)

    nodeNum = 0

    for x in range(left, right + 1):
        for y in range(top, bottom + 1):
            nodeNum += 1
            Dict[nodeNum] = [y, x, 1]
            finalImage[y][x] = mraImages[y][x][0]
            nodesImage[y][x][0] = mraImages[y][x][0]
            nodesImage[y][x][1] = nodeNum

    leftInner = left
    rightInner = right
    topInner = top
    bottomInner = bottom

    # plt.imshow(finalImage, interpolation='nearest')
    # plt.show()

    # Further Layers

    currResolution = 1
    repeat = 0

    leftInner = left
    rightInner = right
    topInner = top
    bottomInner = bottom

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

            repeat = (repeat + 1) % 2

            if (repeat == 0):
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
                if (leftOuterTrue + i * (2**currResolution) > 0):
                    lefti = i
                    break

            for i in range(0, size):
                if (rightOuterTrue - i * (2**currResolution) < size - 1):
                    righti = i
                    break

            for i in range(0, size):
                if (bottomOuterTrue - i * (2**currResolution) < size - 1):
                    bottomi = i
                    break

            for i in range(0, size):
                if (topOuterTrue + i * (2**currResolution) > 0):
                    topi = i
                    break

            # print ("\n bottomOuter: ", bottomOuter)

            direction = 0

            x = leftOuter
            y = topOuter

        elif (direction == 0):

            if (topInner == topOuter or x > rightOuter):
                x = rightOuter
                y = topOuter
                direction = 1
                # print ("New direction")
                continue

            if (x == leftOuterTrue and x != 0):
                lefti += 1

            a = min(rightOuter, leftOuterTrue +
                    lefti * (2**currResolution) - 1)
            # print (a)
            xRep = int((x + a) / 2)
            distX = float(x + a) / 2
            b = topInner - 1
            distY = float(y + b) / 2
            yRep = int((y + b) / 2)

            pixelSize = (a + 1 - x) * (b + 1 - y)

            nodeNum += 1

            # copy the values from mra currResolution
            for p in range(int(x), int(a + 1)):
                for q in range(int(y), int(b + 1)):
                    Dict[nodeNum] = [distY, distX, size]
                    nodesImage[q][p][0] = mraImages[yRep][xRep][currResolution]
                    nodesImage[q][p][1] = nodeNum
                    finalImage[q][p] = mraImages[yRep][xRep][currResolution]

            lefti += 1

            x = a + 1

        elif (direction == 1):

            if (rightInner == rightOuter or y > bottomOuter):
                x = rightOuter
                y = bottomOuter
                direction = 2
                # print ("New direction")
                continue

            if (y == topOuterTrue and y != 0):
                topi += 1

            a = rightInner + 1
            distX = float(x + a) / 2
            xRep = int((x + a) / 2)
            b = min(bottomOuter, topOuterTrue + topi * (2**currResolution) - 1)
            distY = float(y + b) / 2
            yRep = int((y + b) / 2)

            pixelSize = (x + 1 - a) * (b + 1 - y)

            nodeNum += 1

            # copy the values from mra currResolution
            for p in range(int(a), int(x + 1)):
                for q in range(int(y), int(b + 1)):
                    Dict[nodeNum] = [distY, distX, size]
                    nodesImage[q][p][0] = mraImages[yRep][xRep][currResolution]
                    nodesImage[q][p][1] = nodeNum
                    finalImage[q][p] = mraImages[yRep][xRep][currResolution]

            topi += 1

            y = b + 1

        elif (direction == 2):

            if (bottomInner == bottomOuter or x < leftOuter):
                x = leftOuter
                y = bottomOuter
                direction = 3
                # print ("New direction")
                continue

            if (x == rightOuterTrue and x != size - 1):
                righti += 1

            a = max(leftOuter, rightOuterTrue -
                    righti * (2**currResolution) + 1)
            distX = float(x + a) / 2
            xRep = int((x + a) / 2)
            b = bottomInner + 1
            distY = float(y + b) / 2
            yRep = int((y + b) / 2)

            pixelSize = (x + 1 - a) * (y + 1 - b)

            nodeNum += 1

            # copy the values from mra currResolution
            for p in range(int(a), int(x + 1)):
                for q in range(int(b), int(y + 1)):
                    Dict[nodeNum] = [distY, distX, size]
                    nodesImage[q][p][0] = mraImages[yRep][xRep][currResolution]
                    nodesImage[q][p][1] = nodeNum
                    finalImage[q][p] = mraImages[yRep][xRep][currResolution]

            righti += 1

            x = a - 1

        elif (direction == 3):
            # print ("Done 1")

            if (leftInner == leftOuter or y < topOuter):
                direction = 4
                # print ("New layer")
                continue

            if (y == bottomOuterTrue and y != size - 1):
                bottomi += 1

            a = leftInner - 1
            distX = float(x + a) / 2
            xRep = int((x + a) / 2)
            b = max(topOuter, bottomOuterTrue -
                    bottomi * (2**currResolution) + 1)
            distY = float(y + b) / 2
            yRep = int((y + b) / 2)

            pixelSize = (a + 1 - x) * (y + 1 - b)

            nodeNum += 1

            # copy the values from mra currResolution
            for p in range(int(x), int(a + 1)):
                for q in range(int(b), int(y + 1)):
                    Dict[nodeNum] = [distY, distX, size]
                    nodesImage[q][p][0] = mraImages[yRep][xRep][currResolution]
                    nodesImage[q][p][1] = nodeNum
                    finalImage[q][p] = mraImages[yRep][xRep][currResolution]

            bottomi += 1

            y = b - 1

    # plt.imshow(finalImage, interpolation='nearest')
    # plt.show()

    return nodesImage, Dict

if __name__ == '__main__':

    size = 128
    power = 7
    w, h = size, size
    t = (w, h)
    A = np.random.randint(0, 255, t, dtype=np.uint8)
    # plt.imshow(A, interpolation='nearest')
    # plt.show()

    start_point = np.random.randint(1, size - 1, (1, 2))
    # print ("Start Point: \n", start_point)

    s_0 = start_point[0][0]
    s_1 = start_point[0][1]

    # a, b = cellDecomp(A, size, power, s_0, s_1)
