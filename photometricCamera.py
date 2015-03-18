import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as ln
import cv2
import time
import subprocess
import opc
import os

def normalise(a):
    a /= ln.norm(a)
    return a

# sRGB conversion functions from http://excamera.com/sphinx/article-srgb.html

def s2lin(x):
    a = 0.055
    return np.where(x <= 0.04045,
                 x * (1.0 / 12.92),
                 pow((x + a) * (1.0 / (1 + a)), 2.4))

def lin2s(x):
    a = 0.055
    return np.where(x <= 0.0031308,
                 x * 12.92,
                 (1 + a) * pow(x, 1 / 2.4) - a)

class PhotometricCamera():
    """ Class to control the photometric camera """

    def __init__(self):

        ########### Initialise variables ###########

        ############# State variables ##############
        self.haveImages = False
        self.haveCalibration = False

        ################## Others ##################
        np.set_printoptions(precision=3)

        print('Camera initialised')

        pass

    def calibrate(self):

        if self.haveImages:

            # Delete existing calibration data
            self.ls = []
            self.As = []

            for im in self.ims:
                imWidth = im.shape[1]
                imHeight = im.shape[0]

                # Convert the image to grayscale
                imGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

                # Find the co-ords of maximum brightness
                blurSize = 10
                imBlur = cv2.blur(imGray, (blurSize, blurSize))
                maxInd = np.argmax(imBlur)
                (yMax, xMax) = np.unravel_index(maxInd, imBlur.shape)

                # Change variables to r2
                [x, y] = np.meshgrid(range(imWidth), range(imHeight))
                x -= xMax
                y -= yMax

                r2 = x ** 2 + y ** 2
                Ipow = np.power(imGray, -2/3.0)
                p = np.polyfit(Ipow.flatten(), r2.flatten(), 1)
                print('Fit complete:' + np.array_str(p))

                lz = np.sqrt(np.abs(p[1]))
                A = p[0] ** (2/3.0)

                self.As.append( A )
                self.ls.append( np.array([xMax, yMax, lz]) )

            print('Calibration complete')
            self.haveCalibration = True
            self.printCalibration()

        else:
            print('No calibration images loaded')

        pass

    def capture(self):

        print('Capturing photos')
        # initialise the camera
        cam = cv2.VideoCapture(0)
        # make sure the camera is getting frames
        ret, _ = cam.read()
        timeDelay = 1

        # Set up the opc client and variables
        client = opc.Client('localhost:7890')
        numLEDs = 3 # Number of illumination LEDs
        nFlush = 5 # Number of frames to capture to flush the camera
        black = (0,0,0)
        white = (255,255,255)

        # Clear existing images
        ims = []

        client.put_pixels([white, black, black])
        ims.append( self.grabFrame(cam, nFlush, timeDelay) )

        client.put_pixels([black, white, black])
        ims.append( self.grabFrame(cam, nFlush, timeDelay) )

        client.put_pixels([black, black, white])
        ims.append( self.grabFrame(cam, nFlush, timeDelay) )

        # Reset the LEDs
        client.put_pixels([black, black, black])

    def grabFrame(self, nFlush, timeDelay):
        # flush the camera
        for f in range(nFlush):
            ret, _ = cam.read()

        time.sleep(timeDelay)
        ret, camIm = cam.read()
        time.sleep(timeDelay)

        return camIm

    def reconstruct(self):
        pass


    def loadImages(self, folderName):
        """ Load images from a folder, expected file names are im1, im2, im3 (jpg) """

        try:
            im1 = cv2.imread(folderName + '/im1.jpg')
            im2 = cv2.imread(folderName + '/im2.jpg')
            im3 = cv2.imread(folderName + '/im3.jpg')

            self.ims = [im1, im2, im3]

            print('Images loaded')
            self.haveImages = True

        except:
            print('Failed to load images')

        pass

    def saveImages(self):
        pass

    ############ UTILITY FUNCTIONS ################
    def printCalibration(self):
        if self.haveCalibration:
            print('###### Current Calibration ######')
            print('Lighting vectors:')
            print(self.ls)
            print('Intensity constants:')
            print(self.As)
            print('#################################')

        else:
            print('No calibration loaded')
        
if __name__ == '__main__':

    cam = PhotometricCamera()
    cam.loadImages('./flat')
    cam.calibrate()
    cam.capture()
    cam.reconstruct()