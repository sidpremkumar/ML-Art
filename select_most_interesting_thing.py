import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm
import sys

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
mod = SourceModule("""
__global__ void multiply_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  dest[i] = a[i] * b[i];
}
""")

#global variables
num_iterations = 50
#Open the images
img = cv.imread('messi5.jpg')
base = Image.open('messi5.jpg')

#Implementation of grabcut algorithum. Taken from: https://docs.opencv.org/3.4/d8/d83/tutorial_py_grabcut.html
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (50,50,450,290)
cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')


def incr(pix,factor):
    # print("before: ", pix)
    one = max(255, pix[0]+factor)
    two = max(255, pix[1]+factor)
    three = max(255, pix[2]+factor)
    new = (one, two, three)
    # print("after: ", new)
    return pix


new = img.copy()

#Repetedily grab and blend the images, hopefully getting a good idea of what we want to see !
#for i in tqdm(range(num_iterations)): #repeat!
for i in tqdm(range(num_iterations)): #repeat!
    if num_iterations == 1: #base case
            img = img*mask2[:,:,np.newaxis] #grabcut image
            img = Image.fromarray(img) #convert it to ndarray
            test = img
            test.save('outputs/most_interesting/' + 'crop_' + str(i) + '.jpg')
            width, height = img.size
            pix = img.copy()
            factor = 25
            for x in tqdm(range(10)):
                for w in xrange(width-1):
                    for h in xrange(height):
                        #TODO: Change it so its looking for != 0
                        if(pix.getpixel( (w,h) ) != (0,0,0)):
                            try:
                                if (img.getpixel( (w+1,h) ) == (0,0,0)):
                                    img.putpixel( (w+1, h), incr(img.getpixel( (w+1,h) ),factor))
                                if (img.getpixel( (w,h+1) ) == (0,0,0)):
                                    img.putpixel( (w, h+1), incr(img.getpixel( (w+1,h) ),factor))
                                if (img.getpixel( (w+1,h+1) ) == (0,0,0)):
                                    img.putpixel( (w+1, h+1), incr(img.getpixel( (w+1,h) ),factor))
                                if(img.getpixel( (w-1,h) ) == (0,0,0)):
                                    img.putpixel( (w-1, h), incr(img.getpixel( (w+1,h) ),factor))
                                if (img.getpixel( (w,h-1) ) == (0,0,0)):
                                    img.putpixel( (w, h-1), incr(img.getpixel( (w+1,h) ),factor))
                                if (img.getpixel( (w-1,h-1) ) == (0,0,0)):
                                    img.putpixel( (w-1, h-1), incr(img.getpixel( (w+1,h) ),factor))
                            except:
                                  continue
                # img.save('outputs/most_interesting/' + str(i) + '-messi2.jpg')

            # pix = np.clip(pix, 0, 255).astype('uint8')
            # print(type(pix))
            # new = Image.fromarray(pix)
            new = Image.blend(base,img, 0.5) #blend the image and the original image
            new.save('outputs/most_interesting/' + 'img_' + str(i) + '.jpg')


    img = new*mask2[:,:,np.newaxis] #grabcut image
    img = Image.fromarray(img) #convert it to ndarray
    test = img
    test.save('outputs/most_interesting/' + 'crop_' + str(i) + '.jpg')
    width, height = img.size
    pix = img.copy()
    factor = 50
    for x in tqdm(range(25)):
        for w in xrange(width-1):
            for h in xrange(height):
                #TODO: Change it so its looking for != 0
                if(pix.getpixel( (w,h) ) != (0,0,0)):
                    try:
                        if (img.getpixel( (w+1,h) ) == (0,0,0)):
                            img.putpixel( (w+1, h), incr(img.getpixel( (w+1,h) ),factor))
                        if (img.getpixel( (w,h+1) ) == (0,0,0)):
                            img.putpixel( (w, h+1), incr(img.getpixel( (w+1,h) ),factor))
                        if (img.getpixel( (w+1,h+1) ) == (0,0,0)):
                            img.putpixel( (w+1, h+1), incr(img.getpixel( (w+1,h) ),factor))
                        if(img.getpixel( (w-1,h) ) == (0,0,0)):
                            img.putpixel( (w-1, h), incr(img.getpixel( (w+1,h) ),factor))
                        if (img.getpixel( (w,h-1) ) == (0,0,0)):
                            img.putpixel( (w, h-1), incr(img.getpixel( (w+1,h) ),factor))
                        if (img.getpixel( (w-1,h-1) ) == (0,0,0)):
                            img.putpixel( (w-1, h-1), incr(img.getpixel( (w+1,h) ),factor))
                    except:
                          continue
        # img.save('outputs/most_interesting/' + str(i) + '-messi2.jpg')

    # pix = np.clip(pix, 0, 255).astype('uint8')
    # print(type(pix))
    # new = Image.fromarray(pix)
    new = Image.blend(base,img, 0.5) #blend the image and the original image

    new.save('outputs/most_interesting/' + 'img_' + str(i) + '.jpg')


    # img.save('outputs/most_interesting/' + str(i) + '-crop.jpg')
    #
    #
    # new = Image.blend(base,img, 0.5) #blend the image and the original image
    #
    # img = new
    #
    # new.save('outputs/most_interesting/' + str(i) + '-messi.jpg')


print("Done!")

if __name__ == '__main__':
    multiply_them = mod.get_function("multiply_them")

    a = numpy.random.randn(400).astype(numpy.float32)
    b = numpy.random.randn(400).astype(numpy.float32)

    dest = numpy.zeros_like(a)
    multiply_them(
            drv.Out(dest), drv.In(a), drv.In(b),
            block=(400,1,1), grid=(1,1))

    print dest-a*b
