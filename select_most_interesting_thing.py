import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm
from random import randint
import math
import glob

# WHAT IS THIS?
# This is a python file that is attempting to select the most interesting thing
# from an image. It's a work in progress and I haven't had time to properly document
# what I'm trying to do :(

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

#global variables ->
SIZE = 1000
image = 'outputs/most_interesting/test1.jpeg'


def rect(x1, x2, y1, y2):
    return (x1,x2,y1,y2)


def load_img(path_to_img):
    max_dim = SIZE
    img = Image.open(path_to_img)
    # long
    x = max(img.size)
    scale = float(float(max_dim) / float(x))
    width, height = img.size
    # scale and resize the images, so that they are the same
    img = img.resize((int(width * scale), int(height * scale)), Image.ANTIALIAS)

    #convert to openCV format
    img = np.array(img)
    return img


def grabCut(img, rect):
    # img = cv2.imread('outputs/most_interesting/messi5.jpg')
    img_local = img
    mask = np.zeros(img_local.shape[:2],np.uint8)
    cv2.grabCut(img_local,mask,rect,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img_local = img_local*mask2[:,:,np.newaxis]
    new = Image.fromarray(img_local)
    return new


def findOpt(imgs,percent):

    new_imgs = []
    print("Finding Optimal... ")
    for i in tqdm(range(len(imgs))):
        # TODO: No more image.open when working
        img_iter = Image.open(imgs[i])
        totalpix = img_iter.size[0] * img_iter.size[1]
        num_black_pix = 0
        pixles = img_iter.getdata()
        for pixle in pixles:
            if pixle == (0,0,0):
                num_black_pix += 1
        percenta = float(float(num_black_pix)/float(totalpix))
        #print(1-percenta)
        if (1-percenta) >= percent:
            new_imgs.append(img_iter)

    return new_imgs
def averagePix(img_iter):
    img = img_iter.copy()
    width, height = img.size

    total1 = 0
    total2 = 0
    total3 = 0
    for i in range(0, width):
        for j in range(0, height):
            total1 += img.getpixel((i,j))[0]
            total2 += img.getpixel((i,j))[1]
            total3 += img.getpixel((i,j))[2]

    mean1 = total1 / (width * height)
    mean2 = total2 / (width * height)
    mean3 = total3 / (width * height)
    return mean1, mean2, mean3


def concatImgs(imgs,base):
    print("Concat best images...")
    zeroed = False
    base_copy = base.copy()
    average = averagePix(base_copy)
    for i in tqdm (range(len(imgs))):
        #img_iter = Image.open(imgs[i])
        img_iter = imgs[i]
        width, height = img_iter.size
        for x in range(width):
            for y in range(height):
                try:
                    if (zeroed != True):
                        base.putpixel((x,y), average)
                    if (img_iter.getpixel((x,y)) != (0,0,0)):
                        base.putpixel((x,y), base_copy.getpixel((x,y)))
                except Exception as error:
                     print(error)
        zeroed = True
    return base



def main():
    #size -> 1000x1000
    img = load_img(image)

    imgs = []
    x1 = 0
    x2 = 0
    y2 = 0
    y1 = 0
    print("Collecting all possibilities...")
    #TODO: testing we will just load from array
    # help1 = 1000^2
    # help2 = 750^2
    # tot = math.sqrt(help1 * help1 * help2 * help2)
    # counter = 0
    # while x1 < 50:
    #     x1 += randint(0,200)
    #
    #     while x2 < 1000:
    #         x2 += randint(0,300)
    #
    #         while y1 < 750:
    #             y1 += randint(0,400)
    #
    #             while y2 < 750:
    #                 y2 += randint(250,500)
    #                 #TODO: Make this prediction better. Move it to the first while loop, and set x2, y1, y2 to 1000, 750, 750, then make it a %
    #                 help1 = x1^2
    #                 help11 = x2^2
    #                 help2 = y2^2
    #                 help22 = y2^2
    #                 left = math.sqrt(help1 * help11 * help2 * help22)
    #                 #remaining is just calcualted similar to euclidian distance
    #                 print("remaining = :", tot - left - 300000)
    #                 try:
    #                     rectangle = rect(int(x1),int(x2),int(y1),int(y2))
    #                     #print(rectangle)
    #                     new = grabCut(img,rectangle)
    #                     imgs.append(new)
    #                     new.save('outputs/most_interesting/testing-' + str(counter)+'.jpg')
    #                     counter += 1
    #                 except:
    #                     continue
    #             y2 = 0
    #         y1 = 0
    #     x2 = 0
    base = load_img(image)
    base = Image.fromarray(base)
    imgs = glob.glob('outputs/most_interesting/testing-*.jpg')
    #findOpt (image array, percent of blacsk pix at most)
    img = findOpt(imgs, 0.15)
    # counter = 0
    # for i in img:
    #     i.save('outputs/most_interesting/findOpt-' + str(i) + '.jpg')
    #     counter += 1
    img = concatImgs(img, base)

    img.save('outputs/most_interesting/findOpt-OPT.jpg')
    # for i in img:
    #     i.save('outputs/most_interesting/findOpt-' + str(i) + '.jpg')



if __name__ == '__main__':
    main()
print("Done!")
