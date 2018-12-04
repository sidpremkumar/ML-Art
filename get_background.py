import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm
from random import randint
import math
import glob


BASE = 'outputs/most_interesting/test1.jpeg'
IMAGE = 'outputs/most_interesting/findOpt-OPT.jpg'

SIZE = 1000

def load_img(path_to_img):
    max_dim = SIZE
    img = Image.open(path_to_img)
    # long
    x = max(img.size)
    scale = float(float(max_dim) / float(x))
    width, height = img.size
    # scale and resize the images, so that they are the same
    img = img.resize((int(width * scale), int(height * scale)), Image.ANTIALIAS)

    return img



def getBackground(img, base):
    zeroed = False
    width,height = img.size
    for x in tqdm(range(width)):
        for y in range(height):
            try:
                # if(zeroed != True):
                #     base.putpixel((x,y), (0,0,0))
                if(img.getpixel((x,y)) != (0,0,0) ):
                    base.putpixel((x,y), (0,0,0))
            except Exeption as error:
                print(error)
    zeroed = True
    return base

def main():
    #load in the image we want to get the background of
    img = load_img(IMAGE)
    #load in the base image
    base = load_img(BASE)

    #get the background
    img = getBackground(img, base)
    img.save('outputs/most_interesting/Background-OPT.jpg')

if __name__ == '__main__':
    main()
print("Done!")
