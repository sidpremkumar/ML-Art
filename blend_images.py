
from PIL import Image
from tensorflow.python.keras.preprocessing import image as kp_image
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from tqdm import tqdm

def load_img(path_to_img):
    max_dim = 1000
    img = Image.open(path_to_img)
    # long
    x = max(img.size)
    scale = float(float(max_dim) / float(x))
    width, height = img.size
    # scale and resize the images, so that they are the same
    img = img.resize((int(width * scale), int(height * scale)), Image.ANTIALIAS)
    # We need to broadcast the image array such that it has a batch dimension
    return img

#content
two = 'outputs/temp/temp-1999.bmp'

#background
one = 'outputs/most_interesting/test1.jpeg'
cropped = 'outputs/most_interesting/findOpt-OPT.jpg'



img = load_img(two) #the style
style = img.copy()
#img = cv.imread(two)
cropped_img = load_img(cropped)
#cropped_img = cv.imread(cropped)
base = load_img(one) #the base
#base = cv.imread(one)

cropped_img = np.array(cropped_img)
img = np.array(img)
base = np.array(base)
width, height = cropped_img.shape[0], cropped_img.shape[1]
# width , height = cropped_img.size()


# cropped_img = Image.fromarray(cropped_img)
# base = Image.fromarray(np.array(base))
# img = Image.fromarray(np.array(img))
# pix = cropped_img.copy()
#
# cropped_img1 = Image.new('I', img.size, 0xffffff)


for h in tqdm(xrange(0, height)):
    for w in xrange(0, width):
            # if(cropped_img.getpixel( (w,h) ) == (0,0,0)):
            #     pix.putpixel( (w,h), img.getpixel( (w,h)) )
            # else:
            #     pix.putpixel( (w,h) , img.getpixel( (w,h) ))
            if(cropped_img[ w , h ].all(0)):
                img[w,h] = img [w,h]
            else:
                img[w, h] = base [w,h]

img = Image.fromarray(img)

#blend the original and style image just a little!
new = Image.blend(img, style, 0.1)
new.save('outputs/most_interesting/' + 'img_obama2.jpg')



# new = Image.blend(base,img, 0.25) #blend the image and the original image
# for i in range(10):
#     new = Image.blend(new,img, 0.1)
# new.save('outputs/most_interesting/' + 'img_sama.jpg')
