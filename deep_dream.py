import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')
import tensorflow as tf
import numpy as np
import random
import math
from tqdm import tqdm
from resizeimage import resizeimage

# Image manipulation.
from PIL import Image
import PIL.Image
from scipy.ndimage.filters import gaussian_filter


import inception5h

style_name = 'static'

#Helper functions for messing with the image
def load_image(filename):
    image = PIL.Image.open(filename)
    #returns a np.float32 version of the image
    return np.float32(image)

def save_image(image,filename):
    #need to make sure that pixle values are between 0 and 225
    image = np.clip(image,0.0,255.0)
    #Converting to bytes
    image = image.astype(np.uint8)
    #write the image
    with open(filename,'wb') as file:
        PIL.Image.fromarray(image).save(file,'jpeg')

def plot_image(image):
    #ensure pixles are between 0 and 225
    image = np.clip(image,0.0,225.0)
    #convert to bytes
    image = image.astype(np.uint8)
    #display the image
    display(PIL.Image.fromarray(image))

def normalize_image(x):
    #get the biggest and smallest values from the pixles
    _min = x.min()
    _max = x.max()

    #we want all numbers to be between 0-1
    _norm = (x-_min)/(_max - _min)
    return _norm

def plot_gradient(gradient):
    #normalize the gradient
    gradient_norm = normalize_image(gradient)

    #show what we normalized
    # plt.imshow(gradient_norm, interpolation='bilinear')
    # plt.show()

def resize_image(image, size=None, factor=None):
    if factor is not None:
        #scale the numpy array's shate for height and width
        size = np.array(image.shape[0:2]) * factor
        #PIL reqires integers
        size = size.astype(int)
    else:
        size = size[0:2]

    #height and width is oppositve in numpy then in PIL

    size = tuple(reversed(size))
    x = size[0]
    y = size[1]
    z = x,y
    #ensure pixles are between 0 and 225
    img = np.clip(image, 0.0,225.0)
    #convert to bytes
    img = img.astype(np.uint8)
    #create pil object
    image= Image.fromarray(img)


    #resize the image
    # image_resize = img.resize(z, PIL.Image.LANCZOS)
    image_resize = resizeimage.resize_cover(image, z)

    image_resize = np.float32(image_resize)


    return image_resize

def get_tile_size(num_pixles,tile_size=400):
    #how many tiles can we create
    num_tiles = int(round(num_pixles/tile_size))
    #ensure at least 1 tile
    num_tiles = max(1,num_tiles)

    actual_tile_size = math.ceil(num_pixles / num_tiles)

    return actual_tile_size

def tiled_gradient(gradient, image, tile_size=400):
    #allocate an array for the gradient of the image
    grad = np.zeros_like(image)

    #number of pixles for the x- and y- axis
    x_max, y_max, _ = image.shape
    #tile size for x-axis
    x_tile_size = get_tile_size(num_pixles=x_max, tile_size=tile_size)
    #1/4 of the tile size
    x_tile_size4 = x_tile_size // 4

    #tile size for y-axis
    y_tile_size = get_tile_size(num_pixles=y_max, tile_size=tile_size)
    y_tile_size4 = y_tile_size // 4

    #random the start position for the tiles on the x-axis
    x_start = random.randint(-3*x_tile_size4, -x_tile_size4)


    while x_start < x_max:
        #end position for current tile
        x_end = x_start+x_tile_size

        #ensure that the tiles start and end positions are valid
        x_start_lim = max(x_start, 0)
        x_end_lim = min(x_end, x_max)

        #random the start position for the tiles on the y-axis
        y_start = random.randint(-3*y_tile_size4, -y_tile_size4)

        while(y_start <y_max):
            y_end = y_start + y_tile_size
            #ensure that the values are valid
            y_start_lim = max(y_start,0)
            y_end_lim = min(y_end, y_max)

            #get the actual image tile
            img_tile = image[x_start_lim:x_end_lim, y_start_lim:y_end_lim, :]

            #create a feed-dict so we can actually pass it through tensorflow
            feed_dict = model.create_feed_dict(image=img_tile)

            #use tensorflow to calculate the gradient-value
            g = session.run(gradient, feed_dict=feed_dict)

            #normalize the gradient for the tile
            g /= (np.std(g) + 1e-8)
            grad[x_start_lim:x_end_lim, y_start_lim:y_end_lim, :] = g
            y_start = y_end
        x_start = x_end
    return grad

def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                               "dimension [1, height, width, channel] or [height, width, channel]")
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")

    # perform the inverse of the preprocessiing step
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x



def optimize_image(layer_tensor, image, num_iterations=10, step_size=3.0, tile_size=400, show_image=False, rLevel=0):
    #take a layer_tensor (that we want to be maximised) and a image, and a step_size for the gradient,
    # and if we want to show the gradient
    img = image.copy()

    print("Processing Image:", end="")
    #get the gradient from the spesific layer to the input image (called from the inception5h.py)
    #uses tf.gradients to calcualte the gradient after we square the tensor
    gradient = model.get_gradient(layer_tensor)

    for i in tqdm(range(num_iterations)):
        grad = tiled_gradient(gradient=gradient, image=img)

        sigma = (i * 4.0) / num_iterations + 0.5
        grad_smooth1 = gaussian_filter(grad, sigma=sigma)
        grad_smooth2 = gaussian_filter(grad, sigma=sigma*2)
        grad_smooth3 = gaussian_filter(grad, sigma=sigma*0.5)
        grad = (grad_smooth1 + grad_smooth2 + grad_smooth3)

        step_size_scaled = step_size / (np.std(grad) + 1e-8)

        img += grad * step_size_scaled
        if show_image:
            # Print statistics for the gradient.
            msg = "Gradient min: {0:>9.6f}, max: {1:>9.6f}, stepsize: {2:>9.2f}"
            #print(msg.format(grad.min(), grad.max(), step_size_scaled))

            # Plot the gradient.
            plot_gradient(grad)
            plot_img = deprocess_img(img)
            final_image = Image.fromarray(plot_img)
            # Save the image
            # final_image.save('outputs/deep_dream_test1/' + str(style_name) + '/' +  str(style_name) + '-' + str(i) + '.bmp')
            save_image(img, filename='outputs/deep_dream_test1/' + str(style_name) + '/' +  str(style_name) + '-' + str(rLevel) + '-' + str(i) + '.jpg')
        else:
            # Otherwise show a little progress-indicator.
            print(". ", end="")

    print()
    print("Done!")

    return img

def recursize_optimizer(layer_tensor, image, num_repeates=4, rescale_factor=0.7, blend=0.2,
                        num_iterations=100, step_size=3.0, tile_size=400):
    if num_repeates>0:
        #blur the input image to prevent artifacts when downscaling

        sigma = 0.5

        img_blur = gaussian_filter(image, sigma=(sigma, sigma, 0.0))

        #downscale the image
        img_downscale = resize_image(image=img_blur, factor=rescale_factor)

        #recursivly call the function, subtracting 1 from numrepeates
        img_result = recursize_optimizer(layer_tensor=layer_tensor,
                                        image=img_downscale,
                                        num_repeates= num_repeates-1,
                                        rescale_factor=rescale_factor,
                                        blend=blend,
                                        num_iterations=num_iterations,
                                        step_size=step_size,
                                        tile_size=tile_size)
        #upscale the image
        img_upscale = resize_image(image=img_blur, size=image.shape)

        #blend the original and processed image
        image = blend * image + (1.0 - blend) * img_upscale
    print("Recursive Level: ", num_repeates)


    #process the image using the deep dream algorithum

    img_result = optimize_image(layer_tensor=layer_tensor,
                                image=image,
                                num_iterations=num_iterations,
                                step_size=step_size,
                                tile_size=tile_size, show_image=True, rLevel=num_repeates)
    return img_result





def main():
    # model = inception5h.Inception5h()
    # session = tf.InteractiveSession(graph=model.graph)
    image = load_image(filename='img/s4.jpeg')
    layer_tensor = model.layer_tensors[2]
    # img_result = optimize_image(layer_tensor, image,
    #                num_iterations=10, step_size=6.0, tile_size=400,
    #                show_image=True)

    img_result = recursize_optimizer(layer_tensor=layer_tensor, image=image,
                 num_iterations=1000, step_size=3.0, rescale_factor=0.7,
                 num_repeates=4, blend=0.2)


if __name__ == '__main__':
    #GPU setup ->
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    session = tf.Session(config=config)
    model = inception5h.Inception5h()
    session = tf.InteractiveSession(graph=model.graph)
    inception5h.maybe_download()
    main()
