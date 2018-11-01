from main import *
import pickle
import multiprocessing


#trying to make it faster
pool = multiprocessing.Pool(processes=4)

# enabling eager execution
enableEagerExecution()

# text_file = open('variables.txt','r')
# clipped = pickle.load(text_file)
#
#
# init_image = load_and_process_img(content_path)
# init_image = tfe.Variable(init_image, dtype=tf.float32)
# print(type(clipped))
# init_image.assign(clipped)
#
# plot_img = init_image.numpy()
# plot_img = deprocess_img(plot_img)
# Image.fromarray(plot_img).show()

main_init()