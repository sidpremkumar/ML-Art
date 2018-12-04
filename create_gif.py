import glob, os
import moviepy.editor as mpy
#os.chdir("outputs")
style_name = 'mozGanMNIST'





gif_name = style_name
fps = 12
file_list = sorted(glob.glob('plots/*.bmp')) # Get all the pngs in the current directory


with open('image_list.txt', 'w') as file:
    for item in file_list:
        file.write("%s\n" % item)
os.system('convert @image_list.txt {}.gif'.format(gif_name)) # On windows convert is 'magick'

# # x = list.sort(file_list) # Sort the images by #, this may need to be tweaked for your use case
# # print(x)
# # clip = mpy.ImageSequenceClip(file_list, fps=fps)
# # clip.write_gif('{}.gif'.format(gif_name), fps=fps)
