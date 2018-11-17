import glob, os
import moviepy.editor as mpy
os.chdir("outputs")
style_name = 'rick'

gif_name = style_name
fps = 12
file_list = sorted(glob.glob('*.bmp')) # Get all the pngs in the current directory
# print(file_list)
# x = list.sort(file_list) # Sort the images by #, this may need to be tweaked for your use case
# print(x)
clip = mpy.ImageSequenceClip(file_list, fps=fps)
clip.write_gif('{}.gif'.format(gif_name), fps=fps)
