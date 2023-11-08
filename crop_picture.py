from PIL import Image
from PIL import ImageDraw
import os
"""
This code is used to crop two boxes from an image, show the border on origin picture and concat them.
Please first use a tool (e.g. Microsoft painter) to locate the boxes you want to crop out.
box1 = (x1,y1,x2,y2) # input the corrdinate of the diagonal line of the box.
"""
######################################################################

box1 = (30,332,296,474)  # red box, (x1,y1,x2,y2)
box2 = (42,493,225,597)  # green box, it should has same size as box1,if you don't need box2, please make box2=(0,0,0,0)
in_file_dir  = './input/'  # input dir, contain several pictures.
out_file_dir = './output/' # output dir

######################################################################

for files in os.listdir(in_file_dir):
    img_name, img_suffix = files.split('.')
    if img_suffix == 'DS_Store': #  For macos
        continue
    img = Image.open(in_file_dir + img_name + '.' + img_suffix)
    width, height = img.size
    crop_width = int(width) if sum(box2)==0 else int(width/2)
    crop_height = int(crop_width/(box1[2]-box1[0])*(box1[3]-box1[1]))
    
    #crop out the box
    crop1 = img.crop(box1)
    crop2 = img.crop(box2)

    #draw the Border to the origin picture
    img_add_box = ImageDraw.ImageDraw(img)
    img_add_box.rectangle(((box1[0], box1[1]), (box1[2], box1[3])), outline='red', width=5)
    img_add_box.rectangle(((box2[0], box2[1]), (box2[2], box2[3])), outline='green', width=5)

    crop1 = crop1.resize((crop_width, crop_height))
    crop2 = crop2.resize((crop_width, crop_height))

    crop1_add_box = ImageDraw.ImageDraw(crop1)
    crop1_add_box.rectangle(((0, 0), (crop1.width, crop1.height)), outline='red', width=5)
    crop2_add_box = ImageDraw.ImageDraw(crop2)
    crop2_add_box.rectangle(((0, 0), (crop2.width, crop2.height)), outline='green', width=5)

    #concat the result
    concat = Image.new('RGB', (width, height+crop_height))
    concat.paste(img, (0, 0))
    concat.paste(crop1, (0, height))
    concat.paste(crop2, (crop1.width, height))
    concat.save(out_file_dir + img_name + '.' + img_suffix)