import os
import numpy as np
from PIL import Image, ImageEnhance
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
import imageio.v2 as io

def fnumber(x):
    if(x<10):
        return "000" + str(x)
    elif(x<100):
        return "00" + str(x)
    elif(x<1000):
        return "0" + str(x) 
    return str(x)

def augment_sample(new_id, setting, image, target_imagesTr, target_labelsTr):
    new_name = 'HarDNet-MSEG-' + 'aug' + new_id
    #image
    im = Image.open(target_imagesTr + image)
        # BRIGHTNESS
    enhancer = ImageEnhance.Brightness(im)
    im = enhancer.enhance(setting[0])
        # CONTRAST
    enhancer = ImageEnhance.Contrast(im)
    im = enhancer.enhance(setting[1])
        # COLOR
    enhancer = ImageEnhance.Color(im)
    im = enhancer.enhance(setting[2])
        # SHARPNESS
    enhancer = ImageEnhance.Sharpness(im)
    im = enhancer.enhance(setting[3])
    im.save(target_imagesTr + new_name + '_0000.png')
    #label
    mask = image[:-9] + '.png'
    double_label = "cp " + target_labelsTr + mask + " " + target_labelsTr + new_name + '.png'
    os.system(double_label)


if __name__ == "__main__":
    #target path
    nnUNet_raw = 'nnUNet_raw/'
    dataset_name = 'Dataset002_HarDNet-MSEG/'
    target_folder = os.path.join(nnUNet_raw, dataset_name)
    target_imagesTr = os.path.join(target_folder, 'imagesTr/')
    target_labelsTr = os.path.join(target_folder, 'labelsTr/')

    images = os.listdir(target_imagesTr)

    i = len(images)
    N = int(i*2)

    setting_matrix = [[0.4, 1.3, 1.7, 0.7], [1.1, 0.8, 1.4, 0.4], [1.5, 1.2, 0.6, 0.5], [1.6, 0.5, 1.8, 0.9], [1.4, 1.2, 0.8, 1.6]]

    setting_rotation = 0
    for e in images:
        print("augmenting " + e + "...")
        augment_sample(fnumber(i), setting_matrix[setting_rotation], e, target_imagesTr, target_labelsTr)
        i += 1
        setting_rotation += 1
        if(setting_rotation==5):
            setting_rotation = 0
    

    #json        
    print("generating new json...")
    generate_dataset_json(os.path.join(nnUNet_raw, dataset_name), {0: 'R', 1: 'G', 2: 'B'}, {'background': 0, 'polyp': 1},
                            N, '.png', dataset_name=dataset_name, description='adapted dataset from HarDNet-MSEG') 

    print("Done.")