import os
import numpy as np
from PIL import Image
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
import imageio.v2 as io


# make dir if not existing
def maybe_mkdir_p(directory):
    os.makedirs(directory, exist_ok=True)

# format number
def fnumber(x):
    if(x<10):
        return "00" + str(x)
    elif(x<100):
        return "0" + str(x)
    return str(x)

def convert_image_png(path, png_path):
    im = Image.open(path)
    im.save(png_path + '.png')

def convert_label_png(jpg, png_path):
    im = Image.open(jpg).convert("L")
    im = np.array(im)
    binary_mask = np.where(im > 127, 1, 0).astype(np.uint8)
    io.imwrite(png_path + '.png', binary_mask)

def convert_sample(i, init_images_path, image, target_images, init_masks_path, mask, target_labels):
    print("converting " + str(i) + "...")
    convert_image_png(os.path.join(init_images_path, image), target_images + 'Kvasir-SEG' + fnumber(i) + '_0000')
    convert_label_png(os.path.join(init_masks_path, mask), target_labels + 'Kvasir-SEG' + fnumber(i))



if __name__ == "__main__":
    # dataset guidelines https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md

    # set  how many test samples, TotalSamples/condition
    # i.e. 1000/10 = 100 images for test set, then 900 images for training set
    condition = 10

    #to path
    nnUNet_raw = 'nnUNet_raw/'
    dataset_name = 'Dataset001_kvasir-seg/'
    target_folder = os.path.join(nnUNet_raw, dataset_name)
    target_imagesTr = os.path.join(target_folder, 'imagesTr/')
    target_imagesTs = os.path.join(target_folder, 'imagesTs/')
    target_labelsTr = os.path.join(target_folder, 'labelsTr/')
    target_labelsTs = os.path.join(target_folder, 'labelsTs/')
    maybe_mkdir_p(nnUNet_raw)
    maybe_mkdir_p(target_folder)
    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)
    maybe_mkdir_p(target_labelsTs)


    #training and test set
    folder = 'Kvasir-SEG'
    init_images_path = os.path.join(folder, 'images')
    images = os.listdir(init_images_path)
    init_masks_path = os.path.join(folder, 'masks')
    #masks = os.listdir(init_masks_path)
    N = len(images)
    Test = 0
    for i in range(N):
        if(i%condition==0):
            convert_sample(i,init_images_path,images[i],target_imagesTs,init_masks_path,images[i],target_labelsTs)
            Test += 1
        else:
            convert_sample(i,init_images_path,images[i],target_imagesTr,init_masks_path,images[i],target_labelsTr)

    #json        
    print("generating json...")
    generate_dataset_json(os.path.join(nnUNet_raw, dataset_name), {0: 'R', 1: 'G', 2: 'B'}, {'background': 0, 'polyp': 1},
                            N - Test, '.png', dataset_name=dataset_name, description='adapted dataset from Kvasir-SEG') 

    print("Done.")
