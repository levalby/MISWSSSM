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
        return "000" + str(x)
    elif(x<100):
        return "00" + str(x)
    elif(x<1000):
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

def convert_sample(i, image, dataset, init_images_path, target_images, init_masks_path, target_labels):
    print("converting " + str(i) + "...")
    convert_image_png(os.path.join(init_images_path, image), target_images + 'HarDNet-MSEG-' + dataset + fnumber(i) + '_0000')
    convert_label_png(os.path.join(init_masks_path, image), target_labels + 'HarDNet-MSEG-' + dataset + fnumber(i))



if __name__ == "__main__":
    # dataset guidelines https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md

    #to path
    nnUNet_raw = 'nnUNet_raw/'
    dataset_name = 'Dataset002_HarDNet-MSEG/'
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


    #training set
    folder_train = 'TrainDataset'
    #all-in-one
    init_images_path = os.path.join(folder_train, 'image')
    images = os.listdir(init_images_path)
    init_masks_path = os.path.join(folder_train, 'mask')
    set0 = len(images)
    train_str = "Training set: 0 - " + str(set0-1)
    N = set0
    i = 0
    while(i<set0):
        convert_sample(i,images[i],"train",init_images_path,target_imagesTr,init_masks_path,target_labelsTr)
        i += 1
    

    #test sets
    folder_test = 'TestDataset'
    dataset_folders = os.listdir(folder_test)

    #CVC-300
    init_images_path = os.path.join(folder_test, dataset_folders[0],'images')
    images = os.listdir(init_images_path)
    init_masks_path = os.path.join(folder_test, dataset_folders[0],'masks')
    set1 = len(images)
    test1 = "CVC-300 set: " + str(i) + " - " + str(i+set1-1)
    j = 0
    N += set1
    while(j<set1):
        convert_sample(i,images[j],"CVC-300",init_images_path,target_imagesTs,init_masks_path,target_labelsTs)
        i += 1
        j += 1
    
    #CVC-ClinicDB
    init_images_path = os.path.join(folder_test, dataset_folders[1],'images')
    images = os.listdir(init_images_path)
    init_masks_path = os.path.join(folder_test, dataset_folders[1],'masks')
    set2 = len(images)
    test2 = "CVC-ClinicDB set: " + str(i) + " - " + str(i+set2-1)
    j = 0
    N += set2
    while(j<set2):
        convert_sample(i,images[j],"CVC-ClinicDB",init_images_path,target_imagesTs,init_masks_path,target_labelsTs)
        i += 1
        j += 1

    #CVC-ColonDB
    init_images_path = os.path.join(folder_test, dataset_folders[2],'images')
    images = os.listdir(init_images_path)
    init_masks_path = os.path.join(folder_test, dataset_folders[2],'masks')
    set3 = len(images)
    test3 = "CVC-ColonDB set: " + str(i) + " - " + str(i+set3-1)
    j = 0
    N += set3
    while(j<set3):
        convert_sample(i,images[j],"CVC-ColonDB",init_images_path,target_imagesTs,init_masks_path,target_labelsTs)
        i += 1
        j += 1

    #ETIS-LaribPolypDB
    init_images_path = os.path.join(folder_test, dataset_folders[3],'images')
    images = os.listdir(init_images_path)
    init_masks_path = os.path.join(folder_test, dataset_folders[3],'masks')
    set4 = len(images)
    test4 = "ETIS-LaribPolypDB set: " + str(i) + " - " + str(i+set4-1)
    j = 0
    N += set4
    while(j<set4):
        convert_sample(i,images[j],"ETIS-LaribPolypDB",init_images_path,target_imagesTs,init_masks_path,target_labelsTs)
        i += 1
        j += 1

    #Kvasir
    init_images_path = os.path.join(folder_test, dataset_folders[4],'images')
    images = os.listdir(init_images_path)
    init_masks_path = os.path.join(folder_test, dataset_folders[4],'masks')
    set5 = len(images)
    test5 = "Kvasir-SEG set: " + str(i) + " - " + str(i+set5-1)
    j = 0
    N += set5
    while(j<set5):
        convert_sample(i,images[j],"Kvasir-Seg",init_images_path,target_imagesTs,init_masks_path,target_labelsTs)
        i += 1
        j += 1

    print(train_str)
    print(test1)
    print(test2)
    print(test3)
    print(test4)
    print(test5)

    print("total train samples: " + str(set0))
    print("total test samples: " + str(set1+set2+set3+set4+set5))

    #json        
    print("generating json...")
    generate_dataset_json(os.path.join(nnUNet_raw, dataset_name), {0: 'R', 1: 'G', 2: 'B'}, {'background': 0, 'polyp': 1},
                            set0, '.png', dataset_name=dataset_name, description='adapted dataset from HarDNet-MSEG') 

    print("Done.")