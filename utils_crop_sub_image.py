import cv2
import numpy as np
import tensorflow as tf
import os 
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt


__DEBUG__ = False
__DEBUG__SHOW__IMAGE = False
__DEBUG__SHOW_CV = False
__DEBUG__imagePath = "./__DEBUG__"

# Get the Image
def imread(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    return img

def imsave(image, path, config):
    #checkimage(image)
    # Check the check dir, if not, create one
    if not os.path.isdir(os.path.join(os.getcwd(),config.result_dir)):
        os.makedirs(os.path.join(os.getcwd(),config.result_dir))

    # NOTE: because normial, we need mutlify 255 back    
    cv2.imwrite(os.path.join(os.getcwd(),path),image * 255.)
    print("imshow")
    #checkimage(image)
    if __DEBUG__SHOW__IMAGE :
        imBGR2RGB = cv2.cvtColor(image.astype(np.float32),cv2.COLOR_BGR2RGB)
        plt.imshow(imBGR2RGB)
        plt.show()

    
def checkimage(image,path):
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_YCR_CB2BGR)
    cv2.imwrite(path,image)
    cv2.imshow(path,image)
    cv2.waitKey(0)

def modcrop(img, scale =3):
    """
        To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
    """
    # Check the image is gray

    if len(img.shape) ==3:
        h, w, _ = img.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        img = img[0:h, 0:w, :]
    else:
        h, w = img.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        img = img[0:h, 0:w]
    return img

def downSample(image,scale=3):
    h,w,_ =image.shape
    h_n = h//scale
    w_n = w//scale
    img = np.full((h_n,w_n,_),0)
    
    for i in range(0,h):
        for j in range(0,w):
            if(i % scale==0 and j % scale==0):
                img[i//scale][j//scale] = image[i][j]
    return img

def checkpoint_dir(is_train, checkpoint_dir):
    if is_train:
        return os.path.join('./{}'.format(checkpoint_dir), "train.h5")
    else:
        return os.path.join('./{}'.format(checkpoint_dir), "test.h5")

def preprocess(path ,scale = 3):
    img = imread(path)
    label_ = modcrop(img, scale)
    down_sampling_img = downSample(label_, scale)
    down_sampling_img = down_sampling_img.astype(np.uint8)

    #bicbuic_img = cv2.resize(label_,None,fx = 1.0/scale ,fy = 1.0/scale, interpolation = cv2.INTER_CUBIC)# Resize by scaling factor
    input_ = cv2.resize(down_sampling_img,None,fx = scale ,fy=scale, interpolation = cv2.INTER_CUBIC)
    # Resize by scaling factor
    cv2.imwrite(os.path.join('./{}'.format(__DEBUG__imagePath + "/debug.png")), input_)

    print(input_.shape)
    print(label_.shape)
    return input_,label_

def prepare_data(dataset="Train",Input_img=""):
    """
        Args:
            dataset: choose train dataset or test dataset
            For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp',..., 't99.bmp']
    """
    if dataset == "Train":
        data_dir = os.path.join(os.getcwd(), dataset) # Join the Train dir to current directory
        data = glob.glob(os.path.join(data_dir, "*.bmp")) # make set of all dataset file path
    else:
        if Input_img !="":
            data = [os.path.join(os.getcwd(),Input_img)]
        else:
            data_dir = os.path.join(os.path.join(os.getcwd(), dataset), "Set5")
            data = glob.glob(os.path.join(data_dir, "*.bmp")) # make set of all dataset file path
    return data

def load_data(is_train, test_img):
    if is_train:
        data = prepare_data(dataset="Train")
    else:
        if test_img != "":
            return prepare_data(dataset="Test",Input_img=test_img)
        data = prepare_data(dataset="Test")
    return data

def make_sub_data(data, scale, is_train=True):
    """
        Make the sub_data set
        Args:
            data : the set of all file path 
            padding : the image padding of input to label
            config : the all flags
    """
    input_sequence = []
    label_sequence = []
    for i in range(len(data)):
        if is_train:
            input_, label_, = preprocess(data[i], scale) # do bicbuic
        else: # Test just one picture
            input_, label_, = preprocess(data[i], scale) # do bicbuic
        # input_ = imread(data[i])
        if len(input_.shape) == 3: # is color
            h, w, c = input_.shape
        else:
            h, w = input_.shape # is grayscale
        #checkimage(input_)
                # Add to sequence
        print(input_.shape)

        input_sequence.append(input_)
        label_sequence.append(label_)
        
        
    # NOTE: The nx, ny can be ignore in train
    return input_sequence,label_sequence


def read_data(path):
    """
        Read h5 format data file

        Args:
            path: file path of desired file
            data: '.h5' file format that contains  input values
            label: '.h5' file format that contains label values 
    """
    with h5py.File(path, 'r') as hf:
        input_ = np.array(hf.get('input'))
        label_ = np.array(hf.get('label'))
        return input_, label_

def make_data_hf(input_,label_,is_train, checkpoint_dir):
    """
        Make input data as h5 file format
        Depending on "is_train" (flag value), savepath would be change.
    """
    # Check the check dir, if not, create one
    if not os.path.isdir(os.path.join(os.getcwd(),checkpoint_dir)):
        os.makedirs(os.path.join(os.getcwd(),checkpoint_dir))

    if is_train:
        savepath = os.path.join(os.getcwd(), checkpoint_dir + '/train.h5')
    else:
        savepath = os.path.join(os.getcwd(), checkpoint_dir + '/test.h5')

    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('input', data=input_)
        hf.create_dataset('label', data=label_)


def input_setup(scale, is_train, checkpoint_dir):
    """
        Read image make sub-images -> saved as a h5 file format
    """

    # Load data path, if is_train False, get test data
    data = load_data(is_train, '')

    # padding = abs(config.image_size - config.label_size)//2#need double "/" -> "//" from "6.0" -> "6"

    # Make sub_input and sub_label, if is_train false more return nx, ny
    input_sequence,label_sequence= make_sub_data(data, scale, is_train)


    # Make list to numpy array. With this transform
    arrinput = np.asarray(input_sequence) # [?, 33, 33, 3]
    arrlabel = np.asarray(label_sequence) # [?, 21, 21, 3]

    print(arrinput.shape)
    print(arrlabel.shape)
    make_data_hf(arrinput,arrlabel,is_train, checkpoint_dir)

    

