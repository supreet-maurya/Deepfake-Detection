

#################################################################
###### Test whether a given video is fake or real

#####################################################################################################
###### Import Libraries
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torchvision
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL
import imgaug as ia
import PIL.Image as Image
import albumentations as alb


from PIL import Image
from facenet_pytorch.models.mtcnn import MTCNN
from tqdm.notebook import tqdm
from imgaug import augmenters as iaa
from torchvision import transforms as tt




#####################################################################################################
# data_parallel 
import torchvision.transforms as tt
#####################################################################################################

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
    







########################################################################
#augmentations
########################################################################
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import torchvision.transforms as tt
import PIL

from imgaug import augmenters as iaa
import imgaug as ia
import numpy as np
from fastai.vision.all import *
import albumentations
#####################################################################################################


class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
            iaa.Fliplr(0.5),
            iaa.Affine(rotate=(-20, 20), mode='symmetric'),
            iaa.Sometimes(0.25,
                          iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
                                     iaa.CoarseDropout(0.1, size_percent=0.5)])),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
        ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


def get_train_aug(): return albumentations.Compose([
     albumentations.RandomResizedCrop(224,224),
            albumentations.Transpose(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.ShiftScaleRotate(p=0.5),
            albumentations.HueSaturationValue(
                hue_shift_limit=0.2,
                sat_shift_limit=0.2,
                val_shift_limit=0.2,
                p=0.5
            ),
            albumentations.CoarseDropout(p=0.2),
            albumentations.Cutout(p=0.2)
])




class AlbumentationsTransform(Transform):
    def __init__(self, aug): self.aug = aug
    def encodes(self, img: PILImage):
        aug_img = self.aug(image=np.array(img))['image']
        return PILImage.create(aug_img)











import argparse
#########################################################
#  strategies
#########################################################
import numpy as np

def max(a,b):
    if(a>b):
        return a
    else:
        return b

def ensemble_strategy_1(a,b):
    return (a+b)/2

def ensemble_strategy_2(a,b):
    return max(a,b)



def ensemble_strategy_3(a,b):
    if(a>=0.7 and b>=0.7):
        return max(a,b)
    elif(a>=0.4 and a<= 0.6 and b>=0.4 and b<=0.6):
        return min(a,b)
    elif(a>=0.6 and a<=0.7 and b>=0.6 and b<=0.7):
        return max(a,b)
    else:
        return min(a,b)

def ensemble_strategy_4(a,b,w1=0.4,w2=0.6):
    return (a*w1+b*w2)


def confident_strategy_4(preds):
    preds.sort()
    average=0
    start=int(len(preds)/3)
    end = len(preds)
    for i in range(start,end):
        average+=preds[i]
    return average/(end-start)


def ensemble_strategy_5(a,b):
    if(a>=0.7 and b>=0.7):
        return max(a,b)
    if(a<=0.25 and b<=0.25):
        return min(a,b)
    else:
        return (a+b)/2

confident = lambda p: np.mean(np.abs(p-0.5)*2) >= 0.7
label_spread = lambda x: x-np.log10(x) if x >= 0.8 else x


def confident_strategy_1(pred):
    return np.mean(pred)

def confident_strategy_2(pred, t=0.8):
    pred = np.array(pred)
    sz = len(pred)
    fakes = np.count_nonzero(pred > t)
    # 11 frames are detected as fakes with high probability
    if fakes > sz // 2.5 and fakes > 11:
        return np.mean(pred[pred > t])
    elif np.count_nonzero(pred < 0.2) > 0.9 * sz:
        return np.mean(pred[pred < 0.2])
    else:
        return np.mean(pred)

def confident_strategy_3(preds):
	#
	# If there is a fake id and we're confident,
	# return spreaded fake score, otherwise return
	# the original fake score.
	# If everyone is real and we're confident return
	# the minimum real score, otherwise return the
	# mean of all predictions.
	#
	preds = np.array(preds)
	p_max = np.max(preds)
	if p_max >= 0.8:
		if confident(preds):
			return label_spread(p_max)
		return p_max
	if confident(preds):
		return np.min(preds)
	return np.mean(preds)










#####################################################################################################
####### Training And Testing Time TransformationsS
#####################################################################################################
stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
transforms = torchvision.transforms.Compose([
    ImgAugTransform(),
    lambda x: PIL.Image.fromarray(x),
    torchvision.transforms.RandomVerticalFlip(),
    tt.RandomHorizontalFlip(),
                         tt.RandomResizedCrop((224,224),interpolation=2),
                         tt.ToTensor(),
                         tt.Normalize(*stats,inplace=True)
                  ])


train_tfms=transforms
aug_3 = alb.Compose([
         alb.RandomResizedCrop(224,224)
        ])






"""Ouputs a Score depiciting how fake the video is"""

def fake_or_real(model_1=None,model_2=None,ensemble_strat=1,conf_strat=1,video_path="F",per_frame=10,device="cpu"):
    if(model_1==None and model_2==None):
        print("Load Model Correctly !!!! ")
        exit()

    total_frames=0
    total_evaluated_frames=0
    total_fake_frames=0
    total_real_frames=0
    v_cap = cv2.VideoCapture(video_path)#### Load Video
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))##### Count Number Frames
    total_frames=v_len
    k=1
    d23=min(v_len-2,per_frame)##### Every nth frame to evaluate.If d23 is 5,every 5th frame is tested and others are skipped
    y_preds_2=[]
    y_preds_2.append(0)

    for i in range(v_len):
        # Load frame
        d21=0
        success = v_cap.grab()
        if i % d23 == 0:
            success, frame_4 = v_cap.retrieve()##### Loading the Frame
        else:
            continue
        if not success:
            continue
        mtcnn = MTCNN(keep_all=True,min_face_size=176,thresholds=[0.85,0.90,0.90],device=device)##### Loading the Face Detector
        frame = cv2.cvtColor(frame_4, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame.astype(np.uint8))

        frame=img

        # Detect face
        boxes,landmarks = mtcnn.detect(frame, landmarks=False)###### Detecting the Faces in a given Frame

        try:
            if(len(boxes)!=None):
                pass
        except:
            continue
        total_evaluated_frames = total_evaluated_frames+1

        b1 = img

        try:
            for i in range(0,len(boxes)):
                x,y,width,height =boxes[i]

                ##### Crop Faces from frames
                max_width,max_height = b1.size
                boxes[i][0]-=width/10
                boxes[i][1]-=height/10
                boxes[i][0] = max(0,boxes[i][0])
                boxes[i][1] = max(0,boxes[i][1])
                boxes[i][2] =min(boxes[i][2]+(width/10),max_width)
                boxes[i][3] =min(boxes[i][3]+(height/10),max_height)
                b4 = b1.crop(boxes[i])

                if(model_1 !=None):
                    img = train_tfms(b4)##### Apply Transformations
                    img = img.unsqueeze(0)##### Converting the Single Image into a batch of Sinle Image-----(3,380,380) to(1,3,380,380)
                    img = to_device(img,device)##### Put the Image On the GPU if available
                    out_3 = model_1(img)###### Pass through the model
                    out_3=torch.softmax(out_3.squeeze(),0)
                    out_5 = out_3.cpu().detach().numpy()[0]
                else:
                    out_5=0
                if(model_2 != None):
                    b4.save("test_face_image.jpg")##### Save Cropped Image

                    image = plt.imread("test_face_image.jpg")
                    #print("ghnd")

                    image = Image.fromarray(image).convert('RGB')
                    #print("gfnk")
                    image = aug_3(image=np.array(image))['image']
                    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
                    image = torch.tensor(image, dtype=torch.float)


                    #img = train_tfms(b4)##### Apply Transformations
                    img = image.unsqueeze(0)##### Converting the Single Image into a batch of Sinle Image-----(3,380,380) to(1,3,380,380)
                    img = to_device(img,device)##### Put the Image On the GPU if available
                    out_3 = model_2(img)###### Pass through the model
                    out_3=torch.softmax(out_3.squeeze(),0)
                    out_6 = out_3.cpu().detach().numpy()[1]
                else:
                    out_6=0

                if(model_1==None):
                    out_7=out_6
                elif(model_2==None):
                    out_7=out_5
                else:

                    if ensemble_strat==1:
                        out_7 = ensemble_strategy_1(out_5,out_6)
                        out_7 = (out_5+out_6)/2
                    elif ensemble_strat==2:
                        out_7 = ensemble_strategy_2(out_5,out_6)
                    elif ensemble_strat==3:
                        out_7 = ensemble_strategy_3(out_5,out_6)
                    elif ensemble_strat==4:
                        out_7 = ensemble_strategy_4(out_5,out_6)
                    elif ensemble_strat==5:
                        out_7 = ensemble_strategy_5(out_5,out_6)
                    else:
                        out_7 = (out_5+out_6)/2

                y_preds_2.append(out_7)


                k=k+1

                if(out_7>0.5):
                    total_fake_frames+=1
                else:
                    total_real_frames+=1
        except:
            y_preds_2.append(0.0)##### If no faces found ,then append 0.0 to the score

    if conf_strat==1:
        y_preds_3 = np.array(y_preds_2).mean()
    elif conf_strat==2:
        y_preds_3 = confident_strategy_2(np.array(y_preds_2))
    elif conf_strat==3:
        y_preds_3 = confident_strategy_3(np.array(y_preds_2))
    else:
        y_preds_3 = confident_strategy_4(np.array(y_preds_2))

    # return {
    #     "status": "Fake" if y_preds_3 >= 0.5 else "Real",
    #     "mean_fake_score": y_preds_3,
    #     "total_frames": total_frames,
    #     "total_evaluated_frames": total_evaluated_frames,
    #     "total_fake_frames": total_fake_frames,
    #     "total_real_frames": total_real_frames
    # }

    return y_preds_3,total_frames,total_evaluated_frames,total_fake_frames,total_real_frames#### Return a score depicitng how fake the video is









if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Predict Video is Fake Or Real")
    parser.add_argument("--model-name", help="Name of the Model",nargs="?",type=str,default="F")
    parser.add_argument("--video-path", help="Name of the Video",nargs="?",type=str,default="F")
    parser.add_argument("--ensemble-strategy", help="Ensemble Strategy To Follow",nargs="?",type=int,default=1)
    parser.add_argument("--confident-strategy", help="Strategy To Decide Whether Video Is Fake Or Not",nargs="?",type=int,default=1)
    parser.add_argument("--per-frame", help="Epochs",nargs="?",type=int,default=10)



    args = parser.parse_args()

    print(args.video_path)

    if(args.video_path=="F"):
        print("xxxxxxxxxx---------Path To Video Not Given---------xxxxxxxxxx")
        exit()
    if(args.confident_strategy<1 or args.confident_strategy>4):
        print("xxxxxxxxxx---------Incorrect Value of Confident Strategy Given---------xxxxxxxxxx")
        exit()

    if(args.ensemble_strategy <1 or args.ensemble_strategy>5):
        print("xxxxxxxxxx---------Incorrect Value of Ensemble Strategy Given---------xxxxxxxxxx")
        exit()

    print(os.path.join("test_video",args.video_path))

    video_folder = os.path.abspath("test_video")
    video_path = os.path.join(video_folder, args.video_path)
    print(video_folder)
    print(video_path)

    if not os.path.isfile(video_path):
        print(f"Video not found: {video_path}")
        exit()


    # if(not os.path.isfile(os.path.join("test_video",args.video_path))):
    #     print('Video does not exist in the folder')
    #     exit()
    ####### Check if Path to video doesn't exist
    #elif():
    #    print("xxxxxxxxxx---------Path To Video Doesn't Exist---------xxxxxxxxxx")
    #    exit()
    model_1=None
    model_2=None
    device=get_default_device()
    if(args.model_name=="F" or args.model_name=="model_1.pth"):
        print("------------------------Model Path Not Given,Loading Default Model------------------------")
        model_1= torch.load(os.path.join("Trained_Models","model_1.pth"), map_location=device)
        model_1.to(device)##### Put Model on the CPU/GPU
        model_1.eval()##### Start the Evaluation Mode for the Pytorch Model

    elif(args.model_name=="model_2.pth"):
        model_2= torch.load(os.path.join("Trained_Models",args.model_name), map_location=device)
        model_2.to(device)##### Put Model on the CPU/GPU
        model_2.eval()##### Start the Evaluation Mode for the Pytorch Model
        print("------------------------Model Loaded Correctly------------------------")

    elif(args.model_name=="ensemble"):
        print("----------------------------Model Loading----------------------------")
        model_1= torch.load(os.path.join("Trained_Models","model_1.pth"), map_location=device)
        model_1.to(device)##### Put Model on the CPU/GPU
        model_1.eval()##### Start the Evaluation Mode for the Pytorch Model

        model_2= torch.load(os.path.join("Trained_Models","model_2.pth"), map_location=device)
        model_2.to(device)##### Put Model on the CPU/GPU
        model_2.eval()##### Start the Evaluation Mode for the Pytorch Model
        print("------------------------Both Models Loaded Correctly------------------------")
    else:
        print("Model Name not correctly given, check docs!!! ")
        exit()


    b58,total_frames,total_evaluated_frames,total_fake_frames,total_real_frames = fake_or_real(model_1=model_1,model_2=model_2,ensemble_strat=args.ensemble_strategy,conf_strat=args.confident_strategy,video_path=os.path.join("test_video",args.video_path),per_frame=args.per_frame,device=device)##### Call the Function
    if(b58==0.0):
        print("No Faces Found")

    elif(b58 >= 0.5):
        print("Video is Fake")
    else:
        print("Video is Real")
    print("Mean Fake Score is: ")
    print(b58)
    print("Total Frames are: ")
    print(total_frames)
    print("Total Evaluated Frames are: ")
    print(total_evaluated_frames)
    print("Total Fake Frames are: ")
    print(total_fake_frames)
    print("Total Real Frames are: ")
    print(total_real_frames)
