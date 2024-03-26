# camera-ready

import sys

sys.path.append("/root/deeplabv3")
from deeplabv3.datasets import DatasetVal # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)

sys.path.append("/root/deeplabv3/model")
from deeplabv3.model.deeplabv3 import DeepLabV3

sys.path.append("/root/deeplabv3/utils")
from deeplabv3.utils.utils import label_img_to_color

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2 as cv2
from PIL import Image
from torchvision.transforms.transforms import ToTensor


batch_size = 2

network = DeepLabV3("eval_val", project_dir="E:\\dataset\\Synthetic_Rain_Datasets\\deeplabv3").cuda()
network.load_state_dict(torch.load(r"E:\code\deeplabv3\deeplabv3\pretrained_models\model_13_2_2_2_epoch_580.pth"))
val_dataset = DatasetVal(cityscapes_data_path="E:\\dataset\\Synthetic_Rain_Datasets\\deeplabv3\\data\\cityscapes",
                         cityscapes_meta_path="E:\\dataset\\Synthetic_Rain_Datasets\\deeplabv3\\data\\cityscapes\\meta")



num_val_batches = int(len(val_dataset)/batch_size)
print ("num_val_batches:", num_val_batches)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=batch_size, shuffle=False,
                                         num_workers=1)

with open("E:\\dataset\\Synthetic_Rain_Datasets\\deeplabv3\\data\\cityscapes\\meta\\class_weights.pkl", "rb") as file: # (needed for python3)
    class_weights = np.array(pickle.load(file))
class_weights = torch.from_numpy(class_weights)
class_weights = Variable(class_weights.type(torch.FloatTensor)).cuda()

# loss function
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

def val_epoch():
    network.eval()  # (set in evaluation mode, this affects BatchNorm and dropout)
    batch_losses = []
    # for step, (imgs, label_imgs, img_ids) in enumerate(val_loader):
    # img_path = r"D:\dataset\Synthetic_Rain_Datasets\train\target\104.jpg"
    img_path = r"C:\Users\yunjeongyong\Desktop\pic10.png"
    img = cv2.imread(img_path, -1)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    # img = cv2.imread(img_path, -1)
    print('sdfsdfsfd', img.shape)
    # resize img without interpolation (want the image to still match
    # label_img, which we resize below):
    # img = cv2.resize(img, (1024, 512), interpolation=cv2.INTER_NEAREST)  # (shape: (512, 1024, 3))
    # normalize the img (with the mean and std for the pretrained ResNet):
    img = img / 255.0
    print('img', img.shape)
    img = img - np.array([0.485, 0.456, 0.406])
    img = img / np.array([0.229, 0.224, 0.225])  # (shape: (512, 1024, 3))
    img = np.transpose(img, (2, 0, 1))  # (shape: (3, 512, 1024))
    img = img.astype(np.float32)
    imgs = torch.from_numpy(img)  # (shape: (3, 512, 1024))

    # imgs = ToTensor()(imgs)
    img_ids = ['4']
    with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
        imgs = imgs.unsqueeze(0)
        imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_h, img_w))
        print('imgs', imgs.shape)
        # label_imgs = Variable(label_imgs.type(torch.LongTensor)).cuda() # (shape: (batch_size, img_h, img_w))

        outputs = network(imgs) # (shape: (batch_size, num_classes, img_h, img_w))

        # compute the loss:
        # loss = loss_fn(outputs, label_imgs)
        # loss_value = loss.data.cpu().numpy()
        # batch_losses.append(loss_value)

        ########################################################################
        # save data for visualization:
        ########################################################################
        outputs = outputs.data.cpu().numpy() # (shape: (batch_size, num_classes, img_h, img_w))
        pred_label_imgs = np.argmax(outputs, axis=1) # (shape: (batch_size, img_h, img_w))
        pred_label_imgs = pred_label_imgs.astype(np.uint8)

        for i in range(pred_label_imgs.shape[0]):
            if i == 0:
                pred_label_img = pred_label_imgs[i] # (shape: (img_h, img_w))
                img_id = img_ids[i]
                img = imgs[i] # (shape: (3, img_h, img_w))

                img = img.data.cpu().numpy()
                img = np.transpose(img, (1, 2, 0)) # (shape: (img_h, img_w, 3))
                img = img*np.array([0.229, 0.224, 0.225])
                img = img + np.array([0.485, 0.456, 0.406])
                img = img*255.0
                img = img.astype(np.uint8)

                cv2.imwrite(network.model_dir + "/" + img_id + "_overlayed_nocolor.png", pred_label_img)

                pred_label_img_color = label_img_to_color(pred_label_img)
                overlayed_img = 0.35*img + 0.65*pred_label_img_color
                overlayed_img = overlayed_img.astype(np.uint8)

                cv2.imwrite(network.model_dir + "/" + img_id + "_overlayed.png", overlayed_img)

    # val_loss = np.mean(batch_losses)
    # print ("val loss: %g" % val_loss)
    # return val_loss
    return 0


if __name__=='__main__':
    val_loss = val_epoch()
    # for epoch in range(start_epoch, num_epochs):
    #     epoch_loss = train_epoch(epoch)
    #
    #     if (epoch + 1) % val_freq == 0:
    #         epoch_loss = val_epoch(epoch)
