import cv2
import numpy as np
from PIL import Image


# def make_colormap(num=256):
#     def bit_get(val, idx):
#         return (val >> idx) & 1
#
#     colormap = np.zeros((num, 3), dtype=int)
#     ind = np.arange(num, dtype=int)
#
#     for shift in reversed(list(range(8))):
#         for channel in range(3):
#             colormap[:, channel] |= bit_get(ind, channel) << shift
#         ind >>= 3
#
#     return colormap
#
#
# cmap = make_colormap(256).tolist()
# palette = [value for color in cmap for value in color]
# print(cmap, "\n", palette)
#
#
# # Image data to save = image_data(numpy.ndarray)
# image_data = Image.open(r"D:\dataset\Synthetic_Rain_Datasets\deeplabv3\data\cityscapes\leftImg8bit\train\aachen\aachen_000000_000019_leftImg8bit.png")
# image_data.show()
# label_img = np.array(image_data)
#
# # if image array has BGR order
# label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)
# # Create an unsigned-int (8bit) empty numpy.ndarray of the same size (shape)
# img_png = np.zeros((label_img.shape[0], label_img.shape[1]), np.uint8)
#
# # Assign index to empty ndarray. Finding pixel location using np.where.
# # If you don't use np.where, you have to run a double for-loop for each row/column.
# for index, val_col in enumerate(cmap):
#     img_png[np.where(np.all(label_img == val_col, axis=-1))] = index
#
# # Convert ndarray with index into Image object (P mode) of PIL package
# img_png = Image.fromarray(img_png).convert('P')
# # Palette information injection
# img_png.putpalette(palette)
# # save image
# img_png.save('output.png')

from deeplabv3.model.deeplabv3 import DeepLabV3
from torch.autograd import Variable
from deeplabv3.utils.utils import label_img_to_color
from torchvision.transforms import ToTensor
import torch


network = DeepLabV3("eval_val", project_dir="E:\\dataset\\Synthetic_Rain_Datasets\\deeplabv3").cuda()
network.load_state_dict(torch.load(r"E:\code\deeplabv3\deeplabv3\pretrained_models\model_13_2_2_2_epoch_580.pth"))
network.eval()

img = Image.open(r"E:\dataset\Synthetic_Rain_Datasets\deeplabv3\data\cityscapes\leftImg8bit\train\aachen\aachen_000000_000019_leftImg8bit.png")
img = ToTensor()(img)
# output = network(img)

with torch.no_grad():  # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
    img = Variable(img).cuda()  # (shape: (batch_size, 3, img_h, img_w))
    outputs = network(img)  # (shape: (batch_size, num_classes, img_h, img_w))

    ########################################################################
    # save data for visualization:
    ########################################################################
    outputs = outputs.data.cpu().numpy()  # (shape: (batch_size, num_classes, img_h, img_w))
    pred_label_imgs = np.argmax(outputs, axis=1)  # (shape: (batch_size, img_h, img_w))
    pred_label_imgs = pred_label_imgs.astype(np.uint8)

    for i in range(pred_label_imgs.shape[0]):
        if i == 0:
            pred_label_img = pred_label_imgs[i]  # (shape: (img_h, img_w))

            img = img.data.cpu().numpy()
            img = np.transpose(img, (1, 2, 0))  # (shape: (img_h, img_w, 3))
            img = img * np.array([0.229, 0.224, 0.225])
            img = img + np.array([0.485, 0.456, 0.406])
            img = img * 255.0
            img = img.astype(np.uint8)

            pred_label_img_color = label_img_to_color(pred_label_img)
            overlayed_img = 0.35 * img + 0.65 * pred_label_img_color
            overlayed_img = overlayed_img.astype(np.uint8)

            cv2.imwrite(network.model_dir + "/" + "overlayed.png", overlayed_img)
            # cv2.imwrite()
