import cv2
import numpy as np
from PIL import Image


def make_colormap(num=256):
    def bit_get(val, idx):
        return (val >> idx) & 1

    colormap = np.zeros((num, 3), dtype=int)
    ind = np.arange(num, dtype=int)

    for shift in reversed(list(range(8))):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3

    return colormap


cmap = make_colormap(256).tolist()
palette = [value for color in cmap for value in color]
print(cmap, "\n", palette)


# Image data to save = image_data(numpy.ndarray)
image_data = Image.open(r"D:\dataset\Synthetic_Rain_Datasets\deeplabv3\data\cityscapes\leftImg8bit\train\aachen\aachen_000000_000019_leftImg8bit.png")
# image_data.show()
label_img = np.array(image_data)

# if image array has BGR order
label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)
# Create an unsigned-int (8bit) empty numpy.ndarray of the same size (shape)
img_png = np.zeros((label_img.shape[0], label_img.shape[1]), np.uint8)

# Assign index to empty ndarray. Finding pixel location using np.where.
# If you don't use np.where, you have to run a double for-loop for each row/column.
for index, val_col in enumerate(cmap):
    img_png[np.where(np.all(label_img == val_col, axis=-1))] = index

# Convert ndarray with index into Image object (P mode) of PIL package
img_png = Image.fromarray(img_png).convert('P')
# Palette information injection
img_png.putpalette(palette)
# save image
# img_png.save('output.png')
img_png.show()

