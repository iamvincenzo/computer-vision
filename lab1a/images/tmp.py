from numpy.lib.stride_tricks import as_strided
import cv2
import numpy as np


def strided_convolution(image, weight, stride = 2):

    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, 0)

    im_h, im_w = image.shape
    f_h, f_w = weight.shape

    out_shape = (1 + (im_h + 2 - f_h) // stride, 1 +
                (im_w + 2 - f_w) // stride, f_h, f_w)
    out_strides = (image.strides[0] * stride, image.strides[1]
                * stride, image.strides[0], image.strides[1])
    windows = as_strided(image, shape=out_shape, strides=out_strides)

    return np.tensordot(windows, weight, axes=((2, 3), (0, 1)))


# read image
img = cv2.imread('lena.jpg')
old_image_height, old_image_width, channels = img.shape

# create new image of desired size and color (blue) for padding
new_image_width = 300
new_image_height = 300
color = (255, 0, 0)
result = np.full((new_image_height, new_image_width,
                 channels), color, dtype=np.uint8)

# compute center offset
x_center = (new_image_width - old_image_width) // 2
y_center = (new_image_height - old_image_height) // 2

# copy img image into center of result image
result[y_center:y_center+old_image_height,
       x_center:x_center+old_image_width] = img

# view result
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# save result
cv2.imwrite("lena_centered.jpg", result)
