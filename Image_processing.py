import matplotlib.pyplot as plt
import imageio
import skimage.util as util
import skimage.color as color
import skimage.feature as feature
import skimage.filters as filters
import skimage.transform as transform
import numpy as np
import cv2

DEBUGGING = False

# data path
DATA_DIR  = 'E:/KCL/coursework/cousework2-DM/data/data/image_data/'
DATA_FILE_AVG = 'avengers_imdb.jpg'
DATA_FILE_BUS = 'bush_house_wikipedia.jpg'
DATA_FILE_FOR = 'forestry_commission_gov_uk.jpg'
DATA_FILE_ROL = 'rolland_garros_tv5monde.jpg'

# 2.1
# Determine the size of the avengers imdb.jpg image. Produce a grayscale and a black-and-white representation of it.
# read image from a file
im_avg = imageio.imread(DATA_DIR + DATA_FILE_AVG)
# Get the size of the avengers imdb.jpg image
width, height, channel = im_avg.shape
print('The width, height and channel of the avengers_imdb.jpg image are {}, {} and {}, respectively.'.format(width, height, channel))

# convert the image to greyscale
img_avg = color.rgb2gray(im_avg)
# plot grey scale image
fig_gray = plt.figure(figsize = (6,12))
plt.imshow(img_avg, cmap=plt.cm.gray, interpolation='nearest')
plt.axis('off')
plt.savefig(DATA_DIR + '1.1 Greyscale avengers_imdb.jpg')
# plt.show()

# convert the image to Black&White
threshold = filters.threshold_otsu(img_avg)
print('Otsu method threshold = ', threshold)
binary_img_avg = img_avg > threshold
# plot black and white image
fig_bw = plt.figure(figsize = (6, 12))
plt.imshow(binary_img_avg, cmap=plt.cm.gray, interpolation='nearest')
plt.axis('off')
plt.savefig(DATA_DIR + '1.2 Black & White image avengers_imdb.jpg')

# 2.2
# Add Gaussian random noise in bush house wikipedia.jpg (with variance 0.1)
mean = 0
var = 0.1
sigma = 1

# read bush house wikipedia.jpg image
im_bus = imageio.imread(DATA_DIR+DATA_FILE_BUS)

# add gaussian noise to a image with varrience 0.1
im_bus_noisy = util.random_noise(im_bus, mode='gaussian', var=var)

# filter the gaussian mask to the image
gaussian_im_bus = cv2.GaussianBlur(im_bus_noisy, (9, 9), sigma, borderType=cv2.BORDER_CONSTANT)

# filter the uniform smoothing mask to the image
blur_im_bus = cv2.blur(im_bus_noisy, (9, 9))

fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=4, figsize=(8, 3), sharex=True, sharey=True)
# show original image
ax0.imshow(im_bus)
ax0.axis('off')
ax0.set_title("Original image")

# show the image wirh gaussian noisy
ax1.imshow(im_bus_noisy)
ax1.axis('off')
ax1.set_title("Image with gaussian noise")

# show the perturbed image with a Gaussian mask (sigma equal to 1)
ax2.imshow(gaussian_im_bus)
ax2.axis('off')
ax2.set_title('Gaussian mask')

# show the image with a smoothing mask
ax3.imshow(blur_im_bus)
ax3.axis('off')
ax3.set_title('Smoothing mask')

fig.tight_layout()
plt.savefig(DATA_DIR + '2 modified bush house wikipedia.jpg')

# 2.3
# Divide forestry commission gov uk.jpg into 5 segments using k-means segmentation.
im_for = imageio.imread(DATA_DIR+DATA_FILE_FOR)

# flatten the image
im_for_flag = im_for.reshape((-1, 3))
# increase the data accuracy
im_for_flag = np.float32(im_for_flag)

# build a kmeans model for it
# two criteria: the desired accuracy + the rule of max iteration
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 20, 1.0)
# set random centers in kmeans
flags = cv2.KMEANS_RANDOM_CENTERS

c1, labels, centers = cv2.kmeans(im_for_flag, 5, None, criteria, 10, flags)
# turn into integers
centers = np.uint8(centers)
# append its channel and output the component kmeans image data.
output = centers[labels.flatten()]
# reshape data
im_for_out = output.reshape((im_for.shape))

# plot results
fig_kmeans = plt.figure(figsize=(6, 4))

# show the image with kmeans
plt.imshow(im_for_out)
plt.axis('off')

plt.savefig(DATA_DIR + '3 modified forestry commission gov uk.jpg')

# 2.4
# apply Hough transform on rolland garros tv5monde.jpg.
im_rol = imageio.imread(DATA_DIR + DATA_FILE_ROL)

# convert the image to greyscale
img_rol = color.rgb2gray(im_rol)

# perform Canny edge detection
edges = feature.canny(img_rol)

# plot canny edges image
fig_canny = plt.figure(figsize=(10, 6))
plt.imshow(edges, cmap=plt.cm.gray, interpolation='nearest')
plt.axis('off')
plt.title('canny edges')
plt.savefig(DATA_DIR + '4.1 Canny edges rolland garros tv5monde.jpg')

# apply classic straight-line Hough transform
lines = transform.probabilistic_hough_line(edges, threshold=70, line_length=60, line_gap=3)

# plot each line through Hough transform
fig_hough = plt.figure(figsize=(10, 6))
for line in lines:
    p0, p1 = line
    plt.plot((p0[0], p1[0]), (p0[1], p1[1]))
plt.xlim((0, img_rol.shape[1]))
plt.ylim((img_rol.shape[0], 0))
plt.title('Probabilistic Hough')
plt.savefig(DATA_DIR + '4.2 Hough transform rolland garros tv5monde.jpg')

plt.show()
