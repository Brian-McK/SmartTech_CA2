import glob
from os.path import isfile, join

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import random
from PIL import Image
import cv2
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model

import pandas as pd
import cv2
import os
import ntpath

from matplotlib import image as mpimg
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
from keras.applications import ResNet50

resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

for layer in resnet.layers[:-4]:
    layer.trainable = False

for layer in resnet.layers:
    print(layer, layer.trainable)


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail





def preprocess_img(img):
    img = mpimg.imread(img)
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (100, 100))
    img = img/255
    return img


def preprocess_img_no_imread(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (100, 100))
    img = img/255
    return img


def nvidia_model():
    model = Sequential()
    model.add(Convolution2D(24, (5, 5), strides=(2, 2), input_shape=(100, 100, 3), activation='elu'))
    model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    #model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    #model.add(Dropout(0.5))
    model.add(Dense(50, activation='elu'))
    #model.add(Dropout(0.5))
    model.add(Dense(10, activation='elu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=optimizer)
    return model


def zoom(image_to_zoom):
    zoom_func = iaa.Affine(scale=(1, 1.3))
    z_image = zoom_func.augment_image(image_to_zoom)
    return z_image


def pan(image_to_pan):
    pan_func = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
    pan_image = pan_func.augment_image(image_to_pan)
    return pan_image


def img_brightness(image_to_brighten):
    hsv = cv2.cvtColor(image_to_brighten, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:, :, 2] = hsv[:, :, 2] * ratio
    bright_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return bright_image


def img_random_flip(image_to_flip, steering_angle):
    flipped_image = cv2.flip(image_to_flip, 1)
    steering_angle = -steering_angle
    return flipped_image, steering_angle
def random_translate(image, steering_angle, range_x, range_y):
    """
    Randomly shift the image virtially and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle


def random_shadow(image):
    """
    Generates and adds random shadow
    """
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = 100 * np.random.rand(), 0
    x2, y2 = 100 * np.random.rand(), 100
    xm, ym = np.mgrid[0:100, 0:100]

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line:
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

def random_augment(image_to_augment, steering_angle):
    augment_image = mpimg.imread(image_to_augment)
    range_x = 100
    range_y = 10
    if np.random.rand() < 0.5:
        augment_image = random_translate(image_to_augment, steering_angle,range_x,range_y)
    if np.random.rand() < 0.5:
        augment_image = random_shadow(augment_image)
    if np.random.rand() < 0.5:
        augment_image = zoom(augment_image)
    if np.random.rand() < 0.5:
        augment_image = pan(augment_image)
    if np.random.rand() < 0.5:
        augment_image = img_brightness(augment_image)
    if np.random.rand() < 0.5:
        augment_image, steering_angle = img_random_flip(augment_image, steering_angle)
    return augment_image, steering_angle


def batch_generator(image_paths, steering_ang, batch_size, is_training):
    while True:
        batch_img = []
        batch_steering = []
        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths)-1)
            if is_training:
                im, steering = random_augment(image_paths[random_index], steering_ang[random_index])
            else:
                im = mpimg.imread(image_paths[random_index])
                steering = steering_ang[random_index]

            im = preprocess_img_no_imread(im)
            batch_img.append(im)
            batch_steering.append(steering)
        yield np.asarray(batch_img), np.asarray(batch_steering)

dir = "C:\\smart_techn"
dir1 = "C:\\smart_tech"
datadir = "C:\\smart_techn\\mountain2"
datadir1 = "C:\\smart_techn\\test1"

columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
df = pd.read_csv(os.path.join(datadir, 'driving_log.csv'), names = columns)
data1 = pd.read_csv(os.path.join(datadir, 'driving_log.csv'), names = columns)
print(data1)
data2 = pd.read_csv(os.path.join(datadir1, 'driving_log.csv'), names = columns)
print(data2)

csv_files = [os.path.join(datadir, 'driving_log.csv'), os.path.join(datadir1, 'driving_log.csv')]
concat_csvs = []

for filename in csv_files:
    df1 = pd.read_csv(filename, names=columns)
    concat_csvs.append(df1)
frame = pd.concat(concat_csvs,join='inner', ignore_index=True)
# frame.to_csv("C:\\smart_tech\\testing2.csv")
data3 = pd.read_csv(os.path.join(dir1, 'testing2.csv'), names = columns)


data = pd.concat([data1, data2],join='inner',ignore_index=True)
pd.set_option('display.max_columns', 7)
print(data)

# def load_steering_img(data_dir, data):
#     image_path = []
#     steering = []
#     for i in range(len(data)):
#         indexed_data = data.iloc[i]
#         centre, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
#         image_path.append(os.path.join(data_dir, centre.strip()))
#         steering.append(float(indexed_data[3]))
#         image_path.append(os.path.join(data_dir, left.strip()))
#         steering.append(float(indexed_data[3]) + 0.15)
#         image_path.append(os.path.join(data_dir, right.strip()))
#         steering.append(float(indexed_data[3]) - 0.15)
#     image_paths = np.asarray(image_path)
#     steerings = np.asarray(steering)
#     return image_paths, steerings


# for filename in os.listdir(dir):
#     print("file", filename)
def load_steering1_img(data_dir, df):
    images_path = []
    steering = []
    for filename in os.listdir(data_dir):
        data_path = os.path.join(data_dir + '\\' + filename + '\\IMG')
        for i in range(len(df)):
            indexed_data = df.iloc[i]
            centre, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
            if os.path.exists(os.path.join(data_path, centre.strip())):
                images_path.append(os.path.join(data_path, centre.strip()))
                steering.append(float(indexed_data[3]))

            if os.path.exists(os.path.join(data_path, left.strip())):
                images_path.append(os.path.join(data_path, left.strip()))
                steering.append(float(indexed_data[3]) + 0.15)

            if os.path.exists(os.path.join(data_path, right.strip())):
                images_path.append(os.path.join(data_path, right.strip()))
                steering.append(float(indexed_data[3]) - 0.15)

    images_paths = np.asarray(images_path)
    steerings = np.asarray(steering)
    return images_paths, steerings

# def load_images(dir):
#     images = []
#     for filename in os.listdir(dir):
#         data_path = os.path.join(dir + '\\' + filename + '\\IMG')
#         print(data_path)
#         onlyfiles = [f for f in os.listdir(data_path) if isfile(join(data_path, f))]
#         for n in range(0, len(onlyfiles)):
#             images.append(cv2.imread(join(data_path, onlyfiles[n])))
#     image_arr = np.array(images)
#     return image_arr
#
#
# print(load_images(dir).shape)
# for filename in os.listdir(dir):
#     print(filename)
print(load_steering1_img(dir, df))

data['center'] = data['center'].apply(path_leaf)
data['left'] = data['left'].apply(path_leaf)
data['right'] = data['right'].apply(path_leaf)

num_bins = 25
samples_per_bin = 400
hist, bins = np.histogram(data['steering'], num_bins)
centre = (bins[:-1] + bins[1:])*0.5
plt.bar(centre, hist, width=0.05)
plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))
plt.show()

remove_list=[]
print('Total data: ', len(data))

for j in range(num_bins):
    list_ = []
    for i in range(len(data['steering'])):
        if bins[j] <= data['steering'][i] <= bins[j+1]:
            list_.append(i)
    list_ = shuffle(list_)
    list_ = list_[samples_per_bin:]
    remove_list.extend(list_)

print("Remove: ", len(remove_list))
data.drop(data.index[remove_list], inplace=True)
print("Remaining: ", len(data))

hist, bins = np.histogram(data['steering'], num_bins)
plt.bar(centre, hist, width=0.05)
plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))
plt.show()

image_paths, steerings = load_steering1_img(dir, data)
# count = 0
# count1 = 0
# for j in steerings:
#     print(j)
# image_arr = []
# steer_arr = []
# for i in image_path:
#     if os.path.exists(i):
#         image_arr.append(i)
#         count += 1
#         for j in steerings:
#            count1 +=1
#
#
#
# print(count)
# print(count1)
#
# image_paths = np.asarray(image_arr)
from pathlib import Path

X_train, X_valid, y_train, y_valid = train_test_split(image_paths, steerings, test_size=0.2, random_state=6)
print(f"Training samples {len(X_train)}\n Validation samples {len(X_valid)}")

fig, axes = plt.subplots(1,2, figsize=(12,4))
axes[0].hist(y_train, bins=num_bins, width=0.05, color='blue')
axes[0].set_title("Training set")
axes[1].hist(y_valid, bins=num_bins, width=0.05, color='red')
axes[1].set_title("Validation set")
plt.show()

image = image_paths[100]
original_image = mpimg.imread(image)
preprocessed_image = preprocess_img(image)
fig, axes = plt.subplots(1,2, figsize=(15,10))
fig.tight_layout()
axes[0].imshow(original_image)
axes[0].set_title("Original image")
axes[1].imshow(preprocessed_image)
axes[1].set_title("Preprocessed image")
plt.show()

#X_train = np.array(list(map(preprocess_img, X_train)))
#X_valid = np.array(list(map(preprocess_img, X_valid)))

# plt.imshow(X_train[random.randint(0, len(X_train)-1)])
# plt.axis('off')
# plt.show()
# print(X_train.shape)

image = image_paths[random.randint(0, 1000)]
original_image = mpimg.imread(image)
zoomed_image = zoom(original_image)
fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title("Original Image")
axs[1].imshow(zoomed_image)
axs[1].set_title("Zoomed Image")
plt.show()

image = image_paths[random.randint(0, 1000)]
original_image = mpimg.imread(image)
panned_image = pan(original_image)
fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title("Original Image")
axs[1].imshow(panned_image)
axs[1].set_title("Panned Image")
plt.show()

image = image_paths[random.randint(0, 1000)]
original_image = mpimg.imread(image)
bright_image = img_brightness(original_image)
fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title("Original Image")
axs[1].imshow(bright_image)
axs[1].set_title("Bright Image")
plt.show()

random_index = random.randint(0, 1000)
image = image_paths[random_index]
steering_angle = steerings[random_index]
original_image = mpimg.imread(image)
flipped_image, flipped_angle = img_random_flip(original_image, steering_angle)
fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title("Original Image - " + "Steering Angle: " + str(steering_angle))
axs[1].imshow(flipped_image)
axs[1].set_title("Flipped Image"+ "Steering Angle: " + str(flipped_angle))
plt.show()

ncols = 2
nrows = 10
fig, axs = plt.subplots(nrows, ncols, figsize=(15, 50))
fig.tight_layout()
print(len(image_paths))
for i in range(10):
    rand_num = random.randint(0, len(image_paths) - 1)
    random_image = image_paths[rand_num]
    random_steering = steerings[rand_num]
    original_image = mpimg.imread(random_image)
    augmented_image, steering_angle = random_augment(random_image, random_steering)
    axs[i][0].imshow(original_image)
    axs[i][0].set_title("Original Image")
    axs[i][1].imshow(augmented_image)
    axs[i][1].set_title("Augmented Image")
plt.show()

model = nvidia_model()
print(model.summary())

history = model.fit(batch_generator(X_train, y_train, 200, 1), steps_per_epoch=100, epochs=10, validation_data=batch_generator(X_valid, y_valid, 200, 0), validation_steps=200, verbose=1, shuffle=1)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(["Training", "Validation"])
plt.title('Loss')
plt.xlabel("Epoch")
plt.show()

model.save('model.h5')




















