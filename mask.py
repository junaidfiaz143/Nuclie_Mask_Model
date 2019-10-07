import os
import random
import cv2
import numpy as np
# import matplotlib.pyplot as plt
from tensorflow import keras

seed = 2019
random.seed = seed

class DataGen(keras.utils.Sequence):
  def __init__(self, ids, path, batch_size=8, image_size=128):
    self.ids = ids
    self.path = path
    self.batch_size = batch_size
    self.image_size = image_size

  def __load__(self, id_name):
    image_path = os.path.join(self.path, id_name, "images", id_name) + ".png"
    mask_path = os.path.join(self.path, id_name, "masks/")
    all_masks = os.listdir(mask_path)

    image = cv2.imread(image_path, 1)
    image = cv2.resize(image, (self.image_size, self.image_size))

    mask = np.zeros((self.image_size, self.image_size, 1))

    for name in all_masks:
      _mask_path = os.path.join(mask_path, name)
      _mask_image = cv2.imread(_mask_path, -1)
      _mask_image = cv2.resize(_mask_image, (self.image_size, self.image_size))
      _mask_image = np.expand_dims(_mask_image, axis=-1)
      mask = np.maximum(mask, _mask_image)

    image = image/255.0
    mask = mask/255.0

    return image, mask

  def __getitem__(self, index):
    if (index+1)*self.batch_size > len(self.ids):
      self.batch_size = len(self.ids) - index*self.batch_size

    files_batch = self.ids[index*self.batch_size : (index+1)*batch_size]

    image = []
    mask = []

    for id_name in files_batch:
      _img, _mask = self.__load__(id_name)
      image.append(_img)
      mask.append(_mask)

    image = np.array(image)
    mask = np.array(mask)

    return image, mask

  def __len__(self):
    return int(np.ceil(len(self.ids)/float(self.batch_size)))

image_size = 128
batch_size = 8
epochs = 5

train_path = "train_nuclie/"

train_ids = next(os.walk(train_path))[1]

valid_data_size = 10

valid_ids = train_ids[:valid_data_size]
train_ids = train_ids[valid_data_size:]

gen = DataGen(train_ids, train_path, batch_size=batch_size, image_size=image_size)

image_batch, mask_batch = gen.__getitem__(0)
print(image_batch.shape, mask_batch.shape)

r =  random.randint(0, len(image_batch)-1)

print("random numbers: ", r)

print("length of x: ", len(image_batch))

# fig = plt.figure()
# fig.subplots_adjust(hspace=0.4, wspace=0.4)

# ax = fig.add_subplot(1, 2, 1)
# ax.imshow(image_batch[r])

# ax = fig.add_subplot(1, 2, 2)
# ax.imshow(np.reshape(mask_batch[r], (image_size, image_size)), cmap="gray")

# plt.savefig("mask.png")

def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    return c, p

def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us = keras.layers.UpSampling2D((2, 2))(x)
    concat = keras.layers.Concatenate()([us, skip])
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def MaskModel():
    inputs = keras.layers.Input((image_size, image_size, 3))
    
    p0 = inputs
    c1, p1 = down_block(p0, 16) #128 -> 64
    c2, p2 = down_block(p1, 32) #64 -> 32
    c3, p3 = down_block(p2, 64) #32 -> 16
    c4, p4 = down_block(p3, 128) #16->8
    
    bn = bottleneck(p4, 256)
    
    u1 = up_block(bn, c4, 128) #8 -> 16
    u2 = up_block(u1, c3, 64) #16 -> 32
    u3 = up_block(u2, c2, 32) #32 -> 64
    u4 = up_block(u3, c1, 16) #64 -> 128
    
    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)
    model = keras.models.Model(inputs, outputs)
    return model

model = MaskModel()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
print(model.summary())

train_gen = DataGen(train_ids, train_path, image_size=image_size, batch_size=batch_size)
valid_gen = DataGen(valid_ids, train_path, image_size=image_size, batch_size=batch_size)


train_gen = DataGen(train_ids, train_path, image_size=image_size, batch_size=batch_size)
valid_gen = DataGen(valid_ids, train_path, image_size=image_size, batch_size=batch_size)

train_steps = len(train_ids)//batch_size
valid_steps = len(valid_ids)//batch_size

model.fit_generator(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps, epochs=epochs)

model.save("nuclie_mask_model.h5")