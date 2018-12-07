import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import math

# this function adds noise per channel with gaussian distribution
def add_gaussian_noise(img,mean,variance,scale):
	# get dimensions and set the noise size
	height, width, channel  = img.shape[0],img.shape[1],img.shape[2]
	size = width * height * channel

	# create a gaussian noise per channel
	noise = np.random.normal(mean, variance, size)

	# resize the noise to add up into source img
	noise = scale*noise.reshape(height,width,channel)

	# add noise into source image
	noisy_img = img+noise
	return noisy_img

# add salt&pepper with specified ratio and amount
def add_salt_and_pepper_noise(img,sp_ratio,amount):
	# get dimensions and set the noise size
	height, width, channel  = img.shape[0],img.shape[1],img.shape[2]
	size = width * height * channel

	# create a copy of an image to be returned
	copy_img = np.copy(img)

	# create salt noise and add it one channel
	salt = int(np.round(amount * size * sp_ratio))
	noise = []
	for i in copy_img.shape:
		noise.append(np.random.randint(0, i , salt))
	copy_img[noise[0],noise[1],:] = 255

	#create pepper noise and add it per channel
	pepper = int(np.round(amount * size * (1- sp_ratio)))
	noise = []
	for i in img.shape:
		noise.append(np.random.randint(0, i, pepper))
	copy_img[noise[0],noise[1],:] = 0
	return copy_img

# perform horizontal and vertical flip
def flip(img):
	# directly use numpy left right and up down flip function
	return [img.copy(),np.fliplr(img.copy()),np.flipud(img.copy())]

# rotate image by 90 degree via tensorflow api
def rotate(img,number):
	# get dimensions and set the noise size
	height, width, channel = img.shape[0], img.shape[1], img.shape[2]

	# define the inputs through tf placeholder
	tf_img = tf.placeholder(tf.float32, shape=(height, width, channel))
	amount = tf.placeholder(tf.int32)

	# define the tensor graph for rotating image
	return_img = tf.image.rot90(tf_img, amount)

	# run the graph in a session
	with tf.Session() as sess:
		res = sess.run(return_img, feed_dict={tf_img: img, amount:number})
	return res

def adjust_brightness(img,number):
	# get dimensions and set the noise size
	height, width, channel = img.shape[0], img.shape[1], img.shape[2]

	# define the inputs through tf placeholder
	tf_img = tf.placeholder(tf.float32, shape=(height, width, channel))
	delta = tf.placeholder(tf.float32)

	# tensorflow needs image in 0,1 scale so convert it
	copy_img = img.copy()
	copy_img/= 255

	# define the tensor graph for rotating image
	return_img = tf.image.adjust_brightness(tf_img, delta)

	# run the graph in a session
	with tf.Session() as sess:
		res = sess.run(return_img,feed_dict={tf_img: copy_img,delta:number})

	# rescale to 0.255
	res *= 255 / res.max()

	return res

def adjust_contrast(img,number):
	# get dimensions and set the noise size
	height, width, channel = img.shape[0], img.shape[1], img.shape[2]

	# define the inputs through tf placeholder
	tf_img = tf.placeholder(tf.float32, shape=(height, width, channel))
	contrast_factor = tf.placeholder(tf.float32)

	# tensorflow needs image in 0,1 scale so convert it
	copy_img = img.copy()
	copy_img/= 255

	# define the tensor graph for rotating image
	return_img = tf.image.adjust_contrast(tf_img, contrast_factor)

	# run the graph in a session
	with tf.Session() as sess:
		res = sess.run(return_img,feed_dict={tf_img: copy_img,contrast_factor:number})

	# rescale to 0.255
	res *= 255 / res.max()

	return res

def affine_transform(img,degree):
	# get dimensions and set the noise size
	height, width, channel = img.shape[0], img.shape[1], img.shape[2]

	# define the inputs through tf placeholder
	tf_img = tf.placeholder(tf.float32, shape=(height, width, channel))
	angle_rad = degree*math.pi/180.0
	angles = tf.random_uniform([1], -angle_rad, angle_rad)
	#transforms = []

	# define the tensor graph for rotating image
	projected = tf.contrib.image.angles_to_projective_transforms(angles=angles,image_height=height ,image_width = width)
	#transforms.append(projected)
	return_img = tf.contrib.image.transform(img,tf.contrib.image.compose_transforms(projected),interpolation='BILINEAR')

	# run the graph in a session
	with tf.Session() as sess:
		res = sess.run(return_img, feed_dict={tf_img: img})
	return res

# tensorflow implementation
def translation(img,dx,dy):
	# get dimensions and set the noise size
	height, width, channel = img.shape[0], img.shape[1], img.shape[2]

	# define the inputs through tf placeholder
	tf_img = tf.placeholder(tf.float32, shape=(height, width, channel))
	transforms = [1, 0, dx, 0, 1, -50, 0, 0]
	return_img = tf.contrib.image.transform(img, transforms, interpolation='BILINEAR')
	# define the tensor graph for rotating image

	# run the graph in a session
	with tf.Session() as sess:
		res = sess.run(return_img, feed_dict={tf_img: img})
	return res

# one line blurring using scipy library via 2D Gaussian filter.
def gaussian_blur(img,number):
	from scipy import ndimage
	return ndimage.gaussian_filter(img.copy(), sigma=number)

def plot(images):
	fig = plt.figure()
	columns = len(images)
	for i in range(1, columns+1):
		img = images[i-1].astype(np.uint8)
		fig.add_subplot(1, columns, i)
		plt.imshow(img)
		plt.axis('off')
	plt.show(fig)

if __name__ == "__main__":
	img = plt.imread('dog.jpg').astype(np.float32)

	# generate gaussian noisy images with different scales
	# outputs = []
	# for i in range(1,5):
	# 	outputs.append(add_gaussian_noise(img,0,5,i))
	# plot(outputs)

	# outputs = []
	# for i in range(1,5):
	# 	outputs.append(add_salt_and_pepper_noise(img,0.5,i/100))
	# plot(outputs)

	# outputs = flip(img)
	# plot(outputs)

	# outputs = []
	# for i in range(1,5):
	# 	outputs.append(rotate(img,i))
	# plot(outputs)

	# outputs = []
	# for i in range(1,5):
	# 	outputs.append(adjust_brightness(img,i/10))
	# plot(outputs)

	# outputs = []
	# for i in reversed(range(1,5)):
	# 	outputs.append(adjust_contrast(img,i/5))
	# plot(outputs)

	# outputs = []
	# for i in (range(1,5)):
	# 	outputs.append(affine_transform(img,i*30))
	# plot(outputs)

	# outputs = []
	# for i in (range(1,5)):
	# 	outputs.append(translation(img,i*10,i*-10))
	# plot(outputs)

	# outputs = []
	# for i in (range(1,5)):
	# 	outputs.append(gaussian_blur(img,i/2))
	# plot(outputs)