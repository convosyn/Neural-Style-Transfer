from __future__ import print_function

import sys

import time
from PIL import Image
import numpy as np

from keras import backend
from keras.models import Model
from keras.applications.vgg16 import VGG16

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave

class NeuralStyle:

	def __init__(self, parameters):

		self.loss_value = None
		self.grads_values = None

		self.number_of_iterations = int(parameters['iterations'])
		self.width = int(parameters['width'])
		self.height = int(parameters['height'])
		self.red_sub = parameters['red_subtract']
		self.green_sub = parameters['green_subtract']
		self.blue_sub = parameters['blue_subtract']
		self.content_weight = parameters['content_weight']
		self.style_weight = parameters['style_weight']
		self.total_variation_weight = parameters['total_variation_weight']
		self.channel_count = 3
		self.blend_content_ratio = parameters['blend_content_ratio']

	def loadImages(self, content_path, style_path):
		self.content_path = content_path
		self.style_path = style_path

		content_image = Image.open(self.content_path)
		content_image = content_image.resize((self.width, self.height))
		content_image.show()

		style_image = Image.open(self.style_path)
		style_image = style_image.resize((self.width, self.height))
		style_image.show()

		self.content_image = content_image
		self.style_image = style_image

	def preprocess(self):
		self.content_array = np.asarray(self.content_image, dtype='float32')
		self.content_array = np.expand_dims(self.content_array, axis = 0)
		print("dimensions for content_array: ", self.content_array.shape)

		self.style_array = np.asarray(self.style_image, dtype='float32')
		self.style_array = np.expand_dims(self.style_array, axis = 0)
		print("dimensions for style_array: ", self.style_array.shape)

		self.content_array[:, :, :, 0] -= self.red_sub
		self.content_array[:, :, :, 1] -= self.green_sub
		self.content_array[:, :, :, 2] -= self.blue_sub
		self.content_array = self.content_array[:, :, :, ::-1]

		self.style_array[:, :, :, 0] -= self.red_sub
		self.style_array[:, :, :, 1] -= self.green_sub
		self.style_array[:, :, :, 2] -= self.blue_sub
		self.style_array = self.style_array[:, :, :, ::-1]

	def content_loss(self, content, combination):
		return backend.sum(backend.square(combination - content))

	def style_loss(self, style, combination):
		C = self.gram_matrix(style)
		S = self.gram_matrix(combination)
		size = self.height * self.width
		return backend.sum(backend.square(S - C)) / (4.0  * (self.channel_count ** 2) * (size ** 2))

	def gram_matrix(self, x):
		features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
		gram = backend.dot(features, backend.transpose(features))
		return gram

	def total_variation_loss(self, x):
		a = backend.square(x[:, :self.height - 1, :self.width - 1, :] - x[:, 1:, :self.width - 1, :])
		b = backend.square(x[:, :self.height - 1, :self.width - 1, :] - x[:, :self.height - 1, 1:, :])
		return backend.sum(backend.pow(a + b, 1.25))

	def run(self):
		content_image = backend.variable(self.content_array)
		style_image = backend.variable(self.style_array)
		combination_image = backend.placeholder((1, self.height, self.width, self.channel_count))
		
		input_tensor = backend.concatenate([content_image, style_image, combination_image], axis = 0)
		model = VGG16(input_tensor = input_tensor, weights = 'imagenet', include_top = False)
		layers = dict([layer.name, layer.output] for layer in model.layers)
		print("Layers: \n{!s}".format(layer for layer in layers))

		loss = backend.variable(0.)

		layer_features = layers['block2_conv2']
		content_image_features = layer_features[0, :, :, :]
		combination_features = layer_features[2, :, :, :]

		loss += self.content_weight * self.content_loss(content_image_features, combination_features)

		feature_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']
		#feature_layers = ['block3_conv1', 'block3_conv3', 'block4_conv3', 'block5_conv1', 'block5_conv3']

		for layer_name in feature_layers:
			layer_features = layers[layer_name]
			style_features = layer_features[1, :, :, :]
			combination_features = layer_features[2, :, :, :]
			sl = self.style_loss(style_features, combination_features)
			loss += (self.style_weight / len(feature_layers)) * sl

		loss += self.total_variation_weight * self.total_variation_loss(combination_image)
		grads = backend.gradients(loss, combination_image)

		outputs = [loss]
		outputs += grads

		self.f_outputs = backend.function([combination_image], outputs)

		content_array = np.asarray(self.content_image, dtype='float32')
		style_array = np.asarray(self.style_image, dtype='float32')
		#x = np.clip(content_array * self.blend_content_ratio + style_array * (1 - self.blend_content_ratio), 0, 255).reshape((1, self.height, self.width, 3)) - 128
		x = np.random.uniform(0, 255, (1, self.height, self.width, 3)) - 128
		#x = style_array.reshape((1, self.height, self.width, 3)) - 128


		for i in range(self.number_of_iterations):
			if i%5 == 0:
				self.showResult(x)
			print("Start of iteration: ", i)
			start_time = time.time()
			x, min_val, info = fmin_l_bfgs_b(self.loss, x.flatten(), fprime = self.grads, maxfun=4 * self.number_of_iterations)
			print("Current loss Value: ", min_val)
			end_time = time.time()
			print("Iteration {!s} completed in {:.2f}s ".format(i, end_time - start_time))

		self.result = x

	def eval_loss_and_grad(self, x):
		x = x.reshape((1, self.height, self.width, 3))
		outs = self.f_outputs([x])
		loss_value = outs[0]
		grad_values = outs[1].flatten().astype('float64')
		return loss_value, grad_values

	def loss(self, x):
		assert self.loss_value is None
		loss_value, grad_values = self.eval_loss_and_grad(x)
		self.loss_value = loss_value
		self.grad_values = grad_values
		return self.loss_value

	def grads(self, x):
		assert self.loss_value is not None
		grad_values = np.copy(self.grad_values)
		self.loss_value = None
		self.grad_values = None
		return grad_values


	def showResult(self, img_p = None, save_img = False):

		if img_p is None:
			img_p = np.copy(self.result)

		img = np.copy(img_p)
		img = img.reshape((self.height, self.width, 3))
		img = img[:, :, ::-1]
		img[:, :, 0] += self.red_sub
		img[:, :, 1] += self.green_sub
		img[:, :, 2] += self.blue_sub

		img = np.clip(img, 0, 255).astype('uint8')
		img = Image.fromarray(img)
		img.show()

		if save_img == True:
			img.save('output.jpg')



if __name__ == "__main__":

	parameters = {
				"width": 512.0,
				"height": 512.0,
				"red_subtract": 103.939, 
				"green_subtract": 116.779,
				"blue_subtract": 123.68,
				"content_weight": 0.075,
				"style_weight": 5.0,
				"total_variation_weight": 1.0,
				"iterations": 10.0, 					
				"blend_content_ratio": 0.4
			}

	params = ["--width",
				"--height",
				"--red_sub", 
				"--green_sub",
				"--blue_sub",
				"--content_weight",
				"--style_weight",
				"--total_variation_weight",
				"--iterations",
				"--blend_content_ratio"]

	paramsDict = {"--width": "width" ,
				"--height": "height",
				"--red_sub": "red_subtract", 
				"--green_sub": "green_subtract",
				"--blue_sub": "blue_subtract",
				"--content_weight": "content_weight",
				"--style_weight": "style_weight",
				"--total_variation_weight": "total_variation_weight",
				"--iterations": "iterations",
				"--blend_content_ratio": "blend_content_ratio"}

	print(sys.argv[0])

	if len(sys.argv) < 3:
		print('Usage: python NeuralStyle <content-image-path> <style-image-path>')
	
	print("content_image: {!s} \n style_image: {!s}".format(sys.argv[1], sys.argv[2]))

	if len(sys.argv) > 3:
		for i in range(3, len(sys.argv), 2):
			print("argument: {!s} -> Value: {!s}".format(sys.argv[i], sys.argv[i+1]))
			if sys.argv[i] not in params:
				raise IndexError("Value not in parameters.")
			else:
				parameters[paramsDict[sys.argv[i]]] = float(sys.argv[i+1])

	exp = NeuralStyle(parameters)
	exp.loadImages(sys.argv[1], sys.argv[2])
	exp.preprocess()
	exp.run()
	exp.showResult(save_img = True)
