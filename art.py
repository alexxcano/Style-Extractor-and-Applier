from __future__ import print_function, division, absolute_import
import numpy as np
import pickle
from PIL import Image
import scipy
import argparse
import os
import theano
import theano.tensor as T
import lasagne
from lasagne.utils import floatX
from lasagne import layers
from lasagne.layers.dnn import Conv2DDNNLayer


#/---------------------------------------------------COMMAND LINE PARSER------------------------------------------------------------------------\
parser = argparse.ArgumentParser(description='Neural Style')
parser.add_argument('art', help='Image used for style', type=str)
parser.add_argument('photo', help='Image used for content', type=str)
parser.add_argument('-o', '--output', help='Output folder name or file', type=str, default='output',
					action='store', dest='output')
parser.add_argument('-i', '--iterations', help='Number of iterations optimizer iterates', type=int,
					default=8, action='store', dest='iterations')
parser.add_argument('-d', '--dimension', help='Dimension of output image', type=int, default=600,
					action='store', dest='dim')
parser.add_argument('-p', '--preserve', help='Preserve original color of content image', default=False,
				    action='store_true', dest='preserve_color')
parser.add_argument('-f', '--flip', help='Helps a little when builing the CNN for faces by flipping convolutional filters', 
					default=False, action='store_true', dest='filters')
parser.add_argument('-s', '--style-intensity', help='Intensity the art image will have on content image, changes content layer', 
					type=int, choices=(1, 2, 3, 4, 5), default=4, action='store', dest='intensity')
parser.add_argument('-wc', '--weight-content', help='Weight of the content loss function', type=float, default=0.001,
					action='store', dest='weight_c')
parser.add_argument('-ws', '--weight-style', help='Weight of the style loss functions', type=float, default=0.2e6,
					action='store', dest='weight_s')
parser.add_argument('--verbose', help='Increase output verbosity', action='store_true', default=False)
argv = parser.parse_args()

if argv.verbose:
	print(
	'''
	Style Image Name:     {}
	Content Image Name:   {}
	Output Name:          {}
	Number of Iterations: {}
	Image Dimension:      {}
	Flip Conv Filters:    {}
	Intensity of Style:   {}
	Content Loss Weight:  {}
	Style Loss Weight:    {}
	'''.format(argv.art, argv.photo, argv.output, argv.iterations, argv.dim, argv.filters, argv.intensity, argv.weight_c, argv.weight_s)
	)	
	
#/------------------------------------------------------------END---------------------------------------------------------------------------------\	


#/---------------------------------------------------CONVOLUTIONAL NEURAL NETWORK------------------------------------------------------------------\
#Length and with of Input and Output image
IMAGE_DIM = argv.dim

def build_cnn():
	"""
	VGG-19 CNN Network 
	Paper Name: Very Deep Convolutional Networks for Large-Scale Image Recognition
	"""
	flip = argv.filters
	net = {}
	net['input'] = layers.InputLayer((1, 3, IMAGE_DIM, IMAGE_DIM))
	net['conv1_1'] = Conv2DDNNLayer(net['input'], 64, 3, pad=1, flip_filters=flip)
	net['conv1_2'] = Conv2DDNNLayer(net['conv1_1'], 64, 3, pad=1, flip_filters=flip)
	net['pool1'] = layers.Pool2DLayer(net['conv1_2'], 2, mode='average_exc_pad')
	net['conv2_1'] = Conv2DDNNLayer(net['pool1'], 128, 3, pad=1, flip_filters=flip)
	net['conv2_2'] = Conv2DDNNLayer(net['conv2_1'], 128, 3, pad=1, flip_filters=flip)
	net['pool2'] = layers.Pool2DLayer(net['conv2_2'], 2, mode='average_exc_pad')
	net['conv3_1'] = Conv2DDNNLayer(net['pool2'], 256, 3, pad=1, flip_filters=flip)
	net['conv3_2'] = Conv2DDNNLayer(net['conv3_1'], 256, 3, pad=1, flip_filters=flip)
	net['conv3_3'] = Conv2DDNNLayer(net['conv3_2'], 256, 3, pad=1, flip_filters=flip)
	net['conv3_4'] = Conv2DDNNLayer(net['conv3_3'], 256, 3, pad=1, flip_filters=flip)
	net['pool3'] = layers.Pool2DLayer(net['conv3_4'], 2, mode='average_exc_pad')
	net['conv4_1'] = Conv2DDNNLayer(net['pool3'], 512, 3, pad=1, flip_filters=flip)
	net['conv4_2'] = Conv2DDNNLayer(net['conv4_1'], 512, 3, pad=1, flip_filters=flip)
	net['conv4_3'] = Conv2DDNNLayer(net['conv4_2'], 512, 3, pad=1, flip_filters=flip)
	net['conv4_4'] = Conv2DDNNLayer(net['conv4_3'], 512, 3, pad=1, flip_filters=flip)
	net['pool4'] = layers.Pool2DLayer(net['conv4_4'], 2, mode='average_exc_pad')
	net['conv5_1'] = Conv2DDNNLayer(net['pool4'], 512, 3, pad=1, flip_filters=flip)
	net['conv5_2'] = Conv2DDNNLayer(net['conv5_1'], 512, 3, pad=1, flip_filters=flip)
	net['conv5_3'] = Conv2DDNNLayer(net['conv5_2'], 512, 3, pad=1, flip_filters=flip)
	net['conv5_4'] = Conv2DDNNLayer(net['conv5_3'], 512, 3, pad=1, flip_filters=flip)
	net['pool5'] = layers.Pool2DLayer(net['conv5_4'], 2, mode='average_exc_pad')

	return net

#/------------------------------------------------------------END---------------------------------------------------------------------------------\


#/-------------------------------------------------------IMAGE PROCCESSING--------------------------------------------------------------------------\	
#Mean pixel values of BGR
MEAN_VALUES = np.array([104, 117, 123]).reshape((3,1,1))
	
def prep_image(image1, image2):
	'''
	Scales both images down and center crops them
	Changes Color Space from RGB to BGR
	Normalizes both images using mean BGR pixel values (104, 117, 123)
	'''
	art = Image.open(image1)
	photo = Image.open(image2)
		
	if art.height < art.width:
		im1 = art.resize((IMAGE_DIM, art.width*IMAGE_DIM//art.height))
	else:
		im1 = art.resize((art.width*IMAGE_DIM//art.height, IMAGE_DIM))
	
	if photo.height < photo.width:
		im2 = photo.resize((IMAGE_DIM, photo.width*IMAGE_DIM//photo.height))
	else:
		im2 = photo.resize((photo.width*IMAGE_DIM//photo.height, IMAGE_DIM))
	
	
	im1 = im1.crop((im1.width//2-IMAGE_DIM//2, im1.height//2-IMAGE_DIM//2, im1.width//2+IMAGE_DIM//2, im1.height//2+IMAGE_DIM//2))
	im2 = im2.crop((im2.width//2-IMAGE_DIM//2, im2.height//2-IMAGE_DIM//2, im2.width//2+IMAGE_DIM//2, im2.height//2+IMAGE_DIM//2))
	
	im1 = np.swapaxes(np.swapaxes(np.array(im1), 1, 2), 0, 1)
	im2 = np.swapaxes(np.swapaxes(np.array(im2), 1, 2), 0, 1)
	
	im1 = im1[::-1, :, :]
	im2 = im2[::-1, :, :]
	
	im1 = im1 - MEAN_VALUES
	im2 = im2 - MEAN_VALUES
	
	if argv.verbose:
		Image.fromarray(np.clip(np.swapaxes(np.swapaxes(im1, 0, 1), 1, 2), 0, 255).astype('uint8')).show()
		Image.fromarray(np.clip(np.swapaxes(np.swapaxes(im2, 0, 1), 1, 2), 0, 255).astype('uint8')).show()
	
	return floatX(im1[np.newaxis]), floatX(im2[np.newaxis])

def deprocess(im):
	'''
	Undos the Normalization and reverts back to RGB color space
	'''
	im = np.copy(im[0])
	im += MEAN_VALUES

	im = im[::-1]
	im = np.swapaxes(np.swapaxes(im, 0, 1), 1, 2)
    
	im = np.clip(im, 0, 255).astype('uint8')
	return im

	
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def gray2rgb(gray):
	w, h = gray.shape
	rgb = np.empty((w, h, 3), dtype=np.float32)
	rgb[:, :, 2] = rgb[:, :, 1] = rgb[:, :, 0] = gray
	return rgb

def preserve_color(og_im, output_im):
	'''
	Preserves color of output images using YUV color space
	Luminosity transfer steps:
    1. Convert RGB->grayscale accoriding to Rec.601 luma (0.299, 0.587, 0.114)
    2. Convert grayscale into YUV (YCbCr)
    3. Convert second image into YUV (YCbCr)
    4. Recombine (first image YUV.Y, second image YUV.U, first image YUV.V)
	5. Convert recombined image from YUV back to RGB
	'''
	original = np.clip(og_im, 0, 255)
	styled = np.clip(output_im, 0, 255)
	
	styled_gray = rgb2gray(styled)
	styled_gray_rgb = gray2rgb(styled_gray)
	
	styled_gray_yuv = np.array(Image.fromarray(styled_gray_rgb.astype(np.uint8)).convert('YCbCr'))
	
	original_yuv = np.array(Image.fromarray(original.astype(np.uint8)).convert('YCbCr'))
	
	if argv.verbose:
		Image.fromarray(original_yuv).show()
		Image.fromarray(styled_gray_yuv).show()
	
	w, h, _ = original.shape
	combined_yuv = np.empty((w, h, 3), dtype=np.uint8)
	combined_yuv[..., 0] = styled_gray_yuv[..., 0]
	combined_yuv[..., 1] = original_yuv[..., 1]
	combined_yuv[..., 2] = original_yuv[..., 2]

	img_out = np.array(Image.fromarray(combined_yuv, 'YCbCr').convert('RGB'))
	
	return img_out

#/------------------------------------------------------------END---------------------------------------------------------------------------------\


#/--------------------------------------------------------LOSS FUNCTIONS---------------------------------------------------------------------------\	
def gram_matrix(mat):
	mat = mat.flatten(ndim=3)
	g = T.tensordot(mat, mat, axes=([2], [2]))
	return g
	
def content_loss(P, X, layer):
	p = P[layer]
	x = X[layer]

	loss = 1./2 * ((x - p)**2).sum()
	return loss

def style_loss(A, X, layer):
	a = A[layer]
	x = X[layer]
    
	A = gram_matrix(a)
	G = gram_matrix(x)
    
	N = a.shape[1]
	M = a.shape[2] * a.shape[3]
    
	loss = 1./(4 * N**2 * M**2) * ((G - A)**2).sum()
	return loss
	
def total_variation_loss(x):
	return (((x[:,:,:-1,:-1] - x[:,:,1:,:-1])**2 + (x[:,:,:-1,:-1] - x[:,:,:-1,1:])**2)**1.25).sum()

def eval_loss(x):
	x = floatX(x.reshape((1, 3, IMAGE_DIM, IMAGE_DIM)))
	gen_image.set_value(x)
	return f_loss().astype('float64')

def eval_grad(x):
	x = floatX(x.reshape((1, 3, IMAGE_DIM, IMAGE_DIM)))
	gen_image.set_value(x)
	return np.array(f_grad()).flatten().astype('float64')

#/------------------------------------------------------------END---------------------------------------------------------------------------------\

	
#/------------------------------------------------------------MAIN---------------------------------------------------------------------------------\	
output_path = argv.output
placeholder = np.random.uniform(-128, 128, (1, 3, IMAGE_DIM, IMAGE_DIM))
switch_statement = {
	1 : 'conv1_2',
	2 : 'conv2_2',
	3 : 'conv3_2',
	4 : 'conv4_2',
	5 : 'conv5_2'  }
	
content_layer = switch_statement.get(argv.intensity)

cnn = build_cnn()
vgg19 = pickle.load(open('vgg19_normalized.pkl', 'rb'))['param values']
lasagne.layers.set_all_param_values(cnn['pool5'], vgg19)

art_im, photo_im = prep_image(argv.art, argv.photo)

layers = {i: cnn[i] for i in [content_layer, 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']}
	
input = T.tensor4()
outputs = lasagne.layers.get_output(layers.values(), input)

art_features = {i: theano.shared(output.eval({input: art_im}))
				for i, output in zip(layers.keys(), outputs)}
photo_features = {i: theano.shared(output.eval({input: photo_im}))
				for i, output in zip(layers.keys(), outputs)}
	
gen_image = theano.shared(floatX(np.copy(placeholder)))
gen_features = lasagne.layers.get_output(layers.values(), gen_image)
gen_features = {k: v for k, v in zip(layers.keys(), gen_features)}
	
losses = []
	
losses.append(argv.weight_c * content_loss(photo_features, gen_features, content_layer))
	
losses.append(argv.weight_s * style_loss(art_features, gen_features, 'conv1_1'))
losses.append(argv.weight_s * style_loss(art_features, gen_features, 'conv2_1'))
losses.append(argv.weight_s * style_loss(art_features, gen_features, 'conv3_1'))	
losses.append(argv.weight_s * style_loss(art_features, gen_features, 'conv4_1'))
losses.append(argv.weight_s * style_loss(art_features, gen_features, 'conv5_1'))
	
losses.append(0.1e-7 * total_variation_loss(gen_image))

total_loss = sum(losses)

grad = T.grad(total_loss, gen_image)
	
f_loss = theano.function([], total_loss)
f_grad = theano.function([], grad)
	
gen_image.set_value(floatX(np.copy(placeholder)))

new_image = gen_image.get_value().astype('float64')
images = []
images.append(new_image)

for i in range(argv.iterations):
	print('Iteration #', i, sep='')
	scipy.optimize.fmin_l_bfgs_b(eval_loss, new_image.flatten(), fprime=eval_grad, maxfun=40)
	new_image = gen_image.get_value().astype('float64')
	images.append(new_image)
	
if output_path.find('.') == -1 and not os.path.exists(output_path):
	os.mkdir(output_path)

if output_path.find('.') == -1:
	for i in range(1, len(images)):
		Image.fromarray(deprocess(images[i])).save('{}//output{}.jpg'.format(output_path, i))
		if argv.preserve_color:
			Image.fromarray(preserve_color(deprocess(photo_im), deprocess(images[i]))).save('{}//preserved{}.jpg'.format(output_path, i))
else:
	Image.fromarray(deprocess(images[len(images)-1])).save(output_path)
	if argv.preserve_color:
			Image.fromarray(preserve_color(deprocess(photo_im), deprocess(images[len(images)-1]))).save('preserved.jpg')
