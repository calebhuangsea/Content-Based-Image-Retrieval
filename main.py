"""

Content-Based Image Retrieval

To run the program with default arguments and say beach_1.jpg as the query image, type the following in the terminal
$ python main.py -q beach_1

The other available arguments are:
-f       : to specify the type of feature - only color histogram (color) or only lbp histogram (lbp) or both (both)
-color   : to specify the method for color histogram - grayscale with 8 bins (gray_8), grayscale with 256 bins (gray_256) and RGB histogram (rgb)
-lbp     : to specify the method for LBP histogram - using whole image (whole_image) or by dividing the image into grids (grid_image)
-dist    : to specify the distance measure used to compare the image feature vectors

A complete example to run the program is thus:
$ python main.py -q beach_1 -f both -color rgb -lbp whole_image -dist euclidean

The starter code is compatible with both python 2 and 3

"""

### Load libraries

import numpy as np
import imageio
import glob, argparse, sys
sys.path.append('images')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


### Function to extract color histogram for each image
def color_histogram(image,method):
	# convert RGB image to grayscale
	gray_scale= 0.299*image[:,:,0] + 0.587*image[:,:,1] + 0.114*image[:,:,2]
	if method == 'gray_8':
		# compute the color histogram for the grayscale image with 8 bins
		hist_vector = np.zeros((8,))
		for i in range(0, gray_scale.shape[0]):
			for j in range(0, gray_scale.shape[1]):
				# do 8 bins by interval 32
				hist_vector[int(gray_scale[i][j]/32)] += 1

	elif method == 'gray_256':
		# compute the color histogram for the grayscale image with 256 bins
		hist_vector = np.zeros((256,))
		for i in range(0, gray_scale.shape[0]):
			for j in range(0, gray_scale.shape[1]):
				hist_vector[int(gray_scale[i][j])] += 1

	elif method == 'rgb':
		num_bins = 256
		# compute the color histogram for the RGB image with 256 bins per channel
		hist_vector = np.zeros((num_bins*3,))
		for i in range(0, image.shape[0]):
			for j in range(0, image.shape[1]):
				# do 256 bins for all 3 channels
				hist_vector[int(image[i][j][0])] += 1
				hist_vector[int(image[i][j][1]) + num_bins] += 1
				hist_vector[int(image[i][j][2]) + 2 * num_bins] += 1

	else:
		print('Error: incorrect color histogram method')
		return []

	# normalize the histogram (REMOVE THE EPSILON WHEN IMPLEMENTED)

	hist_vector /= sum(hist_vector)
	return list(hist_vector)


### Function to extract LBP histogram for each image
def lbp_histogram(image,method):
	# convert RGB image to grayscale
	gray_image = 0.299*image[:,:,0] + 0.587*image[:,:,1] + 0.114*image[:,:,2]
	# Now to access pixel (h,w) of gray image, use gray_image[h,w]
	# compute the LBP histogram from the entire image, ignore the boundary pixels
	if method == 'whole_image':
		hist_vector = np.zeros((256,))
		for i in range(1, gray_image.shape[0] - 1):
			for j in range(1, gray_image.shape[1] - 1):
				hist_vector[local_binary_pattern(gray_image, i, j)] += 1


	elif method == 'grid_image':
		# compute the LBP histogram for each 16x16 grid
		num_grids = 12
		hist_vector = np.zeros((32*num_grids,))
		hist_vector = grid_hist(hist_vector, 1, 15, 1, 15, gray_image, 0)
		hist_vector = grid_hist(hist_vector, 1, 15, 16, 31, gray_image, 1)
		hist_vector = grid_hist(hist_vector, 1, 15, 32, 47, gray_image, 2)
		hist_vector = grid_hist(hist_vector, 1, 15, 48, 62, gray_image, 3)
		
		hist_vector = grid_hist(hist_vector, 16, 31, 1, 15, gray_image, 4)
		hist_vector = grid_hist(hist_vector, 16, 31, 16, 31, gray_image, 5)
		hist_vector = grid_hist(hist_vector, 16, 31, 32, 47, gray_image, 6)
		hist_vector = grid_hist(hist_vector, 16, 31, 48, 62, gray_image, 7)

		hist_vector = grid_hist(hist_vector, 32, 46, 1, 15, gray_image, 8)
		hist_vector = grid_hist(hist_vector, 32, 46, 16, 31, gray_image, 9)
		hist_vector = grid_hist(hist_vector, 32, 46, 32, 47, gray_image, 10)
		hist_vector = grid_hist(hist_vector, 32, 46, 48, 62, gray_image, 11)
		
	else:
		print('Error: incorrect lbp histogram method')
		return []

	# normalize the histogram (REMOVE THE EPSILON WHEN IMPLEMENTED)
	hist_vector /= sum(hist_vector)
	return list(hist_vector)

### Function to extract LBP histogram for each image within given grid
def grid_hist(hist_vector, x1, x2, y1, y2, image, order):
	for i in range(x1, x2):
		for j in range(y1, y2):
			hist_vector[int(local_binary_pattern(image, i, j) / 8) + order * 32] += 1
	return hist_vector

### Function to compute LBP for each pixel
def local_binary_pattern(image, i, j):
	sum = 0
	if image[i-1][j-1] > image[i][j]:
		sum += 128
	if image[i-1][j] > image[i][j]:
		sum += 64
	if image[i-1][j+1] > image[i][j]:
		sum += 32
	if image[i][j+1] > image[i][j]:
		sum += 16
	if image[i+1][j+1] > image[i][j]:
		sum += 8
	if image[i+1][j] > image[i][j]:
		sum += 4
	if image[i+1][j-1] > image[i][j]:
		sum += 2
	if image[i][j-1] > image[i][j]:
		sum += 1
	return sum

### Function to compute the feature vector for a given image

def calculate_feature(image, featuretype, color_hist_method, lbp_hist_method):
	# create and return the feature vector as a list
	feature_vector = []
	
	if featuretype == 'color':
		feature_vector += color_histogram(image, method=color_hist_method)
	elif featuretype == 'lbp':
		feature_vector += lbp_histogram(image, method=lbp_hist_method)
	elif featuretype == 'both':
		feature_vector += color_histogram(image, method=color_hist_method)
		feature_vector += lbp_histogram(image, method=lbp_hist_method)
	else:
		print('Error: incorrect feature type')

	return feature_vector




########## MAIN PROGRAM ##########

if __name__ == "__main__":

	### Provide the name of the query image, for example: beach_1

	ap = argparse.ArgumentParser()
	ap.add_argument("-q", "--query", type=str, required = True, help = "name of the query image")
	ap.add_argument("-f", "--feature", type=str, default = 'both', help = "image feature(s) to be extracted")
	ap.add_argument("-color", "--color_hist_method", type=str, default = 'rgb', help = "method for color histogram")
	ap.add_argument("-lbp", "--lbp_hist_method", type=str, default = 'whole_image', help = "method for lbp histogram")
	ap.add_argument("-dist", "--distance_measure", type=str, default = 'euclidean', help = "distance measure for image comparison")
	args = ap.parse_args()

	### Get all the image names in the database

	image_names = sorted(glob.glob('images/*.jpg'))
	num_images = len(image_names)

	### Create an empty list to hold the feature vectors
	### We shall append each feature vector to this vector
	### Later it will be converted to an array of shape (#images, feature_dimension)

	features = []

	### Loop over each image and extract a feature vector

	for name in image_names:
		print('Extracting feature for ',name.split('/')[-1])
		image = imageio.imread(name)
		feature = calculate_feature(image, args.feature, args.color_hist_method, args.lbp_hist_method)
		features.append(feature)

	### Read the query image and extract its feature vector

	query_image = imageio.imread('images/'+args.query+'.jpg')
	query_feature = calculate_feature(query_image, args.feature, args.color_hist_method, args.lbp_hist_method)

	### Compare the query feature with the database features

	query_feature = np.reshape(np.array(query_feature),(1,len(query_feature)))
	features = np.array(features)
	print('Calculating distances...')
	distances = cdist(query_feature, features, args.distance_measure)

	### Sort the distance values in ascending order
	
	distances = list(distances[0,:])
	sorted_distances = sorted(distances)
	sorted_imagenames = []	

	### Perform retrieval; plot the images and save the result as an image file in the working folder
	fig = plt.figure()

	for i in range(num_images):
		fig.add_subplot(5,8,i+1)
		image_name = image_names[distances.index(sorted_distances[i])]
		sorted_imagenames.append(image_name.split('/')[-1].rstrip('.jpg'))
		plt.imshow(imageio.imread(image_name))
		plt.axis('off')
		plt.title(str(i+1))

	figure_save_name = 'Q_'+args.query+'_F_'+args.feature+'_C_'+args.color_hist_method+'_L_'+args.lbp_hist_method+'_D_'+args.distance_measure+'.png'
	plt.savefig(figure_save_name, bbox_inches='tight')
	plt.close(fig)

	### Calculate and print precision value (in percentage)

	precision = 0
	query_class = args.query.split('_')[0]
	for i in range(5):
		retrieved_class = sorted_imagenames[i].split('_')[0]
		if retrieved_class == query_class:
			precision += 1
	print('Precision: ',int((precision/5)*100),'%')
