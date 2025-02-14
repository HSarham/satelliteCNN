
# Reference: https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/


import numpy
import pandas
from os import listdir
from os.path import isfile, join
import sys
from PIL import Image
import matplotlib.pyplot as plt

Image.MAX_IMAGE_PIXELS = 999999999 # 458875136 (45p) # IMPORTANT to set when handling big images with size over package default!!

### Notes:
# This package assumes usage of images downloaded from Google Earth Engine, can be used with other images as well, but might need modifications...
# It in general supports creation/usage of Meta-data and creation of subimages from larger images/files...
# IN PIL - image is in (width, height, channel)
# In Numpy - image is in (height, width, channel)


############## START library functions ##############


#########################################
"""
Description:
Plotting of a simple histogram

Input variables/types and functions:
values: 		as a list of values to be plotted in a histogram
selected_bins:	as an integer of how many bins should be used when splitting values
prefix: 		a string to be appended on the histogram title

Output (data and types):
The show of the histogram in an external window

Constraints:
N/A

Exceptions:
N/A

Others:
N/A

"""

def plot_histogram (values, selected_bins, prefix = '', save = True):
	
	plt.hist(values, bins = selected_bins)
	plt.title(prefix + ' values: ' + str(len(values)) + ' bins_'+ str(selected_bins))
	plt.grid(True)
	if save == True:
		plt.savefig(prefix + '_' + str(len(values)) + ".png")
	else:
		plt.show()
	
	plt.clf()

#########################################
"""
Description:
Calculation of the mean pixel-value from given image with possibility to control with a limit

Input variables/types and functions:
viirs_image: as a PIL image
lightlimit: as float, pixel values less than this will be calculated as zero values

Output (data and types):
A float value as the mean value of all pixel-values in given image

Constraints:
N/A

Exceptions:
N/A

Others:
N/A

"""

def viirs_light_calculation (viirs_image, lightlimit = 0.5):

	pil_array = numpy.array(viirs_image)
	pil_array[pil_array < lightlimit] = 0.0
	#print(pil_array)
	return pil_array.mean()

#########################################
"""
Description:
Moves pixel values to start from 0 and upwards if there are negative pixel values and returns the mean value of all pixels

Input variables/types and functions:
viirs_image: as a PIL image

Output (data and types):
The mean value o fall pixel values

Constraints:
N/A

Exceptions:
N/A

Others:

"""

def viirs_light_calculation_zerobased_mean (viirs_image):

	pil_array = numpy.array(viirs_image)
	pixel_min = numpy.amin(pil_array)
	if pixel_min < 0.0:
		pil_array = ((pil_array + (-1.0 * pixel_min)))
	
	print('Mean light value: ', pil_array.mean())
	return pil_array.mean()

#########################################
"""
Description:
Calculates the sum of all pixelvalues after setting values to zero if they are under given limit

Input variables/types and functions:
viirs_image:	as a PIL image
lightlimit:		as float

Output (data and types):
The sum value o fall pixel values

Constraints:
N/A

Exceptions:
N/A

Others:
N/A

"""

def viirs_light_calculation_zerobased_sum (viirs_image, lightlimit = 0.5):

	pil_array = numpy.array(viirs_image)
	pil_array[pil_array < lightlimit] = 0.0
	
	return pil_array.flatten().sum()





#########################################
"""
Description:
Given a PIL image (Viirs) light values, matrix codes and label-strings are calculated from sub-images created fron given image

Input variables/types and functions:
pil_base_image:	as a PIL image
pixel_width:	The width for which a subimage is defined
pixel_height:	The height for which a subimage is defined	
startrow:		Normally 1 (where to start calculating the matrix-code)
startcolumn:	Normally 1 (where to start calculating the matrix-code)
path:			Not implemented yet
filename:		Not implemented yet
resize:			A possibility to first make a resize of the given image before calculation (defined resize as a tuple)
lower_limit:	The upper limit for No_Light values
middle_limit:	The upper limit for Medium_Light values
areacode:		An areacode to be appended to for output
lightlimit:		The limit that is defined for a pixelvalue shall be considered to be defined having light
debuginfo:		Weather to have printouts or not

Output (data and types):
Four lists holding all-lightvalues, matrix_codes, labels and areacodes as strings

Constraints:
pixel_width must be less than pil_base_image width
pixel_height must be less than pil_base_image height


Exceptions:
N/A

Others:
If the functions are supposed to be used in an iterative way it should first be modified.
TBD: fix return of next row and column as in function: create_subimage_matrix_images

Reference: https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.new


"""

def create_subimage_matrix_label_data (pil_base_image, pixel_width, pixel_height, startrow, startcolumn, path = '', filename = '', resize = None, lower_limit = 5, middle_limit = 50, areacode = '', lightlimit = 0.5, debuginfo = False):
	
	if resize is not None:
		pil_base_image = pil_base_image.resize(resize, resample = Image.NEAREST)
	
	row_nr = startrow
	column_nr = startcolumn

	all_lightvalues_list = []
	labels_list = []
	matrix_code_list = []
	area_code_list = []
	image_width = pil_base_image.width
	image_height = pil_base_image.height
	image_mode = pil_base_image.mode # Like... 'F' or 'RGB'
	
	sub_image_nr = 0
	label = ''
	matrix_code = ''
	
	no_light_list = []
	medium_ligth_list = []
	bright_light_list = []
	
	light_mean_min = 1000000.0
	light_mean_max = -1000000.0
	
	pil_array = numpy.array(pil_base_image)
	print('Min: ', numpy.amin(pil_array), ' Max: ', numpy.amax(pil_array))
	
	if debuginfo:
		print('Base image')
		print('In_width: ' + str(image_width) + ' In_height: ' + str(image_height) + ' Mode: ' + str(image_mode))
		#print('left, top, right, bottom: ')
		print(pil_base_image.size, '\n')
	
	nr_sub_images_width = image_width // pixel_width
	nr_sub_images_height = image_height // pixel_height

	new_composed_image = Image.new(mode = image_mode, size = (nr_sub_images_width * pixel_width, nr_sub_images_height * pixel_height), color = 0)
	if debuginfo:
		
		print('New composed image:')
		print('Width: ' + str(nr_sub_images_width * pixel_width) + ' Height: ' + str(nr_sub_images_height * pixel_height))
		print('In_width: ' + str(new_composed_image.width) + ' In_height: ' + str(new_composed_image.height) + ' Mode: ' + str(new_composed_image.mode))
		#new_composed_image.show()
		print('#####################')
		
		
	for row in range(nr_sub_images_height):
		subimages_columns = []
		if debuginfo:
			None
			#print('New row ############')
		for column in range(nr_sub_images_width):
			# left, top, right, bottom (x-min, y-min, x-max, y-max)
			coordinates = (column * pixel_width, row * pixel_height, (column + 1) * pixel_width, (row + 1) * pixel_height)
			subimage = pil_base_image.crop(coordinates)
			#subimages_columns.append( {"image" : subimage, "coordinates" : coordinates, "mode" : subimage.mode} )
			
			lightvalue = viirs_light_calculation_zerobased_sum(subimage, lightlimit = lightlimit)
			
			if lightvalue < light_mean_min:
				light_mean_min = lightvalue
			if lightvalue > light_mean_max:
				light_mean_max = lightvalue
			
			if lightvalue <= lower_limit:
				label = 'No_light'
				no_light_list.append(lightvalue)
			elif lightvalue > lower_limit and lightvalue <= middle_limit:
				label = 'Medium_light'
				medium_ligth_list.append(lightvalue)
			else:
				label = 'Bright_light'
				bright_light_list.append(lightvalue)
			
			labels_list.append(label)
			all_lightvalues_list.append(lightvalue)
			matrix_code_list.append(str(row_nr) + '_' + str(column_nr))
			area_code_list.append(areacode)

			sub_image_nr += 1
			#subimage.save(str(row_nr) + '_' + str(column_nr) + '#' + path + filename)
			
			column_nr += 1
			
		#subimages_matrix.append(subimages_columns)
		
		column_nr = startcolumn #
		row_nr += 1
	
	#plot_histogram (no_light_list, selected_bins = 100, prefix = 'Viirs_No_Light_' + areacode + '_')
	#plot_histogram (medium_ligth_list, selected_bins = 100, prefix = 'Viirs_Medium_Light_' + areacode + '_')
	#plot_histogram (bright_light_list, selected_bins = 100, prefix = 'Viirs_Bright_Light_' + areacode + '_')
	
	return all_lightvalues_list, matrix_code_list, labels_list, area_code_list # , row_nr, column_nr # next row and next column...


#########################################
"""
Description:
The function composes a dataframe from arguments and saves it to a csv file in this functions execution folder

Input variables/types and functions:
all_lightvalues:	A list of float-values
matrix_code_list:	A list of matrix-codes as: row_column i.e. 1_3
labels_list:		A list of strings, in this special case: No_light, Medium_light and Bright_light
filename_suffix:	Suffix in addition to fixed prefix: 'Viirs_224_label_info'
effect:				Either saving file and return a dataframe or only return a pandas dataframe... ('File' or 'Dataframe')


Output (data and types):
The effect of a saved csv-file and a returned dataframe or only saving a file

Constraints:
Composed dataframe column names are hardcoded, plus constraints from arguments

Exceptions:
N/A

Others:
Can be modified and improved according to more flexible file saving...

"""

def create_viirs_label_file (all_lightvalues, matrix_code_list, labels_list, area_code_list, filename_suffix = '', effect = 'File'):

	label_dataframe = pandas.DataFrame()
	
	label_dataframe['Light_values_sum'] = all_lightvalues
	label_dataframe['Label_Light_labels'] = labels_list
	label_dataframe['Label_Matrix_code'] = matrix_code_list
	label_dataframe['Label_Area_code'] = area_code_list
	no_light = 0
	medium_ligth = 0
	bright_light = 0
	
	
	label_code_list = []
	for label in labels_list:
		if label == 'No_light':
			label_code_list.append(0)
			no_light += 1
		elif label == 'Medium_light':
			label_code_list.append(1)
			medium_ligth += 1
		else:
			label_code_list.append(2)
			bright_light += 1
	
	label_dataframe['Label_code'] = label_code_list
	
	print('No light nr: ', no_light, ' Medium light nr: ', medium_ligth, ' Bright light nr: ', bright_light)
	
	if effect == 'File':
		#label_dataframe.to_csv('Viirs_TZA_224_label_info' + filename_suffix + '.csv', sep=';', index = False, index_label = False)	
		label_dataframe.to_csv('Viirs_224_label_info_' + filename_suffix + '.csv', sep=';', index = False, index_label = False)
		return label_dataframe
		
	if effect == 'Dataframe':
		return label_dataframe

# Test 
"""
path = '/data/nilcar/Mixed_Methods/Images/Viirs_NN_org/'
filename = 'viirs_avg_rad_median_year_2016_py_scale_750.tif'
pil_base_image = Image.open((path + filename))
all_lightvalues_list, matrix_code_list, labels_list = image_library_functions.create_subimage_matrix_label_data (pil_base_image, 3, 3, 1, 1, path, filename, resize = (1763, 1667), lower_limit = 1, middle_limit = 30, debuginfo = False)
image_library_functions.create_viirs_label_file (all_lightvalues_list, matrix_code_list, labels_list, filename_suffix = '1_30')

sys.exit()
"""


#########################################
"""
Description:
This function creates sub-images 

Input: variables/types and functions:
pil_base_image:		A PIL.Image to divide into subimages
pixel_width:		Desired width of sub-image created
pixel_height:		Desired height of sub-image created
startrow:			Normally 1, might be different if function is used iteratively.
startcolumn:		Normally 1, might be different if function is used iteratively.
path:				Directory where to save created sub-images (must end with "/")
filename:			Desired filename-suffix which will get a prefix in format: "row_column#", I.e. 1_2#filename
areaname:			Used to define imagepathname
resize:				For potential resize of an image
debuginfo:			Whether to get printouts or not

Output: (data and types):
sub_image_nr:		Number of subimages created
rest_image_width:	A PIL image for the left-over in width to the right from pil_base_image
rest_image_height:	A PIL image for the left-over in height in the bottom from pil_base_image
row_nr:				The next row if function is iteratively used.
column_nr:			The next column if function is iteratively used.

Side-effect:		Saving of created sub-images

Constraints:
pixel_width must be less than pil_base_image width
pixel_height must be less than pil_base_image height

Exceptions:
N/A

Others:
N/A

"""

def create_subimage_matrix_images (pil_base_image, pixel_width, pixel_height, startrow, startcolumn, path, filename, areaname = '', resize = None, debuginfo = False):
	
	if resize is not None:
		pil_base_image = pil_base_image.resize(resize, resample = Image.NEAREST)
	
	#print('\n\n##### From create_subimage_matrix_images #####')
	#print('Image to process size: ', pil_base_image.size)
	
	row_nr = startrow
	column_nr = startcolumn

	#subimages_matrix = []
	image_width = pil_base_image.width
	image_height = pil_base_image.height
	image_mode = pil_base_image.mode # Like... 'F' or 'RGB'
	
	sub_image_nr = 0
	max_row_nr = 0
	max_column_nr = 0
	
	# For debug...
	all_rows = []
	all_columns = []
	
	
	if debuginfo:
		print('Base image')
		print('In_width: ' + str(image_width) + ' In_height: ' + str(image_height) + ' Mode: ' + str(image_mode))
		#print('left, top, right, bottom: ')
		print(pil_base_image.size, '\n')
	
	nr_sub_images_width = image_width // pixel_width
	nr_sub_images_height = image_height // pixel_height

	new_composed_image = Image.new(mode = image_mode, size = (nr_sub_images_width * pixel_width, nr_sub_images_height * pixel_height), color = 0)
	if debuginfo:
		
		print('New composed image:')
		print('Width: ' + str(nr_sub_images_width * pixel_width) + ' Height: ' + str(nr_sub_images_height * pixel_height))
		print('In_width: ' + str(new_composed_image.width) + ' In_height: ' + str(new_composed_image.height) + ' Mode: ' + str(new_composed_image.mode))
		#new_composed_image.show()
		print('#####################')
		
		
	for row in range(nr_sub_images_height):
		subimages_columns = []
		if debuginfo:
			None
			#print('New row ############')
		for column in range(nr_sub_images_width):
			# left, top, right, bottom (x-min, y-min, x-max, y-max)
			coordinates = (column * pixel_width, row * pixel_height, (column + 1) * pixel_width, (row + 1) * pixel_height)
			subimage = pil_base_image.crop(coordinates)
			subimages_columns.append( {"image" : subimage, "coordinates" : coordinates, "mode" : subimage.mode} )
			
			sub_image_nr += 1
			#print('Save as: ' + str(row_nr) + '_' + str(column_nr))
			#print(path)
			#sys.exit()
			if areaname != '':
				subimage.save(path + areaname + '_' + str(row_nr) + '_' + str(column_nr) + '#' + filename) #
			else:
				subimage.save(path + str(row_nr) + '_' + str(column_nr) + '#' + filename) # TZA Rectangle
			
			
			#print('In_width: ' + str(subimages_columns[-1].width) + ' In_height: ' + str(subimages_columns[-1].height) + ' Mode: ' + str(subimages_columns[-1].mode))
			
			if debuginfo:
				None
				#print(coordinates)
			
			all_columns.append(column_nr)
			
			column_nr += 1
			
		#subimages_matrix.append(subimages_columns)
		
		if column_nr > max_column_nr:
			max_column_nr = column_nr
		
		column_nr = startcolumn
		all_rows.append(row_nr)
		row_nr += 1
		
		if row_nr > max_row_nr:
			max_row_nr = row_nr
	
	column_nr = max_column_nr
	
	relative_rest_width = (coordinates[2], 0, image_width, image_height)
	rest_image_width = pil_base_image.crop(relative_rest_width)
	relative_rest_height = (0, coordinates[3], image_width, image_height)
	rest_image_height = pil_base_image.crop(relative_rest_height)
	pil_array = numpy.array(pil_base_image)
	
	#print('\n\n##### From create_subimage_matrix_images #####')
	#print('Last coordinates: ', coordinates, '\n')
	#print('Relative rest width: ', relative_rest_width)
	#print('Sub function: Rest image width size: ', rest_image_width.size) # OK
	#rest_image_width.show()
	#print('Relative rest height: ', relative_rest_height)
	#print('Sub function: Rest image height size: ', rest_image_height.size)
	#rest_image_height.show()	
	#print('Row_out: ', row_nr, ' column out: ', column_nr)
	#print('Max row nr: ', max_row_nr, ' Max column nr: ', max_column_nr)
	#print('PIL array shape: ', pil_array.shape)
	#print('Min: ', numpy.amin(pil_array), ' Max: ', numpy.amax(pil_array))
	#print('Created subimages: ', sub_image_nr)
	#print('##### END From create_subimage_matrix_images #####\n\n')
	
	# For debug, move to main function...
	#plot_histogram (all_rows, len(all_columns), prefix = 'rows: ')
	#plot_histogram (all_columns, len(all_rows), prefix = 'columns: ')
	
	return sub_image_nr, rest_image_width, rest_image_height, row_nr, column_nr # next row and next column...


# Test
"""
pil_base_image = Image.open('/data/nilcar/Mixed_Methods/Images/Sentinel/Sentinel_1C_scale_10_p50_org/Converted_to_pil_images/gdal_to_pil_rgb_uint8_min_191_maxscale_1714_sentinel_least_cloudy_p50_year_py_scale_10-0000000000-0000000000.tif')
create_subimage_matrix_images (pil_base_image, 224, 224, 1, 1, '', '', debuginfo = False)
sys.exit()
"""

"""
Save as: 60_60
Last coordinates:  (13216, 13216, 13440, 13440)

(13440, 0, 13568, 13568)
(128, 13568)
(0, 13440, 13568, 13568)
(13568, 128)
Row_out:  61  column out:  61
Max row nr:  61  Max column nr:  61
PIL array shape:  (13568, 13568, 3)
Min:  25  Max:  255
Created subimages:  3600

"""


#########################################
"""
Description:
Reading a directory of files (only in that directory, not recusively downwards) the function organizes 
the files and their info in a matrix shape (row, column).

Input variables/types and functions:
directory:	The directory holding all files to be used (not more not less), can have subdirectories which will be ignored.
separator:	Used for splitting given filnames and used in filename composition for matrix output

Output (data and types):
main_matrix:	Holding rows and columns with directory and filenames: [row[[directory, filename] ...] ...] # shape(rows, columns, 2)

Constraints:
The files must have names that can be sorted and splitted in order: prefix, separator, lat, separator, long (including filetype I.e. .tif).
The filenames sort order must support the order lat, long (sort order respectively) regardless of prefix!

Exceptions:
N/A

Others:
This function assumes files and their filenames downloaded using Google Earth Engine
The function is a support-function for further usage for indata handling to a Neural network (necessary separator for that is '-')

Note: the function works on meta-data for images and not the images themselfes...
"""

def create_main_image_matrix (directory, separator = '-'):

	datafiles = []
	file_info = []
	matrix_row = []
	main_matrix = []
	lat_reference = ''
	first_file = True
	
	print(directory)
	
	for item in listdir(directory): 
		if isfile(join(directory, item)):
			datafiles.append(directory + item)
	datafiles.sort()
	print('Nr of files: ', str(len(datafiles)))
	#sys.exit()
	
	for filepath in datafiles:
		
		#print('Filepath: ', filepath)
		#sys.exit()
		
		#print('Checking file...')
		file_info = []
		filenameprefix, lat, long = filepath.split(separator) # filenameprefix includes directory, long includes filetype suffix
		filenameprefix = filenameprefix.replace(directory, '')
		
		#print('fp: ', filenameprefix, ' lat: ', lat, ' long: ', long)
		#sys.exit()
		
		#long, filesuffix = long.split('.')
		file_info.append(directory)
		file_info.append(filenameprefix + separator + lat + separator + long) # full filename including filetype suffix
		
		#print(file_info)
		#sys.exit()
		
		if first_file:
			print('First file')
			lat_reference = lat
			matrix_row.append(file_info) # first item in first row
			first_file = False
		else:
			#print('NOT first file')
			if lat == lat_reference: # same row
				#print('Same row: ', lat_reference)
				matrix_row.append(file_info)
			else: # new row
				lat_reference = lat
				print('New row: ', lat_reference)
				main_matrix.append(matrix_row)
				#print('Appended row: ', matrix_row)
				matrix_row = []
				matrix_row.append(file_info)
	
	main_matrix.append(matrix_row)
	
	print(main_matrix[0][0])
	print('\n\n Shape: ', numpy.array(main_matrix).shape)
	
	return main_matrix
		
		

#create_main_image_matrix ('/data/nilcar/Mixed_Methods/Images/Sentinel/Sentinel_1C_scale_10_p45_org/')

#main_matrix = create_main_image_matrix ('/nfs/home/nilcar/Mixed_Methods/TZA_rectangle/Sentinel_py_L1C_tza_scale_10_p25_org/Converted/', separator = '-')

#sys.exit()
	
	
##############################
"""
Description:
Given a main_image_matrix holding Meta data for PIL-images it creates organised subimages (rows, columns) of desired sub-size and saves them in the given directory.


Input variables/types and functions:
main_images:		as 2D matrix (nested list, rows and columns) with each "prime" element as a list of path and filename including filename suffix (filetype like ".tif")
network_imagesize:	The size of desired sub-images as a tuple: (width, height) in pixels. I.e. for Resnet50 (224, 224)
saving_directory:	A subdirectory in the directory given from main_images matrix, must end with "/"
areaname:			Used to define imagepathname
resize:				For potential resize of an image
debug:				Wether to get debug print-outs during execution (for this function its done regardless of setting...) (but will affect calls to sub-functions).

Output (data and types):
None

Side-effect:		Created sub-images are stored according to given arguments.

Constraints:
network_imagesize: Must be smaller than size of images pointed out in main_images

Exceptions:
N/A

Others:
This function is especially developed for images extracted using Google Earth Engine

"""

def create_subimage_main_matrix (main_images = [], network_imagesize = (0, 0), saving_directory = '', areaname = '', resize = None, debug = False):

	debug = debug
	first_image = True
	row_nr = 0
	column_nr = 0
	next_row_nr = 0
	next_column_nr = 0
	start_row_nr = 1
	start_column_nr = 1
	rest_images_width = []
	rest_images_height = []
	row_size = 0 # Number of base images in one row
	image_nr = 0 # base image nr
	sub_image_nr = 0 # generated nr of total subimages
	
	break_row = 0


	# For each main image; extract and store subimages, keep x-rest and y-rest-list.
	# for next main image, add y-rest (not first row) to top and x-rest to left (not first image in row)
	for main_row in main_images:
		
		break_row += 1 #
		if break_row >= 2:
			#break
			None
		
		#print('############## Main row ##############')
		
		column_nr = 0
		#subimages_matrix = []
		rest_images_width = []
	
		for base_image in main_row:
			# subimages for each main image.
			image_nr += 1
			#print('Base image nr: ', image_nr)
			
			# read base_image as PIL image from path
			path = base_image[0]
			filename = base_image[1]
			
			#print('Path-file to subdivide: ' + path + filename)
			#sys.exit()
			
			base_image = Image.open(path + filename) # base_image = Image.open('gdal_to_pil_rgb_uint8_min_191_maxscale_1714_sentinel_1C_least_cloudy_p45_year_2016_float32_py_scale_10_GEO_TIFF-0000113664-0000113664.tif')
			image_mode = base_image.mode
			base_coordinates = (0, 0, base_image.width, base_image.height)
			print('base image pixelsize: ', base_image.width, ': ', base_image.height)
			base_image = base_image.crop(base_coordinates)
			path = saving_directory # Path to subdirectory for storing created subimages...
			
			#sys.exit()
			
			if row_nr == 0:
				if first_image: # First image(0) and first row(0), extract subimages and store x-rest(width) and y-rest(height)
					
					row_size = 1
					#print('Row: ' + str(row_nr + 1) + ' Column: ' + str(column_nr + 1))
					#print('Subimages_matrix shape: ', numpy.array(subimages_matrix).shape)
					created_images, rest_image_width, rest_image_height, next_row_nr, next_column_nr = create_subimage_matrix_images (base_image, network_imagesize[0], network_imagesize[1], start_row_nr, start_column_nr, path, filename, areaname = areaname, resize = resize, debuginfo = debug)
					rest_images_width.append(rest_image_width)
					rest_images_height.append(rest_image_height)
					sub_image_nr += created_images
					# Store subsub images...
					#subimages_matrix = add_submatrix_to_main (subimages_matrix, subsub_images)
					first_image = False
					#start_column_nr = next_column_nr
					
					#print(subimages_matrix)
					#sys.exit()
					
					#print('Subimages_matrix shape: ', numpy.array(subimages_matrix).shape)
					#print('Row: ' + str(row_nr) + ' Column: ' + str(column_nr))
					
					#print('Base_image size: ', base_image.size)
					#print('Rest_width image size: ', rest_images_width[column_nr].size)
					#print('Rest_height image size: ', rest_images_height[column_nr].size)
					#print('**************** End main case ***************')
					
					#break
					
				else: # First row, second columns plus, add x-rest to left, extract subimages and store new x-rest and y-rest
					
					row_size += 1
					#print('Row: ' + str(row_nr + 1) + ' Column: ' + str(column_nr + 1))
					start_column_nr = next_column_nr
					#print('Base_image: ', base_image.size)
					#print('Rest_width_image: ', rest_images_width[column_nr - 1].size)
					
					# New empty image for pasting base_image and previous width/height rest...
					new_image = Image.new(mode = image_mode, size = (base_image.width + rest_images_width[column_nr - 1].width, base_image.height), color = 0)
					#print('New EMPTY image size: ', new_image.size)
					#sys.exit()
					
					# Box for pasting base image
					base_box = (rest_images_width[column_nr - 1].width, 0, base_image.width + rest_images_width[column_nr - 1].width, rest_images_width[column_nr - 1].height)
					# Paste base image
					new_image.paste(base_image, box = base_box)
					# Paste x-rest (width) to left of base image
					rest_box_width = (0, 0, rest_images_width[column_nr - 1].width, rest_images_width[column_nr - 1].height)
					new_image.paste(rest_images_width[column_nr - 1], box = rest_box_width)
					
					#print('Box for pasting base image: ', base_box)
					#print('Box for pasting rest width image: ', rest_box_width)
					
					created_images, rest_image_width, rest_image_height, next_row_nr, next_column_nr = create_subimage_matrix_images (new_image, network_imagesize[0], network_imagesize[1], start_row_nr, start_column_nr, path, filename, areaname = areaname, resize = resize, debuginfo = debug)
					rest_images_width.append(rest_image_width)
					rest_images_height.append(rest_image_height)
					sub_image_nr += created_images
					
					# Store subsub images...
					#subimages_matrix = add_submatrix_to_main (subimages_matrix, subsub_images)
					
					#print('Subimages_matrix shape: ', numpy.array(subimages_matrix).shape)
					
					"""
					print('Base_image size: ', base_image.size)
					print('Rest_width image size: ', rest_images_width[column_nr].size)
					print('Rest_height image size: ', rest_images_height[column_nr].size)
					print('New image size: ', new_image.size)
					print('**************** End main case ***************')
					"""
					
					#new_image.show()
					#new_image.save("new_image.tif")
					#sys.exit()
					
			else: # Second row plus... Use "row_size" for rest_images_height
				
				if column_nr == 0:
					#print('Row: ' + str(row_nr + 1) + ' Column: ' + str(column_nr + 1))
					start_column_nr = 1
					start_row_nr = next_row_nr
					# Add y-rest to top, extract subimages and store new x-rest and y-rest
					# New empty image for pasting base_image and previous width/height rest...
					
					#new_image = Image.new(mode = image_mode, size = (base_image.width, base_image.height + rest_images_height[column_nr].height), color = 0)
					new_image = Image.new(mode = image_mode, size = (base_image.width, base_image.height + rest_images_height[(row_nr -1) * row_size + column_nr].height), color = 0)
					#print('New EMPTY image size: ', new_image.size)
					#print('New image: ', new_image.size)
					#sys.exit()
					
					base_box = (0, rest_images_height[(row_nr -1) * row_size + column_nr].height, base_image.width, base_image.height + rest_images_height[(row_nr -1) * row_size + column_nr].height)
					#print(base_box)
					# Paste base image
					new_image.paste(base_image, box = base_box)
					
					# Paste y-rest (height) to top of base image
					rest_box_height = (0, 0, rest_images_height[(row_nr -1) * row_size + column_nr].width, rest_images_height[(row_nr -1) * row_size + column_nr].height)
					#print(rest_box_width)
					new_image.paste(rest_images_height[(row_nr -1) * row_size + column_nr], box = rest_box_height)
					
					created_images, rest_image_width, rest_image_height, next_row_nr, next_column_nr = create_subimage_matrix_images (new_image, network_imagesize[0], network_imagesize[1], start_row_nr, start_column_nr, path, filename, areaname = areaname, resize = resize, debuginfo = debug)
					rest_images_width.append(rest_image_width)
					rest_images_height.append(rest_image_height)
					sub_image_nr += created_images
					
					# Store subsub images...
					#subimages_matrix = add_submatrix_to_main (subimages_matrix, subsub_images)
					
					#print('Subimages_matrix shape: ', numpy.array(subimages_matrix).shape)
					
					"""
					print('Base_image size: ', base_image.size)
					print('Rest_width image: ', rest_images_width[column_nr].width)
					print('Rest_height image: ', rest_images_height[(row_nr -1) * row_size + column_nr].height)
					print('New image size base + rest-height: ', new_image.size)
					print('**************** End main case ***************')
					"""
					
					#new_image.show()
					#new_image.save("new_image.tif")
					#sys.exit()
					# OK
				
				else: # Second column plus...
					#print('Row: ' + str(row_nr + 1) + ' Column: ' + str(column_nr + 1))
					start_column_nr = next_column_nr
					# Add y-rest to top, add x-rest to left, extract subimages and store new x-rest and y-rest
					new_image = Image.new(mode = image_mode, size = (base_image.width + rest_images_width[column_nr - 1].width, base_image.height + rest_images_height[(row_nr -1) * row_size + column_nr].height), color = 0)
					#print('New EMPTY image size: ', new_image.size)
					#print('New image: ', new_image.size)
					#sys.exit()
					
					base_box = (rest_images_width[column_nr - 1].width, rest_images_height[(row_nr -1) * row_size + column_nr].height, base_image.width + rest_images_width[column_nr - 1].width, base_image.height + rest_images_height[(row_nr -1) * row_size + column_nr].height)
					#print(base_box)
					# Paste base image into right place depending on x-rest and y-rest
					new_image.paste(base_image, box = base_box)
					
					# Paste y-rest (height) to top of base image
					rest_box_height = (0, 0, rest_images_height[(row_nr -1) * row_size + column_nr].width, rest_images_height[(row_nr -1) * row_size + column_nr].height)
					#print(rest_box_width)
					new_image.paste(rest_images_height[(row_nr -1) * row_size + column_nr], box = rest_box_height)
					
					# Paste x-rest (width) to left of base image
					rest_box_width = (0, 0, rest_images_width[column_nr - 1].width, rest_images_width[column_nr - 1].height)
					#print(rest_box_width)
					new_image.paste(rest_images_width[column_nr - 1], box = rest_box_width)
					
					created_images, rest_image_width, rest_image_height, next_row_nr, next_column_nr = create_subimage_matrix_images (new_image, network_imagesize[0], network_imagesize[1], start_row_nr, start_column_nr, path, filename, areaname = areaname,  resize = resize, debuginfo = debug)
					rest_images_width.append(rest_image_width)
					rest_images_height.append(rest_image_height)
					sub_image_nr += created_images
					
					# Store subsub images... TBD
					#subimages_matrix = add_submatrix_to_main (subimages_matrix, subsub_images)
					
					#print('Subimages_matrix shape: ', numpy.array(subimages_matrix).shape)
					
					"""
					print('Base_image size: ', base_image.size)
					print('Rest_width image size: ', rest_images_width[column_nr].size)
					print('Rest_height image size: ', rest_images_height[(row_nr -1) * row_size + column_nr].size)
					print('New image size base + rest-width + rest-height: ', new_image.size)
					print('**************** End main case ***************')
					"""
					
					#new_image.show()
					#new_image.save("new_image.tif")
					#sys.exit()
					
					
			column_nr += 1
			
		# End columns for all of us who loves Ada...	
		
		row_nr += 1
		
	# End rows for all of us who loves Ada again...	
	
	"""
	print('\n\nFrom. create_subimage_main_matrix')
	print('Created images: ', sub_image_nr)
	print('Last_next row: ', next_row_nr, ' Last next column: ', next_column_nr)
	print('\n\n')	
	"""	
	#print('final image_matrix shape: ', numpy.array(final_image_matrix).shape)		
	

# Test Areas
# Paris

# 2023-08 TZA Rectangle...
"""
main_matrix = create_main_image_matrix ('/nfs/home/nilcar/Mixed_Methods/TZA_rectangle/Sentinel_py_L1C_tza_scale_10_p25_org/Converted/', separator = '-')
create_subimage_main_matrix (main_matrix, network_imagesize = (224, 224), saving_directory = '/nfs/home/nilcar/Mixed_Methods/Sentinel_p25_224px_images/TZA_Rectangle/', areaname = '')
sys.exit()
"""


"""
main_images = create_main_image_matrix ('/data/nilcar/Mixed_Methods/Cities/Sentinel_Original/Converted/')
create_subimage_main_matrix (main_images, network_imagesize = (224, 224), saving_directory = 'NN_224_11_Cities_pil_images/', areaname = '', resize = (6032, 6012))
sys.exit()
"""


"""
General:
# Sentinel p50 converged images to pil...
Total image width:  131595  Total height:  13568 OK
First 1 row should be: width: 13568 x 9 + 9483 = 131595px	height: 13568 x 1 => 13568px (224, rows/columns = 60,587) (w-rest = 107, h-rest = 128)

Row nr1: 
Base_image size:  (9483, 13568)
Rest_width image size:  (107, 13568)
Rest_height image size:  (9515, 128)
New image size:  (9515, 13568)
Created images:  35220
Last_next row:  61  Last next column:  588 OK
Sub function: Rest image height size:  (9515, 128) OK

Row nr2: 
Base_image size:  (9483, 13568)
Rest_width image size:  (107, 13696)
Rest_height image size:  (9515, 128)
New image size base + rest-width + rest-height:  (9515, 13696)
Created images:  71027
Last_next row:  122  Last next column:  588 OK
Sub function: Rest image height size:  (9515, 32) OK

Row nr3: 
Base_image size:  (9483, 13568)
Rest_width image size:  (107, 13600)
Rest_height image size:  (9515, 32)
New image size base + rest-width + rest-height:  (9515, 13600)
Created images:  106247
Last_next row:  182  Last next column:  588
Sub function: Rest image height size:  (9515, 160)

Row nr4:
Base_image size:  (9483, 13568)
Rest_width image size:  (107, 13728)
Rest_height image size:  (9515, 160)
New image size base + rest-width + rest-height:  (9515, 13728)
Created images:  142054
Last_next row:  243  Last next column:  588
Sub function: Rest image height size:  (9515, 64)

..................

Row nr9:
Base_image size:  (9483, 13568)
Rest_width image size:  (107, 13696)
Rest_height image size:  (9515, 128)
New image size base + rest-width + rest-height:  (9515, 13696)
Created images:  319915
Last_next row:  546  Last next column:  588
Sub function: Rest image height size:  (9515, 32)

Row nr10:
Base_image size:  (9483, 2357)
Rest_width image size:  (107, 2389)
Rest_height image size:  (9515, 32)
New image size base + rest-width + rest-height:  (9515, 2389)
Created images:  325785
Last_next row:  556  Last next column:  588
Sub function: Rest image width size:  (107, 2389)
Sub function: Rest image height size:  (9515, 149)

ALL OK!

"""



###########################
# Areas Sentinel 
# No files in base_directory!

"""
Description:
Given a base directory the function returns all underlaying datadirectories and areanames (recursively)

Input variables/types and functions:
base_directory:			The directory to investigate
datadirectories = []	An empty list
areanames = []			An empty list

Output (data and types):

Constraints:
N/A

Exceptions:
N/A

Others:
N/A

"""

def find_area_directories (base_directory, datadirectories = [], areanames = []):
	
	for item in listdir(base_directory):
		if isfile(join(base_directory, item)):
			#datafiles.append(base_directory + item)
			None
		else:
			#print(base_directory + item + '/')
			datadirectories.append(base_directory + item  + '/')
			areanames.append(item)
			find_area_directories(base_directory + item + '/', datadirectories, areanames)
	
	return datadirectories, areanames
	
	

	
# Test: OK

#directories, areanames = find_area_directories('/data/Mixed_Methods/11_Cities/Sentinel_Original/Converted/')

"""directories, areanames = find_area_directories('/data/Mixed_Methods/Sentinel_p45_224px_images/')

print(directories)
print(len(directories))
print(areanames)
print(len(areanames))
"""

"""
['/data/Mixed_Methods/11_Cities/Sentinel_Original/Converted/Bujumbura/',  ... ]

['Bujumbura', 'Gitega', 'Harare', 'Kampala', 'Kigali', 'Lilonwe', 'Lusaka', 'Mogadishu', 'Mombasa', 'Nairobi', 'Nakaru']
"""


"""
Description:


Input variables/types and functions:


Output (data and types):


Constraints:
N/A

Exceptions:
N/A

Others:
N/A

"""

def create_NN_area_images (base_directory):
	
	areanames = []
	datadirectories, areanames = find_area_directories(base_directory, areanames) # Source directories for files to convert to NN files
	
	#print('Find function:')
	print(datadirectories)
	#print(areanames)
	
	
	for index in range(len(datadirectories)):
		
		print(datadirectories[index])
		print(areanames[index])

		main_images = create_main_image_matrix (datadirectories[index])
		print('Main matrix' + str(index))
		print(main_images)
		
		#sys.exit()
		
		create_subimage_main_matrix (main_images, network_imagesize = (224, 224), saving_directory = '/nfs/home/nilcar/Mixed_Methods/Sentinel_p25_224px_images/' + areanames[index] + '/' , areaname = areanames[index]) #, resize = (6023, 6012)) # Uses: create_subimage_matrix_images, (saves the file)
		
		#sys.exit()
		
	#datafiles.sort()


# Test OK
#create_NN_area_images ('/data/Mixed_Methods/11_Cities/Sentinel_Original/Converted/')

# 2023_08... Changed filenames to removed one '-' and added relative position in the end: -0.0-0.0; OK
#create_NN_area_images ('/nfs/home/nilcar/Mixed_Methods/11_Cities/Sentinel_Original_p25/Converted/')



###########################
# Areas Viirs

# all_lightvalues_list, matrix_code_list, labels_list = create_subimage_matrix_label_data (pil_base_image, 3, 3, 1, 1, path, filename, resize = None, lower_limit = 9, middle_limit = 50, debuginfo = False)
# create_viirs_label_file (all_lightvalues_list, matrix_code_list, labels_list, filename_suffix = '1_30')

"""
Description:
This function creates a label dataframe


Input variables/types and functions:
base_directory:	The top directory for images that may have subdirectories
lower_limit:	The upper limit for No_Light values
middle_limit:	The upper limit for Medium_Light values
lightlimit:		The limit that is defined for a pixelvalue shall be considered to be defined having light
resize:			The desired size for an image.

Output (data and types):
A dataframe with labeldata.

Constraints:
N/A

Exceptions:
N/A

Others:
N/A

"""

def create_NN_area_labeldata (base_directory, lower_limit, middle_limit, lightlimit = 0.5, resize = None):
	
	label_dataframe = pandas.DataFrame()
	areanames = [] # Last folder name
	datadirectories, areanames = find_area_directories(base_directory, areanames) # Source directories for files to convert to NN labeldata file
	
	print('Data_directories: ', datadirectories)
	
	print('Areanames: ', areanames)
	
	#sys.exit()
	
	
	pixel_width = 3
	pixel_height = 3
	startrow = 1
	startcolumn = 1
	
	for index in range(len(datadirectories)):
		
		datafiles = []
		for item in listdir(datadirectories[index]): 
			if isfile(join(datadirectories[index], item)):
				datafiles.append(item)
		
		print('Main loop \n\n')
		print(datadirectories[index] + datafiles[0])
		#sys.exit()
		
		print('Nr of files: ', len(datafiles))
		
		for filename_index in range(len(datafiles)):
			print('Directory: ', datadirectories[index], 'File: ', datafiles[filename_index])
			pil_base_image = Image.open(datadirectories[index] + datafiles[filename_index]) # path plus filename
			
			areacode = areanames[index]
			
			# TZA surroundings special only, omitted for 11 cities and TZA_rectangle ...
			if 'TZA_Surroundings' in base_directory:
				areacode = datafiles[filename_index].split('-_')
				areacode = areacode[0]
			
			# path = '', filename = '' (not implemented in function yet...)
			all_lightvalues_list, matrix_code_list, labels_list, area_code_list = create_subimage_matrix_label_data (pil_base_image, pixel_width, pixel_height, startrow, startcolumn, path = '', filename = '', resize = resize, lower_limit = lower_limit, middle_limit = middle_limit, areacode = areacode, lightlimit = lightlimit, debuginfo = False)
			
			#plot_histogram (all_lightvalues_list, selected_bins = 100, prefix = 'Viirs_Light_values_' + areanames[index] + '_' + str(lower_limit) + '_' + str(middle_limit), save = True)
			
			label_data = create_viirs_label_file (all_lightvalues_list, matrix_code_list, labels_list, area_code_list, filename_suffix = str(lower_limit) + '_' + str(middle_limit), effect = 'Dataframe')
			#label_dataframe = label_dataframe.append(label_data)
			label_dataframe = pandas.concat([label_dataframe, label_data])
			
	#sys.exit()

	# Debug
	print(label_dataframe.shape)
	print(label_dataframe.head())
	#sys.exit()
	
	return label_dataframe



# Test/Create...
#lower_limit = 5
#middle_limit = 50
#label_dataframe = create_NN_area_labeldata ('/data/Mixed_Methods/TZA_rectangle/Viirs/TZA_rectangle/', lower_limit, middle_limit)
#label_dataframe = create_NN_area_labeldata ('/data/Mixed_Methods/11_Cities/Viirs/', lower_limit, middle_limit, lightlimit = 0.5) # resize = (27,27)
#label_dataframe.to_csv('Viirs_224_label_info' + '_11_Cities_' + str(lower_limit) + '_' + str(middle_limit) + '.csv', sep=';', index = False, index_label = False)


# Help functions
# def create_subimage_matrix_label_data (pil_base_image, pixel_width, pixel_height, startrow, startcolumn, path = '', filename = '', resize = None, lower_limit = 5, middle_limit = 50, areacode = '', lightlimit = 0.5, debuginfo = False):
# create_viirs_label_file (all_lightvalues_list, matrix_code_list, labels_list, area_code_list, filename_suffix = str(lower_limit) + '_' + str(middle_limit), effect = 'Dataframe')


"""
# 2021-09
print('Labeling')
base_directory = '/data/home/nilcar/Mixed_Methods/11_Cities/Viirs/' # Country name directories on level below.
lower_limit = 1
middle_limit = 10
# resize = (27,27)) # 3 pixels * 9 (this case) TZA_Surroundings, TZA_rectangle: 587*3, 555*3 => (1761,1665), 11_Cities: 26*3, 26*3 => (78,78)
label_dataframe = create_NN_area_labeldata (base_directory, lower_limit, middle_limit, lightlimit = 0.5, resize = (78,78)) 
label_dataframe.to_csv('Viirs_224_label_info' + '_11_Cities_complete_ll_0_5_cl_' + str(lower_limit) + '_' + str(middle_limit) + '.csv', sep=';', index = False, index_label = False)
print('Finished labeling')
"""

"""
# 2022-01-26 Lightlimit 0.4 and 0.6 3 Areas...
# 2022-01 TZA_rectangle, 11_Cities, TZA_Surroundings
print('Labeling')
base_directory = '/data2/home/nilcar/Mixed_Methods/TZA_Surroundings/Viirs/' # Country name directories on level below.
lower_limit = 1
middle_limit = 10
#resize = (1761,1665) # (27,27) # 3 pixels * 9 (this case) TZA_Surroundings # TZA_rectangle: 587*3, 555*3 => (1761,1665), 11_Cities: 26*3, 26*3 => (78,78)
label_dataframe = create_NN_area_labeldata (base_directory, lower_limit, middle_limit, lightlimit = 0.0, resize = (27,27))
label_dataframe.to_csv('Results/Viirs_224_label_info' + '_TZA_Surroundings_complete_lightlimit_0#5_cl_' + str(lower_limit) + '_' + str(middle_limit) + '.csv', sep=';', index = False, index_label = False)
print('Finished labeling')
"""







""" Test data
['Bujumbura', 'Gitega', 'Harare', 'Kampala', 'Kigali', 'Lilonwe', 'Lusaka', 'Mogadishu', 'Mombasa', 'Nairobi', 'Nakaru']

# 5,50-0.5
Min:  -0.07001154  Max:  11.532562
No light nr:  657  Medium light nr:  18  Bright light nr:  1
Min:  -0.08001681  Max:  4.5175257
No light nr:  673  Medium light nr:  3  Bright light nr:  0
Min:  -0.03887289  Max:  34.56091
No light nr:  462  Medium light nr:  152  Bright light nr:  62
Min:  -0.061682723  Max:  57.256454
No light nr:  493  Medium light nr:  153  Bright light nr:  30
Min:  -0.08385721  Max:  38.47431
No light nr:  621  Medium light nr:  41  Bright light nr:  14
Min:  -0.08456908  Max:  20.49971
No light nr:  611  Medium light nr:  42  Bright light nr:  23
Min:  -0.033380367  Max:  52.08593
No light nr:  473  Medium light nr:  134  Bright light nr:  69
Min:  -0.049364805  Max:  23.887726
No light nr:  658  Medium light nr:  15  Bright light nr:  3
Min:  -0.04055534  Max:  159.57983
No light nr:  570  Medium light nr:  83  Bright light nr:  23
Min:  -0.015545432  Max:  55.795662
No light nr:  338  Medium light nr:  240  Bright light nr:  98
Min:  -0.063580126  Max:  6.0173855
No light nr:  676  Medium light nr:  0  Bright light nr:  0

### TZA-rectangle: resize = (1763, 1667) 
Min:  -0.121051714  Max:  1253.0894
No light nr:  323966  Medium light nr:  1531  Bright light nr:  288





"""





###########################






"""
Description:
A plot function for an image in an image_matrix

Input variables/types and functions:
image_matrix:	A matrix (nested list) rows/columns where each element must be a dict at least holding a real PIL-image with the key 'image'

Output (data and types):
None

Side-effect: The composed image is saved in the execution directory with hard-coded filename

Constraints:
image_matrix must not be empty and holding at least one row and one column per row...

Exceptions:
N/A

Others:
Can easily be improved to take desired directory and filename as arguments...
It does some debug print-out during execution...

"""

def show_matrix_image (image_matrix):
	
	image_mode = image_matrix[0][0]["image"].mode
	print('Image mode: ', image_mode)
	
	total_width = 0
	total_height = 0
	first_row = True
	for row in image_matrix:
		total_height += row[0]["image"].height
		for column in row:
			if first_row:
				total_width += column["image"].width
		first_row = False		
				
	new_image = Image.new(mode = image_mode, size = (total_width, total_height), color = 0)
	print('Composed image size: ', new_image.size)
	
	start_x = 0
	start_y = 0
	end_x = 0
	end_y = 0
	new_row = True
	row_nr = 1
	
	for row_images in image_matrix:
		print('Row nr: ', row_nr)
		start_x = 0
		end_x = 0
		start_y = end_y
		new_row = True
		for column_images in row_images:
			image = column_images["image"]
			if new_row:
				start_x = end_x
				start_y = end_y
				end_x += image.width
				end_y += image.height
				
			else:
				start_x = end_x
				# start_y remains the same
				end_x += image.width	
				# end_y remains the same	
					
			paste_box = (start_x, start_y, end_x, end_y)
			new_image.paste(image, box = paste_box)
			new_row = False
		row_nr += 1	
			
	#new_image.show()
	new_image.save("Composed_image_2_3.tif")




###########################
"""
Description:
This function is supposed to build a main matrix from submatrices...
Note: It is NOT used...


Input variables/types and functions:
base_matrix:	
new_matrix:		

Output (data and types):
base_matrix:	

Constraints:
N/A

Exceptions:
N/A

Others:
To be further implemented and used...

"""
def add_submatrix_to_main (base_matrix = [], new_matrix = []):
	
	if len(base_matrix) == 0:
		return new_matrix
	
	base_row_nr = len(base_matrix)
	new_row_nr = len(new_matrix[0])
	#print('Subimages_matrix shape start: ', numpy.array(base_matrix[0]).shape)
	#print('New_matrix shape start: ', numpy.array(new_matrix[0]).shape)

	for base_row in range(base_row_nr):
		for new_row in range(new_row_nr):
			base_matrix[base_row].append(new_matrix[base_row][new_row])
			#print('New_matrix row shape adding: ', numpy.array(base_matrix[base_row]).shape)
	
	return base_matrix
	

################### END library functions... ###################




###################### Only partial tests for debugging...
"""
print(image_iceland_org.width)
print(image_iceland_org.height)
print(image_iceland_org.size)

cropped_image = image_iceland_org.crop((0, 0, image_iceland_org.width, image_iceland_org.height))

#new_image = Image.new(mode = image_iceland_org.mode, size = (image_iceland_org.width, image_iceland_org.height), color = 0)
new_image = Image.new(mode = image_iceland_org.mode, size = (4000, 4000), color = 0)

#new_image.show()

print(new_image.size)
print(new_image.mode)

#new_image.paste(image_iceland_org, box = (0, 0, image_iceland_org.width, image_iceland_org.height))
new_image.paste(cropped_image, box = (10, 10))
#new_composed_image.paste(column_images["image"], box = column_images["coordinates"])

new_image.show()

sys.exit()

#image_iceland_org.show()

#sys.exit()

#create_subimage_matrix_test (image_iceland_org, 100, 100, debuginfo = True)

subimages_matrix, rest_image_width, rest_image_height = create_subimage_matrix (image_iceland_org, 100, 100, debuginfo = True)

#rest_image_height.show()
print(rest_image_width.width)
print(rest_image_width.height)
print(rest_image_width.size)

#rest_image_width.show()
print(rest_image_height.width)
print(rest_image_height.height)
print(rest_image_height.size)

sys.exit()

"""


##### Tensorflow

#import tensorflow
#from tensorflow.keras.preprocessing import image

""" Tensorflow:
References: 
https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/img_to_array
https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/load_img

https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/array_to_img
https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image

# img = PIL.Image, formats applicable ?
tf.keras.preprocessing.image.img_to_array(
    img,
    data_format=None,
    dtype=None
)

# 4-dimensional: samples, rows, columns, and channels.

image_batch = np.expand_dims(numpy_image_array, axis=0)

data.append(numpy_image_array) # data np_array ?

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0 # Necessary ?


from keras.applications.vgg16 import preprocess_input
# prepare the image for the VGG model
image = preprocess_input(image) # Is doing what ?

"""

"""
print('TF_tif')
tf_pil_image_tif = image.load_img(path = "Iceland_July_1992_tif.tif", color_mode='rgb')
print(tf_pil_image_tif.mode)
print(tf_pil_image_tif.size)
print(tf_pil_image_tif.info)
tf_np_array_tif = image.img_to_array(tf_pil_image_tif, data_format = None, dtype = None)
print(tf_np_array_tif.shape)

TF_tif
RGB
(2552, 3510)
{'compression': 'tiff_lzw', 'dpi': (300, 300)}
(3510, 2552, 3)

"""

"""
print('\n\nTF_png')
tf_pil_image_png = image.load_img(path = "Iceland_July_1992_png.png", color_mode='rgb')
print(tf_pil_image_png.mode)
print(tf_pil_image_png.size)
print(tf_pil_image_png.info)
#tf_pil_image_png.show()
tf_np_array_png = image.img_to_array(tf_pil_image_png, data_format = None, dtype = None)
print(tf_np_array_png.shape)

RGB
(3510, 2552)
{'srgb': 0, 'gamma': 0.45455, 'dpi': (96, 96)}
(2552, 3510, 3)

"""


"""
print('\n\nTF_Viirs_tif')
tf_pil_image_viirs_tif = image.load_img(path = "viirs_average_radiance_year_py_500.tif", color_mode='rgb')
print(tf_pil_image_viirs_tif.mode)
print(tf_pil_image_viirs_tif.size)
print(tf_pil_image_viirs_tif.info)
#tf_pil_image_viirs_tif.show()
tf_np_array_viirs_tif = image.img_to_array(tf_pil_image_viirs_tif, data_format = None, dtype = None)
print(tf_np_array_viirs_tif.shape)
print(tf_np_array_viirs_tif[0][0][0].dtype)

RGB
(2632, 2477)
{'compression': 'tiff_lzw', 'dpi': (1, 1), 'resolution': (1, 1)}
(2477, 2632, 3)
float32

"""


"""
print('\n\nTF_Sentinel_tif')
tf_pil_image_sentinel_tif = image.load_img(path = "sentinel_least_cloudy_month_py_scale_1000.tif", color_mode='rgb')
print(tf_pil_image_sentinel_tif.mode)
print(tf_pil_image_sentinel_tif.size)
print(tf_pil_image_sentinel_tif.info)
#tf_pil_image_sentinel_tif.show()
tf_np_array_sentinel_tif = image.img_to_array(tf_pil_image_sentinel_tif, data_format = None, dtype = None)
print(tf_np_array_sentinel_tif.shape)

#PIL.UnidentifiedImageError: cannot identify image file 'sentinel_least_cloudy_month_py_scale_1000.tif'

"""







