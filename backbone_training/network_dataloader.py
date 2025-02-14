# Python general
import datetime
import math
import pandas
import numpy
import sys


# Python specifics
from PIL import Image
Image.MAX_IMAGE_PIXELS = 999999999 # Might have to increased, Limit unknown...
from sklearn.model_selection import train_test_split
from os import listdir
from os.path import isfile, join


# Own imports
import image_library_functions



############## START library functions ##############

#########################################
"""
Description:
Function for loading a combined data and label dataframe

Input variables/types and functions:
image_directory:		The directory holding all image-files to be used (not more not less), can have subdirectories which will be ignored.
include_areas:			The directory names on first level of image directory for images to be included...
matrix_code_separator:	A character/string that separates matrix_code from the rest of the filename (matrix_code + matrix_code_separator + rest_of_filename)
						Matrix code in format as: row_column, I.e. 1_42
randomized_result:		Wether the output dataframe shall be in randomized order or not
labelsfile:				A complete path to the labelsfile (.csv) for data-label info composition (corresponding to include_areas (improvemt? TBD...)). 
						A csv file with precalculated labeldata corresponding to files in image_directory
labels_columns:			Holding at least columns as: ['Light_labels', 'Matrix_code', 'Label_code'] who must be defined in the labelsfile
verbose:				Set to 1 if debug printouts are wanted
debug:					
dhs_areas:				


Output (data and types):
result_df:				A pandas dataframe where image meta-data and corresponding label-data is merged. Holding: 'Image_File_Name', 'Image_Matrix_Code'
						plus given labels_columns where each colmn-name gets a prefix of: Label_ + column_name in labels_columns

Constraints:
N/A

Exceptions:
N/A

Others:
Can be improved/extended with other data and labels definitions...

"""

def load_images_and_label_info(base_image_directory, include_areas = [], matrix_code_separator = '#', randomized_result = False, labelsfile = '', labels_columns = [], verbose = 0, debug = False, dhs_areas = False):
	
	print('Entering: load_images_and_label_info')
	
	directories, areanames = image_library_functions.find_area_directories(base_image_directory, datadirectories = [], areanames = []) # The base directory on top of first level image directories...
	
	"""
	print('##### after find_area_directories')
	print('Directories: ', directories)
	print('Nr of areas: ', len(include_areas))
	print('Areas :', include_areas)
	sys.exit()
	"""
	
	result_df = pandas.DataFrame()
	labelinfo_df = pandas.DataFrame()
	
	for index in range(len(directories)):
	
		result_area_df = pandas.DataFrame()
		filenames_list = []
		df_filenames_list = []
		matrix_code_list = []
		area_code_list = []
		country_code_list = []
		
		if not areanames[index] in include_areas:
			continue
		
		image_directory = directories[index]
		
		for item in listdir(image_directory):
			if isfile(join(image_directory, item)):
				filenames_list.append(areanames[index] + '/' + item)
		
		print('Number of files: ', len(filenames_list), ' In area: ', areanames[index])
		
		for imagefile_name in filenames_list: # May Include area_code...
			
			dir, filename = imagefile_name.split('/')
			filename_code, rest = filename.split(matrix_code_separator)
			filename_parts = filename_code.split('_')
			
			# For TZA_Surroundings...
			if dhs_areas:
				df_filenames_list.append('TZA_Surroundings/' + imagefile_name)
				matrix_code_list.append(filename_parts[-2] + '_' + filename_parts[-1])
				country_code_list.append(areanames[index])
					
				code_parts = rest.split('-_') # rest is a string
				
				dhs_area_code = code_parts[0].split('gdal_to_pil_rgb_uint8_min_191_maxscale_1714_') # Ugly, but implies a certain image conversion type (min_191_maxscale_1714)...
				
				area_code_list.append(dhs_area_code[1])
				
			else: # TZA rectangle
				if len(filename_parts) == 2: # No area code
					df_filenames_list.append(imagefile_name)
					matrix_code_list.append(filename_parts[0] + '_' + filename_parts[1])
					area_code_list.append(areanames[index])
					country_code_list.append('TZA_Rectangle')
					
				if len(filename_parts) == 3: # Areas (Area code included)
					df_filenames_list.append(imagefile_name)
					matrix_code_list.append(filename_parts[1] + '_' + filename_parts[2])
					area_code_list.append(areanames[index])
					country_code_list.append('11_Cities')
			
			
		result_area_df['Image_File_Name'] = df_filenames_list
		result_area_df['Image_Matrix_Code'] = matrix_code_list
		result_area_df['Image_Area_Code'] = area_code_list
		result_area_df['Image_Country_Code'] = country_code_list
		
		
		result_df = pandas.concat([result_df, result_area_df], ignore_index = True)
		
	
	result_df = result_df.sort_values(['Image_Area_Code', 'Image_Matrix_Code'], ascending = [True, True])
	
	labelinfo_df = pandas.read_csv(labelsfile, sep = ";", index_col = False)
	labelinfo_df = labelinfo_df.sort_values(['Label_Area_code', 'Label_Matrix_code'], ascending = [True, True]) # 
	
	print('images shape: ', result_df.shape, ' Labels shape: ', labelinfo_df.shape)
	if result_df.shape[0] != labelinfo_df.shape[0]:
		print('ERROR incompatible shapes images and labels')
		sys.exit()
	
	"""
	print('Label sorted shape: ', labelinfo_df.shape)
	print(labelinfo_df.head())
	print('Labels columns names: ', labels_columns, ' Numbers: ', len(labels_columns))
	"""
	
	for column_name in labels_columns: # Merge label info with data info
		
		column = list(labelinfo_df.pop(column_name))
		result_df[column_name] = column
	
	
	if randomized_result:
		result_df = result_df.reindex(numpy.random.permutation(result_df.index))
	
	if verbose == 1:
		print('\n\nDF total shape, load_images_and_label_info: ', result_df.shape)
		print(result_df.head())
	
	return result_df

"""
Description:
This function is a "generator" to reply with data/(labels) for a call from Tensorflow.keras "fit" and/or "predict" function to feed a Neural network...


Input variables/types and functions:
data_directory:			The base directory holding excatly all data/image files, NOT more, NO less in given in meta_dataframe (subdirectory/filenames)
						With appropriate size for used network and readable as a PIL-image, RGB (0 .. 255).
meta_dataframe:			A pandas dataframe holding at least columns for: 'Image_File_Name' and 'Label_Label_code'
batch_size:				The batchsize to be used for NN input.
preprocess_input_fn:	The data preprocess function to be used for data-preprocessing (may be network specific)
testphase:				If True, only data is returnes, otherwise data and labels (ground truth)
verbose:				If True some debug printouts are done during execution.

Output (data and types):
Since this is a generator output will be done according to call and depends on given batch_size and setting of testphase

Constraints:
batch_size: 			>= 1 (Good if complete nr of samples in meta_dataframe modulo batch_size == 0)
preprocess_input_fn:	Should be some that handles indata with shape: (x, y, z, 3) => x samples of images

Exceptions:
N/A

Others:
Note: Ground truth for testphase needs to ba handled outside this function.
Observe: Index in given dataframe must be qnique!

"""

def generate_batches (data_directory, meta_dataframe, batch_size, preprocess_input_fn, testphase = False, verbose = False, label_column = 'Label_code'):
	

	while True: # Only to make the generator infinite, TF.keras needs that
	
		number_of_samples = meta_dataframe.shape[0]
		indexlist = list(meta_dataframe.index)
		
		if verbose:
			print('generate_batches: meta_dataframe size: ', number_of_samples)
			print('generate_batches: Index list length: ', len(indexlist))
	
		for batch in range(0, number_of_samples, batch_size):

			# slice out the current batch according to batch-size
			batch_index_list = indexlist[batch : (batch + batch_size)]
			meta_batch_dataframe = meta_dataframe.loc[batch_index_list]
			
			if verbose:
				print('generate_batches: Batch: ', batch, ' Length indexlist: ', len(batch_index_list))
				print('generate_batches: Batch index list:', batch_index_list)
				print('generate_batches: Meta batch datafarme: ')
				print(meta_batch_dataframe.head(meta_batch_dataframe.shape[0]))

            # initializing the arrays, data and label values
			batch_data_samples = []
			batch_labels = []

			for index, row in meta_batch_dataframe.iterrows():
				
				# get imagedata sample and append it
				batch_data_samples.append(list(numpy.array(Image.open(data_directory + row['Image_File_Name']))))
				# get label and append it
				batch_labels.append(row[label_column])
			
			if len(batch_data_samples) < batch_size:
				print('************* Real batch length less than batchsize returned: ', len(batch_data_samples))
			batch_data_samples = preprocess_input_fn(numpy.array(batch_data_samples))
			batch_lables = numpy.array(batch_labels)
			
			if testphase:
				yield (batch_data_samples)
			else:
				yield (batch_data_samples, batch_lables)
			
			#yield (batch_data_samples, batch_lables)
			
			"""
			The yield statement suspends function’s execution and sends a value back to the caller, 
			but retains enough state to enable function to resume where it is left off. When resumed, 
			the function continues execution immediately after the last yield run. 
			This allows its code to produce a series of values over time, rather than computing them at once and sending them back like a list.
			"""

#########################################

"""
Description:
This function is a "generator" to reply with data/(labels) for a call from Tensorflow.keras "fit" and/or "predict" function to feed a Neural network...


Input variables/types and functions:
data_directory:			The base directory holding excatly all data/image files, NOT more, NO less in given in meta_dataframe (subdirectory/filenames)
						With appropriate size for used network and readable as a PIL-image, RGB (0 .. 255).
meta_dataframe:			A pandas dataframe holding at least columns for: 'Image_File_Name' and 'Label_Label_code'
batch_size:				The batchsize to be used for NN input.
TBD



testphase:				If True, only data is returnes, otherwise data and labels (ground truth)
verbose:				If True some debug printouts are done during execution.

Output (data and types):
Since this is a generator output will be done according to call and depends on given batch_size and setting of testphase

Constraints:
batch_size: 			>= 1 (Good if complete nr of samples in meta_dataframe modulo batch_size == 0)

Exceptions:
N/A

Others:
Note: Ground truth for testphase needs to ba handled outside this function.
Observe: Index in given dataframe must be qnique!

"""

def generate_batches_regression (meta_dataframe, batch_size, data_column, label_column, testphase = False, verbose = False):

	while True: # Only to make the generator infinite, TF.keras needs that
	
		number_of_samples = meta_dataframe.shape[0]
		indexlist = list(meta_dataframe.index)
		
		if verbose:
			print('generate_batches: meta_dataframe size: ', number_of_samples)
			print('generate_batches: Index list length: ', len(indexlist))
	
		for batch in range(0, number_of_samples, batch_size):

			# slice out the current batch according to batch-size
			batch_index_list = indexlist[batch : (batch + batch_size)]
			meta_batch_dataframe = meta_dataframe.loc[batch_index_list]
			
			if verbose:
				print('generate_batches: Batch: ', batch, ' Length indexlist: ', len(batch_index_list))
				print('generate_batches: Batch index list:', batch_index_list)
				print('generate_batches: Meta batch datafarme: ')
				print(meta_batch_dataframe.head(meta_batch_dataframe.shape[0]))

            # initializing the arrays, data and label values
			batch_data_samples = []
			batch_labels = []

			for index, row in meta_batch_dataframe.iterrows():
				
				# Get data and labels...
				input_value_list = row[data_column].replace('[', "").replace(']', "").replace(" ", "").split(',') # list of float strings
				input_float_values = []
				for value in input_value_list:
					input_float_values.append(float(value))
				batch_data_samples.append(input_float_values)
				batch_labels.append(float(row[label_column]))
				
			if len(batch_data_samples) < batch_size:
				print('************* Real batch length less than batchsize returned: ', len(batch_data_samples))
			
			batch_data_samples = numpy.array(batch_data_samples)
			batch_lables = numpy.array(batch_labels)
			
			# return numpy arrays...
			if testphase:
				yield (batch_data_samples)
			else:
				yield (batch_data_samples, batch_lables)
			
			"""
			The yield statement suspends function’s execution and sends a value back to the caller, 
			but retains enough state to enable function to resume where it is left off. When resumed, 
			the function continues execution immediately after the last yield run. 
			This allows its code to produce a series of values over time, rather than computing them at once and sending them back like a list.
			"""


#########################################



"""
Description:
With given arguments this function composes imagedata and labeldata returned as two numpy.arrays

Input variables/types and functions:
image_directory:	The directory holding images, NOT more NO less.
data_label_df:		A dataframe holding image filenames (only filenames) and corresponding label_value
filename_label:		The key do dataframes filename
labelvalue_label:	The key to dataframes label-value


Output (data and types):
Two numpy.arrays of data and corresponding label-values

Constraints:
N/A

Exceptions:
N/A

Others:
N/A

"""

def get_data_and_labels(image_directory, data_label_df, filename_label = '', labelvalue_label = ''):

	data_list = []
	data_pds = data_label_df.pop(filename_label)
	label_pds = data_label_df.pop(labelvalue_label)
		
	for filename in data_pds:
		data_list.append(list(numpy.array(Image.open(image_directory + filename))))
	
	return numpy.array(data_list), numpy.array(label_pds)


# Test


#########################################

"""
Description:
Splits the given dataframe in mutually exclusive desired parts

Input:
dataframe:			A dataframe with data to split in equal pices according to splits argument
dataframe_list:		Normally an empty list which will hold the function return
splits:				Number of splits of the given datafarme	

Output:
A list of dataframes splitted in equally desired sizes

Constraints:
N/A

Exceptions:
N/A

Others:

"""		
def get_splitted_frames(dataframe, dataframe_list = [], splits = 1):
	
	if splits == 1:
		dataframe_list.append(dataframe)
	else:
		trainframe, restframe = train_test_split(dataframe, train_size = (1 / splits), shuffle = False)
		dataframe_list.append(trainframe)
		get_splitted_frames(restframe, dataframe_list, splits - 1)

	return 	dataframe_list
		
#########################################
		
"""
Description:
Given a list of dataframes and an index choosen for validation it returns the complete set for training where the set for validation is excluded


Input:
foldframe_list:		A list of dataframes (Can be from function: get_splitted_frames)
validationindex:	To be the validationframe selected from foldframe_list

Output:
Two pandas dataframes: trainframe and a validationframe

Constraints:
N/A

Exceptions:
N/A

Others:
Note: foldframe_list should contain frames (data and labels) which probably needs to be extracted after return of this function

"""
def getFoldTrainFrames(foldframe_list, validationindex):
	
	foldtrainframe = pandas.DataFrame()
	foldvalidationframe = pandas.DataFrame()
	
	for index in range(len(foldframe_list)):
		if index == validationindex:
			foldvalidationframe = foldvalidationframe.append(foldframe_list[index])
		else:
			foldtrainframe = foldtrainframe.append(foldframe_list[index])
		
	return foldtrainframe, foldvalidationframe

	
#########################################

"""
Description:
The function splits a dataframe into three parts

Input variables/types and functions:
meta_dataframe:		The dataframe to split
first_rows_to_test:	The number of first rows that goes to testset
train_size:			The size in percent of the rest to trainset

Output (data and types):
trainset, validationset, testset

Constraints:
N/A

Exceptions:
N/A

Others:
N/A

"""

def metadata_with_specific_test_rows(meta_dataframe, first_rows_to_test = 100, train_size = 0.8):
	
	
	test_list = []
	for index, row in meta_dataframe.iterrows():
		image_row, image_column =  row['Image_Matrix_Code'].split('_')
		image_row = int(image_row)
		if image_row <= first_rows_to_test:
			test_list.append(index)
		
	testset = meta_dataframe.loc[test_list]
	meta_dataframe = meta_dataframe.drop(test_list)
	trainset, validationset = train_test_split(meta_dataframe, train_size = train_size, shuffle = False)
	
	return trainset, validationset, testset
	
	

#########################################

"""
Description:
This function separates an input dataframe into three separate mutually exclusive dataframes.


Input variables/types and functions:
meta_dataframe:		A complete frame for downselections according to given arguments
times:				Specifically labels like 'No_light' will be extracted times * the amount of complimentary string-labels
trainsize:			The amount/size of returned trainset from meta_dataframe (validation and test set will be in same size from the complimentary set for training)

Output (data and types):
Three pandas dataframes with meta-data for: train, validation and test

Constraints:
Assumes a dataframe with column: 'Label_Light_labels' where at least values: 'Medium_light' and 'Bright_light' must be defined (Best is if third value is 'No_light')

Exceptions:
N/A

Others:
N/A

"""

def image_info_balanced (meta_dataframe, times = 1, trainsize = 0.8, shuffle = True):
	
	delete_samples = []
	times_counter = 0
	light_nr = 0
	limit = 0
	
	for index, row in meta_dataframe.iterrows():
		if row['Label_Light_labels'] == 'Medium_light' or row['Label_Light_labels'] == 'Bright_light':
			light_nr  += 1
	
	limit = light_nr * times
	
	for index, row in meta_dataframe.iterrows():
			
		if row['Label_Light_labels'] == 'Medium_light' or row['Label_Light_labels'] == 'Bright_light':
			None
		else:
			if times_counter > limit:
				delete_samples.append(index)
			times_counter += 1
			
			
	meta_dataframe = meta_dataframe.drop(delete_samples)
	
	# split: validation and test same size...
	trainset, rest = train_test_split(meta_dataframe, train_size = trainsize, shuffle = shuffle)
	validationset, testset = train_test_split(rest, train_size = 0.5, shuffle = shuffle)
			
	return trainset, validationset, testset	
		
	
#########################################

"""
Description:
Filter out unwanted labels from meta_dataframe

Input variables/types and functions:
meta_dataframe:		To do exclusions from
exclude_label_list:	Defining what shall be filtered out

Output (data and types):
One pandas dataframe

Constraints:
N/A

Exceptions:
N/A

Others:

"""

def exclude_labels (meta_dataframe, exclude_label_list):

	print('Exclude dataframe in shape: ', meta_dataframe.shape)
	print('Exclude labels list:', exclude_label_list)

	delete_samples = []
	for index, row in meta_dataframe.iterrows():
		for label in exclude_label_list:
			if row['Label_Light_labels'] == label:
				delete_samples.append(index)
	
	print('deleted samples: ', len(delete_samples))
	
	meta_dataframe = meta_dataframe.drop(index = delete_samples)
	print('Exclude dataframe out shape: ', meta_dataframe.shape)

	return meta_dataframe



#########################################

"""
Description:


Input variables/types and functions:
server_base_directory:	The base directory for the datasetfiles
datasetfiles: 			A list of file(s), 3 or 1 depending on mode (train, validate, test or only test), complete path on selected server

Output (data and types):
Three pandas dataframes: trainset, validationset, testset or only testset

Constraints:
N/A

Exceptions:
N/A

Others:
N/A

"""

def predefined_datasets (server_base_directory, dataset_files):

	"""
	print(server_base_directory)
	print(dataset_files)
	print(len(dataset_files))
	print(server_base_directory + dataset_files[0])
	print('Start reading...')
	"""

	if len(dataset_files) == 3:
		trainset = pandas.read_parquet(server_base_directory + dataset_files[0])
		validationset = pandas.read_parquet(server_base_directory + dataset_files[1])
		testset = pandas.read_parquet(server_base_directory + dataset_files[2])
		# trainset = pandas.read_csv(server_base_directory + dataset_files[0], sep = ";", index_col = False)
		# validationset = pandas.read_csv(server_base_directory + dataset_files[1], sep = ";", index_col = False)
		# testset = pandas.read_csv(server_base_directory + dataset_files[2], sep = ";", index_col = False)
		return trainset, validationset, testset
		
	else: # Only test (train ?)	
		testset = pandas.read_csv(server_base_directory + dataset_files[0], sep = ";", index_col = False)
		return testset


#########################################

"""
Description:
Given a dataframe it is splitted in three parts according to additional arguments

Input variables/types and functions:
meta_dataframe:	The dataframe to do three mutually exclusive selections from
train_size:		The amount of (0.8 == 80%) samples/rows to be selected for trainset; (the complimentary set (the rest) will be equally (50/50) distributed for validationset and testset)

Output (data and types):
Three pandas dataframes: trainset, validationset, testset

Constraints:
N/A

Exceptions:
N/A

Others:

"""

def fixed_selection(meta_dataframe, train_size = 0.8):
	
	trainset, rest = train_test_split(meta_dataframe, train_size = train_size, shuffle = False)
	validationset, testset = train_test_split(rest, train_size = 0.5, shuffle = False)
	
	return trainset, validationset, testset
	
	
#########################################



"""
Description:



Input variables/types and functions:
meta_dataframe:				Inputframe to be filtered from clouds
data_directory:				Base directory for filepaths
pixel_filter_limit = 735.0:	Assumes 3D pixel images, column summary over this limit
cloud_limit = 0.25:			How many percent cloudy pixels for deletion of image

Output (data and types):
Dataframe filtered from clody images

Constraints:
N/A

Exceptions:
N/A

Others:
N/A

"""
	
def cloud_filtering(meta_dataframe, data_directory, pixel_filter_limit = 735.0, cloud_limit = 0.25):

	cloudy_index = []
	non_cloudy_index = []
	nr_pixel_values = 0
	nr_cloudy_pixels = 0
	nr_of_images = 0

	# TBD Add functionality to save with matrix_code and cloud_flag to csv file...?
	for index, row in meta_dataframe.iterrows():
		
		nr_of_images += 1
		#print('Image...: ', nr_of_images)

		np_pil_image = numpy.array(Image.open(data_directory + row['Image_File_Name']))
		
		nr_pixel_values = 0
		nr_cloudy_pixels = 0
		for image_row in np_pil_image:
		
			for image_column in image_row:
				
				pixel_sum = image_column.sum()
				
				nr_pixel_values += 1
				if pixel_sum >= pixel_filter_limit:
					nr_cloudy_pixels += 1
				
		if (nr_cloudy_pixels / nr_pixel_values) >= cloud_limit:
			cloudy_index.append(index)
		else:
			non_cloudy_index.append(index)
	
	print('Total:', len(cloudy_index) + len(non_cloudy_index))
	print('Cloudy nr: ', len(cloudy_index))
	print('Non cloudy nr: ', len(non_cloudy_index))
	
	meta_dataframe = meta_dataframe.drop(cloudy_index)
	
	return meta_dataframe
	

############## END library functions ##############














