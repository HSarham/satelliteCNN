from __future__ import absolute_import, division, print_function, unicode_literals

# Standard libraries genaral
import pandas
import sys
import datetime
import tensorflow
import os

from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# All pretrained networks
from tensorflow.keras import applications

# Local imports
import network_dataloader
import network_result_handling

def classify_images(
	network='mobilenetv2',
	training=True,
	Predefined_data_sets=None,
	weights='imagenet',
	input_mode='own',
	nr_epochs=1,
	image_sizes=(224, 224, 3),
	batch_size=100,
	learningrate=0.001,
	L2reg_scalar=0.1,
	filename_suffix='',
	image_directory="",
	base_directory=""):

	print('####################### Start Main function...')

	# General fixed parameters...
	optimizer = Adagrad(learning_rate=learningrate)
	# 1e-5 For fine tuning...
	optimizer_fine_tuning = Adagrad(learning_rate=0.00001)
	loss_function = 'mse'  
	
	finetune_epochs = 50
	
	metrics_list = ['mae']
	
	print('Epochs: ', nr_epochs, ' FT Epochs: ', finetune_epochs)
		
	label_column = 'Light_values_sum_log10_1_plus_value'
	
	start_time = pandas.to_datetime(datetime.datetime.now())
	datestring = datetime.datetime.now().strftime(
		"%Y-%m-%d %H:%M:%S").replace(' ', '--')
	datestring = datestring.replace(':', '-')
	print('Start-time: ', datestring)

	result_directory = 'Results2gpu/'
	try:
		os.mkdir(result_directory)
	except FileExistsError:
		print('Folder existed')
	
	# Server selection (Execution optimization...)
	myhost = os.uname()[1]
	print('Host: ', myhost)  # >Host:  bidaf.hh.se
	servername = myhost.split('.')
	servername = servername[0]

	file_suffix = servername + '_' + network + '_' + datestring + '_TZArs_Epochs_' + str(nr_epochs) + '_LR_' + str(
		learningrate).replace('.', '') + '_BS_' + str(batch_size) + '_L2reg_' + str(L2reg_scalar).replace('.', '') + '_' + filename_suffix
	resultfile = open(result_directory + "model_results_" +
					  file_suffix + ".txt", "w")

	version_full = tensorflow.__version__
	print('Tensorflow Version: ' + version_full)

	Processor = -1
	
	print('Server: ', servername)
	Processor = 6
	
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
		
	if Processor <= -1:
		print("creating network model using CPU " + str(Processor))
		os.environ['CUDA_VISIBLE_DEVICES'] = str(Processor)
		print('Using CPU')
	else:
		print('Using GPU: ', Processor)
		gpus = tensorflow.config.list_physical_devices('GPU')
		if len(gpus) < 2:
			print('ERROR: Number of GPU\'s only: ', len(gpus))
			print(gpus)
			sys.exit()
		print('Number of GPU\'s: ', len(gpus))
		print(gpus)
		
		tensorflow.config.set_visible_devices([gpus[4], gpus[5]], 'GPU')
		tensorflow.config.experimental.set_memory_growth(gpus[4], True)
		tensorflow.config.experimental.set_memory_growth(gpus[5], True)
		
	strategy = tensorflow.distribute.MirroredStrategy()
	with strategy.scope():
	
		
		print('Model is: ' + network)
		
		if training:
			base_model = applications.MobileNetV2(weights = weights, include_top = False, input_tensor = Input(shape = image_sizes))
		else:
			base_model = applications.MobileNetV2(weights = weights)
		
		if input_mode == 'own':
			preprocess_input = applications.mobilenet_v2.preprocess_input
		else:
			preprocess_input = imagenet_utils.preprocess_input
		
		print('Model is extended/modified...')
		
		
		# base_model.save_weights(result_directory + 'TZArs_original_weights.h5')
		
		base_model.trainable = False # Not an attribute in the API... :-(
		
		inputs = Input(shape=(224, 224, 3))
		base_model = base_model(inputs, training = False)
		
		# added_model = base_model.output # As input to function for added network...
		
		# added_model = Flatten(name = "added_flatten")(base_model)
		added_model = GlobalAveragePooling2D()(base_model) # Instead of flattening...
		
		added_model = Dense(1, activation = "linear", kernel_regularizer = l2(L2reg_scalar))(added_model) # Good! , kernel_regularizer = l2(L2reg_scalar)
		
		model = Model(inputs, added_model)
		
		model.compile(loss = loss_function, optimizer = optimizer, metrics = metrics_list)
		
		# model.save_weights(result_directory + 'TZArs_used_compiled_weights.h5')
			
		resultfile.write('Final model:\n\r')
		
		model.summary(print_fn = lambda x: resultfile.write(x + '\n\r#################\n\r'))
		
	print('############## Loading Metadataframe as model inputs')
	
	print('########## Training model')
	
	estimator_list = []
		
	# Predefined datasets, replace with list[0,1,2] base_directory + ...
	print('Predefined datasets!')
	
	# sys.exit()
	
	trainset, validationset, testset = network_dataloader.predefined_datasets ("", Predefined_data_sets)
	
	print('Final separate and reduced datasets shape before batch adjustment:')
	print(trainset.shape)
	print(validationset.shape)
	print(testset.shape)
	
		
	#################### Final indata adjustments ###################
	# Adjust datasets to batchsize (size % batch_size == 0) # Probably not necessary, but it does not hurt... :-)
	print('Before batch adjustment => Trainset: ', trainset.shape, ' Validationset: ', validationset.shape, ' Testset: ', testset.shape)
	print('Rest train...', trainset.shape[0] % batch_size)
	print('Rest val...', validationset.shape[0] % batch_size)
	print('Rest test...', testset.shape[0] % batch_size)
	
	trainset = trainset.drop(trainset.tail((trainset.shape[0] % batch_size)).index)
	validationset = validationset.drop(validationset.tail((validationset.shape[0] % batch_size)).index)
	testset = testset.drop(testset.tail((testset.shape[0] % batch_size)).index)
	
	print('After batch adjustment => Trainset: ', trainset.shape, ' Validationset: ', validationset.shape, ' Testset: ', testset.shape)
	print('Trainset head before training:\n\r', trainset.head())
	
	# Prints some training dataframe statistics to the resultfile.
	resultfile.write('\n\rTrainframe:\n\r')
	resultfile.write(network_result_handling.analyse_frame(trainset))
	resultfile.write('\n\rValidationframe:\n\r')
	resultfile.write(network_result_handling.analyse_frame(validationset))
	resultfile.write('\n\rTestframe:\n\r')
	resultfile.write(network_result_handling.analyse_frame(testset))
	
	print('Training data...')
	trainsize = trainset.shape[0]
	validationsize = validationset.shape[0]
	
	
	train_batch_generator = network_dataloader.generate_batches(image_directory, trainset, batch_size, preprocess_input, testphase = False, verbose = False, label_column = label_column)
	validate_batch_generator = network_dataloader.generate_batches(image_directory, validationset, batch_size, preprocess_input, testphase = False, verbose = False, label_column = label_column)
	
	print(type(train_batch_generator))
				
	print('### Starting Training...')
	
	
	dataset = tensorflow.data.Dataset.from_generator(lambda: map(tuple, train_batch_generator), output_signature=(tensorflow.TensorSpec(shape=(batch_size, 224, 224, 3), dtype=tensorflow.float32),tensorflow.TensorSpec(shape=(batch_size, ), dtype=tensorflow.float32)))
	options = tensorflow.data.Options()
	#options.experimental_distribute.auto_shard_policy = tensorflow.data.experimental.AutoShardPolicy.OFF
	trainset_fit = dataset.with_options(options)  # use this as input for your model
	dataset = tensorflow.data.Dataset.from_generator(lambda: map(tuple, validate_batch_generator), output_signature=(tensorflow.TensorSpec(shape=(batch_size, 224, 224, 3), dtype=tensorflow.float32),tensorflow.TensorSpec(shape=(batch_size, ), dtype=tensorflow.float32)))
	options = tensorflow.data.Options()
	#options.experimental_distribute.auto_shard_policy = tensorflow.data.experimental.AutoShardPolicy.OFF
	validationset_fit = dataset.with_options(options)  # use this as input for your model        
	
	
	callback_early_stopping = EarlyStopping(monitor = 'val_loss', patience = 2) # Patience 1-5 good...?
	# Improvement Steps set to None...? TBD Dependancy on even baches? TBD, Improvement of result params from estimator history, Now only loss_function...? TBD
	estimator = model.fit(trainset_fit,
				epochs = nr_epochs,
				# initial_epoch = batch * nr_epochs, # + 1, # nr_epochs may be inceased...?
				# batch_size = batch_size,
				shuffle = False, # Shuffle is already done if required...
				steps_per_epoch = trainsize // batch_size, # None?, trainsize // batch_size,
				validation_steps = validationsize // batch_size, # None? validationsize // batch_size,
				validation_data = validationset_fit,
				callbacks = [callback_early_stopping],
				verbose = 1) #,
				#callbacks=[TensorBoard(log_dir = output_log_dir, write_graph = True, write_images = False, histogram_freq = 0)]) #
	
	estimator_list.append(estimator.history)
	
	resultfile.write(str(estimator.history) + '\n\r')
	
	os.makedirs(base_directory + '/models/', exist_ok=True)

	model.save(base_directory + '/models/' + file_suffix + '.tf') # h5 in TF 1.x
	# model.save_weights(result_directory + 'TZArs_used_compiled_weights_after_training.h5')
	
	# sys.exit()
	print('### Finished Training...')
	
	# Testing on training data
	
	# Result handling
	print('Testing on trainingdata ###############')
	
	
	# Test on trained data
	train_batch_generator_test = network_dataloader.generate_batches(image_directory, trainset, batch_size, preprocess_input, testphase = True, verbose = False, label_column = label_column)
	
	prediction_results_train = model.predict(train_batch_generator_test, verbose = 1, steps = trainsize // batch_size) # steps = None goes infinite... Bad...
	print('Finished prediction_results_train...')
	
	# Train output results
	train_ground_truth = list(trainset.pop(label_column))
	label_list = train_ground_truth.copy()
	trainset[label_column] = label_list
	
	# Scatterplot
	network_result_handling.plot_regression_scatter (prediction_results_train, train_ground_truth, 'Train True values versus Predicted values', file_suffix, result_directory = result_directory)
	
	# Test on validation data
	print('Testing on validationdata ###############')
	
	
	validation_batch_generator_test = network_dataloader.generate_batches(image_directory, validationset, batch_size, preprocess_input, testphase = True, verbose = False, label_column = label_column)
	
	prediction_results_validation = model.predict(validation_batch_generator_test, verbose = 1, steps = validationsize // batch_size) # steps = None goes infinite... Bad...
	
	validation_ground_truth = list(validationset.pop(label_column))
	label_list = validation_ground_truth.copy()
	validationset[label_column] = label_list
	
	# Scatterplot
	network_result_handling.plot_regression_scatter (prediction_results_validation, validation_ground_truth, 'Validation True values versus Predicted values', file_suffix, result_directory = result_directory)
	
	file_suffix = 'Fine_tuning_' + file_suffix
	
	model.trainable = True
	model.compile(loss = loss_function, optimizer = optimizer_fine_tuning, metrics = metrics_list)
	
	
	
	train_batch_generator = network_dataloader.generate_batches(image_directory, trainset, batch_size, preprocess_input, testphase = False, verbose = False, label_column = label_column)
	validate_batch_generator = network_dataloader.generate_batches(image_directory, validationset, batch_size, preprocess_input, testphase = False, verbose = False, label_column = label_column)
	test_batch_generator = network_dataloader.generate_batches(image_directory, testset, batch_size, preprocess_input, testphase = True, verbose = False, label_column = label_column)
	
	dataset = tensorflow.data.Dataset.from_generator(lambda: map(tuple, train_batch_generator), output_signature=(tensorflow.TensorSpec(shape=(batch_size, 224, 224, 3), dtype=tensorflow.float32),tensorflow.TensorSpec(shape=(batch_size, ), dtype=tensorflow.float32)))
	options = tensorflow.data.Options()
	#options.experimental_distribute.auto_shard_policy = tensorflow.data.experimental.AutoShardPolicy.OFF
	trainset_fit = dataset.with_options(options)  # use this as input for your model
	dataset = tensorflow.data.Dataset.from_generator(lambda: map(tuple, validate_batch_generator), output_signature=(tensorflow.TensorSpec(shape=(batch_size, 224, 224, 3), dtype=tensorflow.float32),tensorflow.TensorSpec(shape=(batch_size, ), dtype=tensorflow.float32)))
	options = tensorflow.data.Options()
	#options.experimental_distribute.auto_shard_policy = tensorflow.data.experimental.AutoShardPolicy.OFF
	validationset_fit = dataset.with_options(options)  # use this as input for your model        
	
	print('### Starting Training Fine tuning...')
	# Improvement Steps set to None...? TBD Dependancy on even baches? TBD, Improvement of result params from estimator history, Now only loss_function...? TBD
	estimator = model.fit(trainset_fit,
				epochs = finetune_epochs,
				# initial_epoch = batch * nr_epochs, # + 1, # nr_epochs may be inceased...?
				# batch_size = batch_size,
				shuffle = False, # Shuffle is already done if required...
				steps_per_epoch = trainsize // batch_size, # None?, trainsize // batch_size,
				validation_steps = validationsize // batch_size, # None? validationsize // batch_size,
				validation_data = validationset_fit,
				callbacks = [callback_early_stopping],
				verbose = 1) #,
				#callbacks=[TensorBoard(log_dir = output_log_dir, write_graph = True, write_images = False, histogram_freq = 0)]) #
	
	estimator_list.append(estimator.history)
	
	resultfile.write(str(estimator.history) + '\n\r')
	
	model.save(base_directory + '/models/TZA_Rect_complete_plus_TZA_surr_FT_CNN_regression_logvalues_' + file_suffix + '.tf') # h5 in TF 1.x
	# model.save_weights(result_directory + 'TZArs_used_compiled_weights_after_training.h5')
	
	
	print('### Finished Training FT...')
	
	# Testing Fine tuned training
	
	# Result handling
	print('Testing on trainingdata FT ###############')
		
	# Test on trained data
	print('Trainsize: ', trainsize)
	train_batch_generator_test = network_dataloader.generate_batches(image_directory, trainset, batch_size, preprocess_input, testphase = True, verbose = False, label_column = label_column)
	prediction_results_train = model.predict(train_batch_generator_test, verbose = 1, steps = trainsize // batch_size) # steps = None goes infinite... Bad...
	print('Finished prediction_results_train...')
	
	# Train output results
	train_ground_truth = list(trainset.pop(label_column))

	# Scatterplot
	network_result_handling.plot_regression_scatter (prediction_results_train, train_ground_truth, 'FT Train True values versus Predicted values', file_suffix, result_directory = result_directory)
	
	# Test on validation data
	print('Testing on validationdata FT ###############')
	validation_batch_generator_test = network_dataloader.generate_batches(image_directory, validationset, batch_size, preprocess_input, testphase = True, verbose = False, label_column = label_column)
	prediction_results_validation = model.predict(validation_batch_generator_test, verbose = 1, steps = validationsize // batch_size) # steps = None goes infinite... Bad...
	
	validation_ground_truth = list(validationset.pop(label_column))
	
	print('Prediction-results validation shape: ', prediction_results_validation.shape)
	print('True results original validation shape: ', len(validation_ground_truth))
	print(prediction_results_validation)
	
	# Scatterplot
	network_result_handling.plot_regression_scatter (prediction_results_validation, validation_ground_truth, 'FT Validation True values versus Predicted values', file_suffix, result_directory = result_directory)
	
	# Training and validation curves (loss_function and accuracy)
	network_result_handling.plot_training_statistics (estimator_list, 'NTR_FTTR' + file_suffix, metrics = metrics_list, loss_function = loss_function, result_directory = result_directory)
	

	# Summarize...

	stop_time = pandas.to_datetime(datetime.datetime.now())
	timedelta_days = pandas.Timedelta(stop_time - start_time).days
	timedelta_minutes = pandas.Timedelta(stop_time - start_time).seconds / 60.0
	
	
	resultfile.write('\n\rExecution time in minutes: ' + str(timedelta_days * 1440 + timedelta_minutes))
	resultfile.close()
	
	
	datestring = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(' ', '--')
	datestring = datestring.replace(':', '-')
	print('Stop-time: ', datestring)


##################################### Main ########################################

def main(argv):
	
	curr_dir = __file__[:__file__.rfind("/")]

	base_directory = curr_dir[:curr_dir.rfind("/")]
	image_directory = curr_dir+"../image_data/"

	trainset_file_list = [curr_dir+'/p25/TZA_rect_complete_TZA_surr_light_train_2023_09_lyon_mobilenetv2_2023-09-05--13-59-26_TZArs_Epochs_1_LR_0001_BS_100_L2reg_01_p25_TZA_rs_.parquet',
	curr_dir+'/p25/TZA_rect_complete_TZA_surr_light_validation_2023_09_lyon_mobilenetv2_2023-09-05--13-59-26_TZArs_Epochs_1_LR_0001_BS_100_L2reg_01_p25_TZA_rs_.parquet',
	curr_dir+'/p25/TZA_rect_complete_TZA_surr_light_test_2023_09_lyon_mobilenetv2_2023-09-05--13-59-26_TZArs_Epochs_1_LR_0001_BS_100_L2reg_01_p25_TZA_rs_.parquet']

	filename_suffix = 'p25_TZA_rs_' # '3x3_FTTR_FT_LR001_GAP_D0_TZA_Suuroundings_400k_labels_5_50'
		
	# Rewrite according to chunks of arguments depending on main function "decisions"?
	classify_images (
		network = 'mobilenetv2', # 'mobilenetv2' ... ('own') implies that Own_trained_model is defined with at least own output-layer...
		training = True, # For training True, for test False
		Predefined_data_sets = trainset_file_list, # trainset_file_list, #testset_file_list, # Change to list of sets (train, validate, test) or only test... or None
		weights = 'imagenet',
		input_mode = 'own', # Networks own method...
		nr_epochs = 50, # Only valid for Training mode...
		image_sizes = (224, 224, 3), # Only valid for Training mode...
		batch_size = 100, # 1 for test only to get maximum input for regression...
		learningrate = 0.01, # Only valid for Training mode...
		L2reg_scalar = 0.1, # Only valid for Training mode...
		filename_suffix = filename_suffix,
		image_directory = image_directory,
		base_directory=base_directory)

if __name__ == '__main__': main(argv = '')







