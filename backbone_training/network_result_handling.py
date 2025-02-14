

import datetime
import math
import pandas
import numpy
from os import listdir
from os.path import isfile, join
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, r2_score
import itertools



############## START library functions ##############

#########################################
"""
Description:
This function analyses the amount of samples for each (fixed coded labels: 'No_light', 'Medium_light' and 'Bright_light')

Input variables/types and functions:
dataframe:	


Output (data and types):
resultstring:	The total number of samples plus the number of label-samples respectively

Constraints:
Non empty dataframe with at least the column-name 'Label_Light_labels' as string values: 'No_light', 'Medium_light' and 'Bright_light'

Exceptions:
N/A

Others:
Improvements might be some generalization

"""

def analyse_frame(dataframe):	
	
	resultstring = ''
	nr_of_samples = 0
	nr_of_no_light = 0
	nr_of_medium_light = 0
	nr_of_bright_light = 0
	for index, row in dataframe.iterrows():
		
		if row['Label_Light_labels'] == 'No_light':
			nr_of_no_light += 1
		if row['Label_Light_labels'] == 'Medium_light':
			nr_of_medium_light += 1
		if row['Label_Light_labels'] == 'Bright_light':
			nr_of_bright_light += 1
	
		nr_of_samples += 1
	
	resultstring += '\n\rNr of samples: ' + str(nr_of_samples)
	resultstring += '\n\rNo Light: ' + str(nr_of_no_light)
	resultstring += '\n\rMedium Light: ' + str(nr_of_medium_light)
	resultstring += '\n\rBright light: ' + str(nr_of_bright_light)
	resultstring += '\n\r'
	
	return resultstring

def print_cm_values(confusion_matrix, labelstrings, filesuffix, plot_header = '', result_directory = 'Results/'):	#, plot_header
	
	classesy = labelstrings
	classesx = labelstrings

	cm_title = 'Confusion Matrix ' + plot_header
	print(cm_title)

	plt.figure()
	#print('Came to 1')
	plt.imshow(confusion_matrix, cmap=plt.cm.Blues) # origin='lower' interpolation='nearest'
	#print('Came to 2')
	plt.colorbar()
	plt.title(cm_title) # + plot_header
	tick_marks = numpy.arange(len(labelstrings))
	plt.xticks(tick_marks, classesx, rotation = 45)
	plt.yticks(tick_marks, classesy)
	thresh = confusion_matrix.max() / 2
	for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
		plt.text(j, i, format(confusion_matrix[i, j], 'd'), horizontalalignment="center", color="white" if confusion_matrix[i, j] > thresh else "black")
	plt.tight_layout()
	plt.xlabel('Predicted')
	plt.ylabel('True')
	plt.savefig(result_directory + 'Confusion_matrix_values_' + plot_header + '_' + filesuffix + '.png')
	plt.clf()	


#########################################
"""
Description:
TBD


Input:
y_true_list:	
y_prob_list:	
filesuffix:	

Output:
A list of AUC-values

Side-effects:	Saving of a plot-file as 'Results/ROC_curve_' + filesuffix

Others:
Fileformat is automatically given as ".png"
TBD: Improvement to handle multiple´curves with separated legends...

"""	
def print_roc_curve(y_true_list, y_prob_list, filesuffix, result_directory = 'Results/'):

	roc_auc_list = []
	
	for label in range(len(y_true_list)):

		fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=1)
		roc_auc = auc(fpr, tpr)
		roc_auc_list.append(roc_auc)
		
		plt.figure()
		plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve unhealthy (area = %0.2f)' % roc_auc)
		plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver operating characteristic')
		plt.legend(loc="lower right")
	
	
	
	plt.savefig(result_directory + 'ROC_curve_' + filesuffix + '.png')
	plt.clf()
	
	return roc_auc_list


#########################################
"""
Description: 
This function creates and save a histogram-plot for given probability values (float)

Input:
probabilities:		The values to plot as a Pandas.Series (to be changed later to a list...)
file_suffix:		Appended to hardcoded prefix for saving in relative execution directory: "Results/Probabilities_" + file_suffix

Output:
None

Side-effects:	Saving of a plot-file as .png

Others:
Fileformat is automatically given as ".png"
Number of bins is fixed to: 40

"""
def print_probabilities(probabilities, file_suffix, plot_header = 'Testdata', result_directory = 'Results/'):

	probabilities.plot(kind = 'hist', bins = 40)
	#x1,x2,y1,y2 = plt.axis()
	#plt.axis((x1,x2,y1,400))
	plt.title("Probability values_" + plot_header)
	plt.grid(True)
	plt.savefig(result_directory + "Probabilities_" + plot_header + '_' + file_suffix + ".png")
	plt.clf()	
	

#########################################
"""
Description:
Note:
This function creates two plots for training loss and accuracy respectively.

Input variables/types and functions:
history:		A dict structure from Tensorflow with keys: 'accuracy', 'val_accuracy', 'loss' and 'val_loss'
file_suffix:	Is appended to hardcoded: "Results/Loss_"

Output (data and types):
None

Side-effext:	Saving of two individual plots för loss and accuracy

Constraints:
N/A

Exceptions:
N/A

Others:
Fileformat is automatically given as ".png"
Can be improved with more flexibility... (TBD)

"""

def plot_training_statistics_regression (history, file_suffix = '', metrics = ['accuracy'], loss_function = 'mse', result_directory = 'Results/'):

	loss = history['loss']
	val_loss = history['val_loss']
	epochs = range(len(loss))
	
	m1 = history[metrics[0]]
	m1_val = history['val_' + metrics[0]]
	m2 = history[metrics[1]]
	m2_val = history['val_' + metrics[1]]
	
	plt.title('Training and validation metrics')
	plt.xlabel('Epochs')
	plt.ylabel('Value')
	r = plt.plot(epochs, m1, 'red', label='Training ' + metrics[0])
	b = plt.plot(epochs, m1_val, 'blue', label='Validation ' + metrics[0])
	m = plt.plot(epochs, m2, 'black', label='Training ' + metrics[1])
	g = plt.plot(epochs, m2_val, 'green', label='Validation ' + metrics[1])
	plt.legend([metrics[0], 'val_' + metrics[0], metrics[1], 'val_' + metrics[1]])
	
	plt.savefig(result_directory + 'Metrics_' + file_suffix + ".png")
	plt.clf()
	
	plt.title('Training and validation loss')
	plt.xlabel('Epochs')
	plt.ylabel(loss_function)
	plt.plot(epochs, loss, 'red', label='Training loss')
	plt.plot(epochs, val_loss, 'blue', label='Validation loss')
	plt.legend(['Training', 'Validation'])
	
	plt.savefig(result_directory + "Loss_" + file_suffix + ".png")
	plt.clf()
	

#########################################
"""
Description:
This function creates two plots for training loss and accuracy respectively (can combine list of history information in same plots).

Input variables/types and functions:
estimator_history_list:		A list of dict structure from Tensorflow (estimator.history for each item) with keys: 'accuracy', 'val_accuracy', 'loss' and 'val_loss'
file_suffix:				Is appended to hardcoded: "Results/Loss_"

Output (data and types):
None

Side-effext:	Saving of two individual plots för loss and accuracy (if history_list contains more than one item, all is appended to the same plot with vertical delimiter)

Constraints:
N/A

Exceptions:
N/A

Others:
Fileformat is automatically given as ".png"
Can be improved with more flexibility... (TBD)

"""

def plot_training_statistics (estimator_history_list, file_suffix = '', metrics = ['accuracy'], loss_function = 'mse', result_directory = 'Results/'):
	
	train_loss = []
	val_loss = []
	metrics_train = [] # Note, only one metrics value so far...
	metrics_val = []
	epochs_list = []
	
	for history_list in estimator_history_list:
		epochs_list.append(len(history_list['loss']))
		train_loss += history_list['loss']
		val_loss += history_list['val_loss']
		metrics_train += history_list[metrics[0]]
		metrics_val += history_list['val_' + metrics[0]]
	
	del epochs_list[-1] # For all consecutive trainings except the last...
	
	legends = ['Training', 'Validation']
	plt.title('Training and validation loss')
	plt.xlabel('Epochs')
	plt.ylabel(loss_function)
	plt.plot(range(len(train_loss)), train_loss, 'red')
	plt.plot(range(len(val_loss)), val_loss, 'blue')
	
	# Reduce first epock limit with 1...
	epoch_nr = 1
	epoch_limit = -1 # To get the FT line correct
	for epoch_line in epochs_list:
		epoch_limit += epoch_line
		plt.plot([epoch_limit, epoch_limit], plt.ylim())
		legends.append('Fine Tuning:' + str(epoch_nr))
		epoch_nr += 1
	
	plt.legend(legends)
	#plt.show()
	#sys.exit()
	
	plt.savefig(result_directory + 'Loss_' + file_suffix + ".png")
	plt.clf()
	
	legends = ['Training', 'Validation']
	plt.title('Training and validation metrics')
	plt.xlabel('Epochs')
	plt.ylabel(metrics[0])
	plt.plot(range(len(metrics_train)), metrics_train, 'red')
	plt.plot(range(len(metrics_val)), metrics_val, 'blue')
	
	epoch_nr = 1
	epoch_limit = -1 # Changed from 0 giving wrong vertical lina... # To get the FT line correct
	for epoch_line in epochs_list:
		epoch_limit += epoch_line
		plt.plot([epoch_limit, epoch_limit], plt.ylim())
		legends.append('Fine Tuning:' + str(epoch_nr))
		epoch_nr += 1
	
	plt.legend(legends)
	
	plt.savefig(result_directory + 'Metrics_' + file_suffix + ".png")
	plt.clf()
	plt.close()
	
	
"""
### Testdata
estimator_history_list = []
estimator_history_list.append({'loss': [0.01582418568432331, 0.015766989439725876, 0.015730028972029686, 0.01570207066833973, 0.015678800642490387, 0.015659207478165627, 0.015641994774341583, 0.01562679558992386, 0.015613114461302757, 0.015600541606545448], 'accuracy': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'val_loss': [1.1543887853622437, 1.1626918315887451, 1.1681809425354004, 1.1727310419082642, 1.1766356229782104, 1.179904818534851, 1.1830401420593262, 1.1858487129211426, 1.1885056495666504, 1.1909172534942627], 'val_accuracy': [0.8633333444595337, 0.8633333444595337, 0.8650000095367432, 0.8650000095367432, 0.8658333420753479, 0.8666666746139526, 0.8666666746139526, 0.8666666746139526, 0.8658333420753479, 0.8658333420753479]})
estimator_history_list.append({'loss': [0.01582418568432331, 0.015766989439725876, 0.015730028972029686, 0.01570207066833973, 0.015678800642490387, 0.015659207478165627, 0.015641994774341583, 0.01562679558992386, 0.015613114461302757, 0.015600541606545448], 'accuracy': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'val_loss': [1.1543887853622437, 1.1626918315887451, 1.1681809425354004, 1.1727310419082642, 1.1766356229782104, 1.179904818534851, 1.1830401420593262, 1.1858487129211426, 1.1885056495666504, 1.1909172534942627], 'val_accuracy': [0.8633333444595337, 0.8633333444595337, 0.8650000095367432, 0.8650000095367432, 0.8658333420753479, 0.8666666746139526, 0.8666666746139526, 0.8666666746139526, 0.8658333420753479, 0.8658333420753479]})


### From calling function
# network_result_handling.plot_training_statistics (estimator_list, file_suffix, loss_function = loss_function)
# plot_training_statistics (estimator_history_list, file_suffix = 'TEST', metrics = ['accuracy'], loss_function = 'X_loss')
"""


#########################################

def plot_regression_scatter (nn_output, ground_truth, plotheader, file_suffix, result_directory = 'Results/'):
	
	"""
	print('Scatter min predicted value: ', numpy.amin(nn_output))
	print('Scatter  max predicted value: ', numpy.amax(nn_output))
	print('Scatter  average predicted value: ', nn_output.mean())
	print('Scatter  min true value: ', numpy.amin(ground_truth))
	print('Scatter  max true value: ', numpy.amax(ground_truth))
	print('Scatter average true value: ', ground_truth.mean())
	"""
	
	""" OK...
	Distance min true value:  1.0
	Distance max true value:  55.0
	Distance average true value:  26.26307665597767
	"""
	
	R2_value = r2_score(ground_truth, nn_output) # true, pred
	plotheader = plotheader + ': R2 = ' + str("{:.2f}".format(round(R2_value, 2)))
	
	plt.figure(figsize=((6.4*2), (4.8*2))) # Default in inches = (6.4, 4.8) * 3 = (19.2, 14.4) # figsize=((6.4*2), (6.4*2))
	plt.title(plotheader, fontsize = 24)# default = ?
	plt.xlabel('True value', fontsize = 24)
	plt.ylabel('Predicted value', fontsize = 24)
	true_values = plt.scatter(ground_truth, nn_output, c='red', marker='*', s = 24) # s=2  (marker size) marker='.' (marker_type) marker='o', marker='*'
	plt.grid(True)
	#plt.show()
	plt.savefig(result_directory + plotheader + '_' + file_suffix + '.png')
	plt.clf()
	plt.close()
	

#########################################
	
# Old, not used...	
def plot_regression_curves (nn_output, ground_truth, plotheader, file_suffix, result_directory = 'Results/'):
	
	index_list = []
	index_counter = 0
	for index in nn_output:
		index_list.append(index_counter)
		index_counter += 1
	
	plt.figure(figsize=((6.4*2), (4.8*2))) #Default in inches = (6.4, 4.8) * 3 = (19.2, 14.4)
	plt.title(plotheader)
	plt.ylabel('Value')
	plt.xlabel('Sample')
	
	true_values = plt.plot(ground_truth, color='blue', marker='.',linestyle='dashed', linewidth = 0.25) # marker='o', linestyle='solid'
	values = plt.plot(nn_output, color='red', marker='.',linestyle='dashed', linewidth = 0.25) # marker='o', linestyle='solid'
	#plt.xticks(index_list)
	
	plt.legend(['True values_blue', 'Predicted_values_red']) # , 'True values'
	plt.grid(True)
	#plt.show()
	plt.savefig(result_directory + plotheader + '_' + file_suffix + '.png')
	plt.clf()
	plt.close()




#########################################

def plot_training_statistics_regression_iterative_results (history_list, x_values_list, file_suffix = '', loss_function = 'mse', log_x = True, result_directory = 'Results/'):
	
	loss_last_list = []
	#loss_mean_list = []
	val_loss_last_list = []
	#val_loss_mean_list = []
	
	for history in history_list:
		loss_last_list.append(history['loss'][-1])
		#loss_mean_list.append(numpy.mean(numpy.array(history['loss'])))
		
		val_loss_last_list.append(history['val_loss'][-1])
		#val_loss_mean_list.append(numpy.mean(numpy.array(history['val_loss'])))
		
		
	#epochs = range(len(loss_last_list))
	
	# My logfunction...
	x_label_extra = ''
	if log_x:
		print('Log values')
		x_values_list = list(map(math.log10, x_values_list))
		#print(x_values_list)
		x_label_extra = 'Log_10 '
	
	plt.title('Training and validation iterative loss')
	plt.xlabel(x_label_extra + 'L2-regularisation')
	plt.ylabel(loss_function)
	
	
	
	
	plt.plot(x_values_list, loss_last_list, marker = 'X', linestyle = 'solid', color = 'red')
	plt.plot(x_values_list, val_loss_last_list, marker = 'X', linestyle = 'solid', color = 'blue')
	
	plt.grid()
	
	plt.legend(['Training_last_value', 'Validation_last_value']) #, 'Training_mean_value', 'Validation_mean_value'])
	#plt.show()
	plt.savefig(result_directory + "Regression_iterative_" + file_suffix + ".png")
	plt.clf()
	plt.close()
	
############## END library functions ##############




