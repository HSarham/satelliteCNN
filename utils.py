from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate, LeaveOneOut
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, matthews_corrcoef, mean_absolute_error, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import keras
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import pickle
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeClassifierCV
from sklearn.svm import SVC, NuSVC
from concurrent.futures import ProcessPoolExecutor
from sklearn.utils.class_weight import compute_sample_weight
from seaborn import barplot, heatmap
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.utils import resample
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
# from imblearn.ensemble import BalancedRandomForestClassifier
from skimage.io import imread
import pandas
from keras.applications.mobilenet_v2 import MobileNetV2
from sklearn.linear_model import RidgeCV, RidgeClassifierCV, LogisticRegressionCV,\
ElasticNetCV, LarsCV
import re

def read_survery_targets(wealth_ratings_file_path):
    print("reading wealth ratings")
    wr = pandas.read_csv(wealth_ratings_file_path)
    targets={}
    targets["mean_combined"] = wr["mean_score270_comb_R_U"].to_numpy().astype(int)
    targets["mean_separated"] = wr["mean270_R_U_separated"].to_numpy().astype(int)
    targets["median_combined"] = wr["median_score_comb_R_U_270"].to_numpy().astype(int)
    targets["median_separated"] = wr["median270a_U_R_separated"].to_numpy().astype(int)
    return targets

def get_group_models_and_predictions(features, targets, groups=None, alpha_per_fold=True, clamp_outputs=True, fit_intercept=True, act="regression",\
                               weighted_samples=False, leave_one_out=False):

    num_samples = len(features)

    if groups == None:
        groups = [[i] for i in range(num_samples)]
    
    groups = np.array(groups)

    #convert the groups format
    sample_group_indices = np.empty(shape=num_samples,dtype=int)
    for i,group in enumerate(groups):
        for sample_index in group:
            sample_group_indices[sample_index] = i

    if leave_one_out or act=="regression":
        # cv = LeaveOneGroupOut()
        cv = LeaveOneOut()
    else:
        #cv = StratifiedGroupKFold(shuffle=True,random_state=0)
        cv = StratifiedKFold(shuffle=True,random_state=0)

    splits = [(train_indices,test_indices) for train_indices,test_indices in cv.split(features,targets)]#,groups)]

    features = np.array(features)
    targets = np.array(targets)
    
    alphas = 10**np.arange(0,7,0.5)
    Cs = 10**np.arange(-6,0.5,0.5)

    best_alphas = []
    best_alpha_models = []
    best_alpha_outputs = np.empty((0,1))
    output_targets = np.empty((0,))

    ss = StandardScaler()

    target_indices = np.empty((0,))

    for i,(train_indices,test_indices) in enumerate(splits):
        
        print("fold index:", i,end="\r")
        
        train_features = features[train_indices]
        train_targets = targets[train_indices]

        test_features = features[test_indices]
        test_targets = targets[test_indices]
 
        if weighted_samples and act!="regression":
            curr_weights = compute_sample_weight("balanced",train_targets)
        else:
            curr_weights=None

        ss.fit(train_features,sample_weight=curr_weights)
        train_features = ss.transform(train_features)
        test_features = ss.transform(test_features)

        if alpha_per_fold:
            if act=="regression":
                best_alpha_model = RidgeCV(alphas=alphas,fit_intercept=fit_intercept).fit(train_features,train_targets,sample_weight=curr_weights)
            elif act=="logistic_regression":
                best_alpha_model = LogisticRegressionCV(Cs=Cs, n_jobs=20, solver="newton-cg", fit_intercept=fit_intercept).fit(train_features,train_targets, sample_weight=curr_weights)
            elif act=="classification":
                best_alpha_model = RidgeClassifierCV(alphas=alphas,fit_intercept=fit_intercept).fit(train_features,train_targets, sample_weight=curr_weights)
            elif act=="lars_regression":
                best_alpha_model = LarsCV(cv=2,fit_intercept=fit_intercept).fit(train_features,train_targets,sample_weight=curr_weights)
            elif act=="elasticnet_regression":
                best_alpha_model = ElasticNetCV(cv=2,fit_intercept=fit_intercept).fit(train_features,train_targets,sample_weight=curr_weights)
            
            if act=="regression" or act=="classification" or act=="elasticnet_regression" or act=="lars_regression":
                best_alpha = best_alpha_model.alpha_
            else:
                best_alpha = best_alpha_model.C_
        
        best_alphas.append(best_alpha)

        best_alpha_models.append(best_alpha_model)

        sample_output = best_alpha_model.predict(test_features).reshape(-1,1)

        if clamp_outputs and act=="regression":
            min_curr_target = np.min(train_targets)
            max_curr_target = np.max(train_targets)

            sample_output = np.minimum(max_curr_target,np.maximum(min_curr_target,sample_output))

        best_alpha_outputs = np.concatenate((best_alpha_outputs,sample_output))
        output_targets = np.concatenate((output_targets,test_targets))
        target_indices = np.concatenate((target_indices,test_indices))

    print()

    return best_alpha_models, best_alpha_outputs, output_targets, best_alphas, target_indices

def get_proper_backbone(inner_model, dataset_type):
    if dataset_type == "1x1":
        shape = (224,224,3)

    elif dataset_type == "3x3":
        shape = (672,672,3)

    FE_backbone = MobileNetV2(input_shape = shape, include_top = False)
    FE_backbone.set_weights(inner_model.get_weights())

    return FE_backbone

def get_feature_extractor(inner_model, dataset_type):
    FE_backbone = get_proper_backbone(inner_model, dataset_type)
    average_pooling_2D = keras.layers.GlobalAveragePooling2D()(FE_backbone.output)
    feature_exractor = keras.Model(inputs=[FE_backbone.input], outputs=[average_pooling_2D])

    return feature_exractor

def coordinates_to_grid(lat, lon, grid_positions_float):
    min_lat = -0.7909869169572299
    max_lat = -11.910350051340407
    min_lon = 28.81584517498368
    max_lon = 40.63713423748368
    num_rows = 552.5
    num_cols = 587.5

    lat_delta = (max_lat-min_lat)/num_rows
    lon_delta = (max_lon-min_lon)/num_cols

    float_row = (lat-min_lat)/lat_delta+1
    float_col = (lon-min_lon)/lon_delta+1

    grid_positions_float += list(zip(list(float_row),list(float_col)))

    row = float_row.astype(int).tolist()
    col = float_col.astype(int).tolist()

    return list(zip(row, col))

def get_images_targets_folder_based(dataset_path, mode, images_subfolder, corrected, grid_positions_float=None):

    if grid_positions_float==None:
        grid_positions_float = []
    
    print("dataset path:", dataset_path)

    dataset = pandas.read_csv(dataset_path, sep=";", index_col=False)

    if corrected:
        lats = dataset["Corrected Latitude"].to_numpy()
        lons = dataset["Corrected Longitude"].to_numpy()
    else:
        lats = dataset["LATNUM,N,24,15"].to_numpy()
        lons = dataset["LONGNUM,N,24,15"].to_numpy()

    if mode == "3x3":
        lats = lats[::9]
        lons = lons[::9]

    grid_positions = coordinates_to_grid(lats,lons,grid_positions_float)

    if mode == "3x3":
        new_grid_positions = []
        for r,c in grid_positions:
            for row in range(r-1,r+2):
                for col in range(c-1,c+2):
                    new_grid_positions.append((row,col))
        grid_positions = new_grid_positions

    wealth_indices = dataset["v191_mean,N,24,15"].to_numpy()/100000 #To make the numbers smaller
    wealth_indices = wealth_indices.tolist()

    # image_matrix_codes = dataset["Image_Matrix_Code"].to_list()
    image_area_codes = dataset["Image_Area_Code"].to_list()

    images_subfolder += "/"+image_area_codes[0]

    folder_images = os.listdir(images_subfolder)

    print("finding images...",end="")
    
    num_re = re.compile("([0-9]*)_([0-9]*)")
    image_names_dict={}
    for image_name in folder_images:
        match = num_re.match(image_name)
        key = (int(match.group(1)),int(match.group(2)))
        image_names_dict[key]=image_name

    image_names=[]
    for grid_position in grid_positions:
        image_names.append(image_names_dict[grid_position])

    print("Done")

    return get_mode_images_targets(image_names, images_subfolder, wealth_indices, mode)

def get_mode_images_targets(image_names, images_subfolder, wealth_indices, mode):
    print("reading images..")
    images = []
    for image_name in image_names:
        image_path = images_subfolder + "/" + image_name
        images.append(imread(image_path))

    if mode == "1x1":
        return np.array(images), wealth_indices

    elif mode == "3x3":
        print("concatenating 3x3 images..")
        num_images = len(images)

        num_positions = num_images//9

        orig_im_rows, orig_im_cols = (images[0].shape[0], images[0].shape[1])

        concat_image_shape = (orig_im_rows*3, orig_im_cols*3, 3)
        #get concatenated images
        concat_images = []
        for i in range(num_positions):
            concat_image = np.empty(concat_image_shape)
            for j in range(9):
                im_row_index = j//3
                im_col_index = j%3
                
                start_pix_row = im_row_index * orig_im_rows
                end_pix_row = (im_row_index + 1) * orig_im_rows
                start_pix_col = im_col_index * orig_im_cols
                end_pix_col = (im_col_index + 1) * orig_im_cols

                concat_image[start_pix_row:end_pix_row, start_pix_col:end_pix_col,:] = images[i*9+j]
            concat_images.append(concat_image)

        concat_images = np.array(concat_images)

        targets = []
        for i in range(len(wealth_indices)//9):
            targets.append(wealth_indices[i*9])

        return concat_images, targets

def plus_minus_accuracy(outputs, targets, weights=None,thresh=1):
    outputs = np.array(outputs)
    targets = np.array(targets)
    if np.any(weights)==None:
        weights = np.ones(outputs.shape)

    hit_mask = (np.abs(outputs-targets)<=thresh).astype(int)
    return np.sum(hit_mask*weights)/np.sum(weights)

def save_output_histogram(outputs,path,title=""):
    plt.hist(outputs,bins=np.arange(0.5,6),edgecolor="white")
    plt.xticks([1,2,3,4,5],["Poorest","Poor","Middle","Rich","Richest"])
    if title != "":
        plt.title(title)
    plt.savefig(path,bbox_inches="tight",pad_inches=0)
    plt.close()

def get_confusion_matrix_plot(targets,outputs,ax,title=""):
    targets = np.array(targets,dtype=int)
    balanced_weights = compute_sample_weight("balanced",targets)

    pm_accuracy = plus_minus_accuracy(outputs,targets)
    weidhted_pm_accuracy = plus_minus_accuracy(outputs,targets,weights=balanced_weights)

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    print(f"pm_accuracy:{pm_accuracy:0.2}")
    print(f"weighted pm_accuracy:{weidhted_pm_accuracy:0.2}")

    mcc = matthews_corrcoef(y_true=targets,y_pred=outputs)
    weighted_mcc = matthews_corrcoef(y_true=targets,y_pred=outputs,sample_weight=balanced_weights)

    print(f"multiclass MCC:{mcc:0.2}")
    print(f"multiclass weighted MCC:{weighted_mcc:0.2}")

    MAE = mean_absolute_error(y_true=targets,y_pred=outputs)
    weighted_MAE = mean_absolute_error(y_true=targets,y_pred=outputs,sample_weight=balanced_weights)

    print("dichotomy binary MCCs:",dichotomy_binary_MCC(targets,outputs))
    print("weighted dichotomy binary MCCs:",dichotomy_binary_MCC(targets,outputs,sample_weights=balanced_weights))

    accuracy = accuracy_score(y_true=targets,y_pred=outputs)
    weighted_accuracy = accuracy_score(y_true=targets,y_pred=outputs,sample_weight=balanced_weights)

    print(f"accruacy:{accuracy:0.2}")
    print(f"weigthed accruacy:{weighted_accuracy:0.2}")
    # num_equal_targets = np.sum(best_alpha_outputs==curr_targets)
    # num_all_targets = len(best_alpha_outputs)
    # classification_rate = num_equal_targets/num_all_targets

    cmd = ConfusionMatrixDisplay.from_predictions(y_true=targets,y_pred=outputs,ax=ax)#,cmap="plasma")#,normalize="true",values_format=".0%")

    labels = ["Poorest","Poor","Middle","Rich","Richest"]
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels,rotation=45)
    ax.set_ylabel("True (DHS) label")

    # ax.set_title(f"{title}MAE: {MAE:0.2}, Balanced MAE: {weighted_MAE:0.2},\n Acc:{accuracy:2.1%}, Balanced Acc:{weighted_accuracy:2.1%}")
    ax.set_title(title)
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

def dichotomy_binary_MCC(y_true,y_pred,sample_weights=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    labels = np.unique(y_true)
    MCCs = {}
    for i in range(len(labels)-1):
        l = labels[i]

        curr_y_pred = y_pred.copy()
        curr_y_true = y_true.copy()

        curr_y_pred[y_pred>l] = 1
        curr_y_pred[y_pred<=l] = 0

        curr_y_true[y_true>l] = 1
        curr_y_true[y_true<=l] = 0

        MCC = matthews_corrcoef(curr_y_true,curr_y_pred,sample_weight=sample_weights)
        MCCs[l]=MCC

    return MCCs

def get_dichotomy_labels(in_labels):
    labels = np.unique(in_labels)
    all_labels = []
    for i in range(len(labels)-1):#the last label is already taken care of by its previous label
        l = labels[i]
        curr_labels = in_labels.copy()
        curr_labels[in_labels>l]=1
        curr_labels[in_labels<=l]=0
        all_labels.append(curr_labels)
    return np.array(all_labels)

def reweight_classes_resample(inputs,targets,sample_weights,random_state=0):
    
    class_labels = np.unique(targets)
    label_inputs = {}

    num_classes = len(class_labels)

    num_class_samples = np.zeros(shape=num_classes,dtype=int)
    sum_class_weights = np.zeros(shape=num_classes)

    for label in class_labels:
        label_indices = np.nonzero(targets == label)
        
        label_inputs[label] = inputs[label_indices]

        class_index = int(label) - 1 
        num_class_samples[class_index] = len(label_indices[0])
        sum_class_weights[class_index] = np.sum(sample_weights[label_indices])

    samples_over_weight = num_class_samples/sum_class_weights

    max_label = np.argmax(samples_over_weight)+1
    ref_samples_over_weights = np.max(samples_over_weight)

    num_new_class_samples = np.round(sum_class_weights * ref_samples_over_weights).astype(int)
    class_samples_diff = num_new_class_samples-num_class_samples

    # print(f"{num_new_class_samples=}")
    # print(f"{num_new_class_samples/np.sum(num_new_class_samples)=}")
    # print(f"{num_class_samples=}")
    # print(f"{sum_class_weights=}")
    # print(f"{num_class_samples/np.sum(num_class_samples)=}")
    # print(f"{sum_class_weights/np.sum(sum_class_weights)=}")
    # print(f"{samples_over_weight=}")
    # print(f"{class_samples_diff=}")
    # print(f"{max_label=}")
    
    # return

    new_inputs = label_inputs[max_label]
    new_targets = [max_label]*len(new_inputs)

    for class_label in class_labels:
        if class_label != max_label:
            class_index = int(class_label) - 1

            print(f"{class_index=}")
            print(f"{class_samples_diff[class_index]=}")
            resampled_inputs = resample(label_inputs[class_label],n_samples=class_samples_diff[class_index],random_state=random_state)

            new_inputs = np.concatenate([new_inputs,label_inputs[class_label],resampled_inputs])
            new_targets += [class_label]*num_new_class_samples[class_index]

    return new_inputs, np.array(new_targets)

def balance_classes_resample(inputs,targets,random_state):
    labels = np.unique(targets)
    label_inputs = {}

    for label in labels:
        label_indices = np.nonzero(targets == label)
        label_inputs[label] = inputs[label_indices]
        
    label_inputs_items = list(label_inputs.items())
    max_index = np.argmax([len(item[1]) for item in label_inputs_items])
    max_label = label_inputs_items[max_index][0]
    class_samples = len(label_inputs[max_label])

    new_inputs = label_inputs[max_label]
    new_targets = [max_label]*class_samples

    for label in labels:
        if label != max_label:
            diff = class_samples-len(label_inputs[label])
            resampled_inputs = resample(label_inputs[label],n_samples=diff,random_state=random_state)
            new_inputs = np.concatenate([new_inputs,label_inputs[label],resampled_inputs])
            new_targets += [label]*class_samples

    return new_inputs,new_targets

def predict(model, inputs, model_type):
        if model_type == "ordinal":
            return np.argmax(model.model.predict(model.params,exog=inputs),axis=1)+1
        elif model_type == "dichotomies":
            predictions = []
            for d_model in model:
                predictions.append(d_model.predict(inputs))
            return predictions
        else:
            return model.predict(inputs)

def get_model_accuracy(outputs,test_Y,sample_weights=None, model_type=None):
    if model_type == "dichotomies":
        accuracies = []
        dich_targets = get_dichotomy_labels(test_Y)
        for i,curr_targets in enumerate(dich_targets):
            accuracies.append(balanced_accuracy_score(curr_targets,outputs[i]))
        return accuracies
    elif outputs.ndim>1:
        outputs_cat = np.argmax(outputs,axis=1)
        test_Y_cat = np.argmax(test_Y,axis=1)
        return accuracy_score(test_Y_cat,outputs_cat,sample_weight=sample_weights)
    else:
        return accuracy_score(test_Y,outputs,sample_weight=sample_weights)
    
def get_confusion_matrix(outputs,test_Y,sample_weights=None):
    outputs_cat = np.argmax(outputs,axis=1)
    test_Y_cat = np.argmax(test_Y,axis=1)
    return confusion_matrix(test_Y_cat,outputs_cat,sample_weight=sample_weights)


def train_model(inputs, targets, random_state=0, model_type=None, class_weight=None, solver="lbfgs", max_iter=1000):
        if model_type=="ordinal":
            if class_weight!=None:
                inputs,targets = balance_classes_resample(inputs,targets,random_state)
            return OrderedModel(targets,inputs,distr="logit").fit(method=solver, disp=False, maxiter=max_iter)
        elif model_type=="dichotomies":
            d_targets = get_dichotomy_labels(targets)
            models = []
            for curr_targets in d_targets:
                models.append(LogisticRegressionCV(solver=solver,max_iter=max_iter,n_jobs=-1,class_weight="balanced").fit(inputs,curr_targets))
            return models
        elif model_type=="multinomial":   #multinomial
            return LogisticRegressionCV(solver=solver,max_iter=max_iter,n_jobs=-1,class_weight=class_weight).fit(inputs,targets)
        elif model_type == "Random Forest":
            return RandomForestClassifier(n_jobs=-1,class_weight=class_weight,random_state=random_state).fit(inputs,targets)
        # elif model_type == "Balanced Random Forest":
        #     return BalancedRandomForestClassifier(n_jobs=-1,class_weight=class_weight,random_state=random_state).fit(inputs,targets)
        elif model_type == "Random Forest Regressor":
            return RandomForestRegressor(n_jobs=-1,random_state=random_state).fit(inputs,targets)


def evaluate_linear_model(inputs, targets, model_type=None, path_prefix=None, feature_names=None, num_experiments=1,
                           num_inner_experiments=1, get_feature_importance=False, output_test_indices=[],outputs={}):
    
    ss = StandardScaler()

    test_modes = {"normal-testing":False} if "regressor" in model_type.lower() else {"normal-testing":False,"balanced-testing":True}
    train_modes = {"normal-training":False} if "regressor" in model_type.lower() else {"normal-training":False,"weighted-samples":True}
    
    accuracies = {train_mode:{test_mode:[] for test_mode in test_modes} for train_mode in train_modes}
    confusion_matrices = {train_mode:{test_mode:[] for test_mode in test_modes} for train_mode in train_modes}
    MAEs = {train_mode:{test_mode:[] for test_mode in test_modes} for train_mode in train_modes}
    dennis_MCCs = {train_mode:{test_mode:[] for test_mode in test_modes} for train_mode in train_modes}
    split_accuracies = {train_mode:{test_mode:[] for test_mode in test_modes} for train_mode in train_modes}
    outputs.update({train_mode:[] for train_mode in train_modes})
    models = {train_mode:[] for train_mode in train_modes}
    
    test_inputs = np.empty(shape=(0,inputs.shape[1]))
    test_targets = []

    mean_importances = {train_mode:[] for train_mode in train_modes}
    std_importances = {train_mode:[] for train_mode in train_modes}

    
    for e in range(num_experiments):

        print(f"experiment {e} of {num_experiments}")

        if model_type == "Random Forest Regressor":
            cv = LeaveOneOut()
        else:
            cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=e)

        splits = cv.split(inputs,targets)
        splits = [(train_indices, test_indices) for train_indices, test_indices in splits]

        for i ,(train_indices, test_indices) in enumerate(splits):
            
            output_test_indices += test_indices.tolist()
            
            train_X = inputs[train_indices]
            train_Y = targets[train_indices]

            test_X = inputs[test_indices]
            test_Y = targets[test_indices]

            train_X = ss.fit_transform(train_X)
            test_X = ss.transform(test_X)
            
            test_weights = compute_sample_weight("balanced",test_Y)
            
            for ie in range(num_inner_experiments):

                test_inputs = np.concatenate([test_inputs,test_X])
                test_targets = np.concatenate([test_targets,test_Y])

                random_state = e*num_inner_experiments + ie

                for train_mode in train_modes:
                    
                    if train_modes[train_mode]:
                        model = train_model(train_X,train_Y,random_state=random_state,model_type=model_type,class_weight="balanced")
                    else:
                        model = train_model(train_X,train_Y,random_state=random_state,model_type=model_type)

                    models[train_mode].append(model)
                        
                    curr_outputs = predict(model,test_X,model_type)

                    if model_type == "Random Forest" and get_feature_importance and train_modes[train_mode]:
                        pi = permutation_importance(model,test_X,test_Y,sample_weight=test_Y,n_repeats=10,n_jobs=-1,random_state=0)
                        mean_importances[train_mode].append(pi.importances_mean)
                        std_importances[train_mode].append(pi.importances_std)

                    if model_type!="dichotomies":
                        outputs[train_mode] = np.concatenate([outputs[train_mode],curr_outputs])

                    if model_type!="Random Forest Regressor":
                        for test_mode in test_modes:
                            if test_modes[test_mode]:
                                curr_accuracies = get_model_accuracy(curr_outputs,test_Y,sample_weights=test_weights,model_type=model_type)
                            else:
                                curr_accuracies = get_model_accuracy(curr_outputs,test_Y,model_type=model_type)
                                
                            split_accuracies[train_mode][test_mode].append(curr_accuracies)

                print(f"split {i}, {(ie+1)/num_inner_experiments:0.2%}",end="\r")

    if model_type == "Random Forest" and get_feature_importance:
        for train_mode in train_modes:
            print(f"{train_mode=}")
            if len(mean_importances[train_mode]) == 0:
                continue
            average_mean_importances = np.mean(mean_importances[train_mode],axis=0)
            average_std_importances = np.mean(std_importances[train_mode],axis=0)
            for i in range(len(feature_names)):
                print(f"av_imp_mean: {average_mean_importances[i]:.2%}, \
                    \tav_imp_std: {average_std_importances[i]:.2%}, \
                    \tmean/std: {average_mean_importances[i]/average_std_importances[i]:.2}\
                        ({feature_names[i]})")


    if model_type == "dichotomies":
        for train_mode in models:
            coeffs = []
            for model_set in models[train_mode]:
                coeffs.append([model.coef_.reshape(-1) for model in model_set])

            mean_feature_weights = np.mean(coeffs,axis=0)
            y_labels=["1 v.s. 2-5","1-2 v.s. 3-5","1-3 v.s. 4-5","1-4 v.s. 5"]
            
            fig = plt.figure(figsize=(15,7))
            ax=heatmap(mean_feature_weights/np.linalg.norm(mean_feature_weights,axis=1).reshape((-1,1)),annot=True,fmt="0.2f",xticklabels=feature_names,yticklabels=y_labels)
            ax.set_xticklabels(ax.get_xticklabels(),rotation=35)
            ax.set_yticklabels(ax.get_yticklabels(),rotation=35)

            fig.savefig(path_prefix+f"{train_mode}_dichotomy_LR_weights_normalized.pdf",bbox_inches="tight",pad_inches=0)
            plt.close()

            fig,ax = plt.subplots(4,1,figsize=(15,7))
            for i in range(4):
                heatmap(mean_feature_weights[i].reshape((1,-1)),annot=True,fmt="0.2f",xticklabels=feature_names,yticklabels=[y_labels[i]],ax=ax[i])
                if i==3:
                    ax[i].set_xticklabels(ax[i].get_xticklabels(),rotation=35)
                else:
                    ax[i].set_xticklabels([])
                ax[i].set_yticklabels(ax[i].get_yticklabels(),rotation=35)

            fig.savefig(path_prefix+f"{train_mode}_dichotomy_LR_weights.pdf",bbox_inches="tight",pad_inches=0)
            plt.close()
            
            for test_mode in split_accuracies[train_mode]:
                print(f"{train_mode}{test_mode}")
                curr_accuracies = split_accuracies[train_mode][test_mode]
                print(np.mean(curr_accuracies,axis=0))

        return
    
    if model_type != "Random Forest Regressor":
        for i,train_mode in enumerate(train_modes):
            print(f"{train_mode=}")

            curr_outputs = outputs[train_mode]
            save_output_histogram(curr_outputs,f"{path_prefix}/hist_{model_type}_LR_{train_mode}.pdf",title=model_type)
            for test_mode in test_modes:
                split_accuracies[train_mode][test_mode] = np.mean(split_accuracies[train_mode][test_mode],axis=0)

                if test_modes[test_mode] == train_modes[train_mode]:
                    plt.close()
                    fig, ax = plt.subplots()
                    
                    get_confusion_matrix_plot(test_targets,curr_outputs,ax,title=f"{model_type}")
                    
                    if test_modes[test_mode]:
                        plt.savefig(f"{path_prefix}/CM_{model_type}_balanced.pdf",bbox_inches="tight",pad_inches=0)
                    else:
                        plt.savefig(f"{path_prefix}/CM_{model_type}_original.pdf",bbox_inches="tight",pad_inches=0)
                    plt.close()
    
        balanced_weights = compute_sample_weight("balanced",test_targets)
        for train_mode in train_modes:
            curr_outputs = outputs[train_mode]
            for test_mode in test_modes:
                if test_modes[test_mode]:
                    accuracies[train_mode][test_mode]=accuracy_score(test_targets,curr_outputs,sample_weight=balanced_weights)
                    confusion_matrices[train_mode][test_mode]=confusion_matrix(test_targets,curr_outputs,sample_weight=balanced_weights)
                    MAEs[train_mode][test_mode]=mean_absolute_error(test_targets,curr_outputs,sample_weight=balanced_weights)
                    dennis_MCCs[train_mode][test_mode]=dichotomy_binary_MCC(test_targets,curr_outputs,sample_weights=balanced_weights)
                else:
                    accuracies[train_mode][test_mode]=accuracy_score(test_targets,curr_outputs)
                    confusion_matrices[train_mode][test_mode]=confusion_matrix(test_targets,curr_outputs)
                    MAEs[train_mode][test_mode]= mean_absolute_error(test_targets,curr_outputs)
                    dennis_MCCs[train_mode][test_mode]=dichotomy_binary_MCC(test_targets,curr_outputs)
                
        return accuracies,confusion_matrices,MAEs,dennis_MCCs,split_accuracies

def backward_elimination(inputs, targets, feature_names, output_path, class_weight=None, model_type=None, balanced_testing=False,num_experiments=100):
        
    num_features = inputs.shape[1]                                                                                                                                                                                                                                                                                                                                            
    experiments = []

    ss = StandardScaler()

    for e in range(num_experiments):
        cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=e)
        splits = [(train_indices, test_indices) for train_indices, test_indices in cv.split(inputs,targets)]

        curr_inputs = inputs.copy()
        curr_feature_names = feature_names.copy()

        sorted_feature_removal_accuracies = []
        sorted_feature_names = []
        # mean_split_curves = []
        original_accuracies = []
        feature_models = []
        for i in range(num_features):# for loop for eliminating features
            split_feature_accuracies = []
            # split_curves = []
            outputs = []
            test_targets = []
            split_models = []
                
            for train_indices, test_indices in splits: #cross-validation
                # num_curr_features = curr_inputs.shape[1]
                #train a model with the current features
                # keras_model = get_keras_model(input_shape=num_curr_features, num_classes=targets.shape[1], num_hiddens=num_hiddens)
                # keras_model.compile(optimizer=keras.optimizers.Adam(),loss=keras.losses.CategoricalCrossentropy(),
                #                     metrics=[keras.metrics.CategoricalAccuracy()])

                train_X = curr_inputs[train_indices]
                train_Y = targets[train_indices]

                test_X = curr_inputs[test_indices]
                test_Y = targets[test_indices]

                train_X = ss.fit_transform(train_X)
                test_X = ss.transform(test_X)

                model = train_model(train_X,train_Y,random_state=e,model_type=model_type,class_weight=class_weight)
                split_models.append(model)
                # model, split_curve = train_MLP(train_X,train_Y,test_X,test_Y,num_hiddens,epochs)
                # split_curves.append(split_curve)

                test_targets = np.concatenate([test_targets,test_Y])
                outputs = np.concatenate([outputs,predict(model,test_X,model_type)])

                if balanced_testing:
                    feature_accuracies = get_feature_effects(model, test_X, test_Y, compute_sample_weight("balanced",test_Y), model_type)
                else:
                    feature_accuracies = get_feature_effects(model, test_X, test_Y, None, model_type)
                split_feature_accuracies.append(feature_accuracies)
            
            feature_models.append(split_models)
            
            if balanced_testing:
                original_accuracy = get_model_accuracy(outputs,test_targets,compute_sample_weight("balanced",test_targets), model_type)
            else:
                original_accuracy = get_model_accuracy(outputs,test_targets,None, model_type)
                
            original_accuracies.append(original_accuracy)
            
            mean_feature_removal_accuracies = np.mean(split_feature_accuracies,axis=0)
            worst_feature_index = np.argmax(mean_feature_removal_accuracies)
            
            print("mean_feature_removal_accuracies:",mean_feature_removal_accuracies)
            print("worst_feature_index",worst_feature_index)

            sorted_feature_removal_accuracies.append(mean_feature_removal_accuracies[worst_feature_index])
            sorted_feature_names.append(curr_feature_names[worst_feature_index])

            curr_inputs = np.delete(curr_inputs,worst_feature_index,axis=1)
            curr_feature_names = np.delete(curr_feature_names,worst_feature_index,axis=0)
        
        experiment = {}
        experiment["models"] = feature_models
        experiment["sorted_feature_removal_accuracies"] = sorted_feature_removal_accuracies
        experiment["sorted_feature_names"] = sorted_feature_names
        experiment["original_accuracies"] = original_accuracies

        experiments.append(experiment)

    with open(output_path,"wb") as file:
        pickle.dump(experiments,file)

def visualize_backward_elimination_results(input_path, feature_names, save_path_prefix, file_name_prefix, model_type, dichotomy=False, num_folds=5):

    all_experiments=[]

    if dichotomy:
        for dich_i in range(4):
            curr_input_path = input_path.format(dich_i=dich_i)
            print(curr_input_path)
            with open(curr_input_path,"rb") as file:
                experiments = pickle.load(file)
            all_experiments.append(experiments)
    else:
        with open(input_path,"rb") as file:
            all_experiments.append(pickle.load(file))

    # num_features = len(experiments[0]["sorted_feature_names"])
    # validation_curves = []

    if dichotomy:
        save_path_prefix += "/BE_logistic_regression/dichotomy/"
    else:
        save_path_prefix += "/BE_logistic_regression/" + model_type + "/"

    for folder_name in ["weights","curve","ranking"]:
        os.makedirs(save_path_prefix+folder_name,exist_ok=True)

    all_mean_feature_weights = []
    for experiment_set in all_experiments:
        feature_weights = []
        for experiment in experiment_set:
            num_eliminated_features = 0
            fold_models = experiment["models"][num_eliminated_features]
            
            experiemnt_feature_weights = []

            for fold_model in fold_models:
                
                if model_type=="ordinal":
                    experiemnt_feature_weights.append(fold_model.params)
                else:
                    experiemnt_feature_weights.append(fold_model.coef_)
            
            feature_weights.append(experiemnt_feature_weights)

        mean_feature_weights = np.mean(feature_weights,axis=(0,1))
        
        all_mean_feature_weights.append(mean_feature_weights)

    if model_type == "ordinal":
        fig_height = 1.5
    elif dichotomy:
        fig_height = 6.5
    else:
        fig_height = 7

    dich_labels = ["1 vs 2-5","1-2 vs 3-5","1-3 vs 4-5","1-4 vs 5"]
    multinomial_labels = ["Poorer","Poor","Average","Wealthy","Wealthier"]

    fig, ax = plt.subplots(len(all_experiments),1,figsize=(15,fig_height))
    for i, mean_feature_weights in enumerate(all_mean_feature_weights):
        if model_type=="ordinal":
            heatmap(mean_feature_weights[:-4].reshape(1,-1),annot=True,fmt="0.2f",xticklabels=feature_names,ax=ax)
            ax.set_yticks([])
        elif dichotomy:
            heatmap(mean_feature_weights[0].reshape(1,-1),annot=True,fmt="0.2f",xticklabels=feature_names,ax=ax[i])
            ax[i].set_yticklabels([dich_labels[i]],rotation=35)
        else:
            heatmap(mean_feature_weights,annot=True,fmt="0.2f",xticklabels=feature_names,
                            yticklabels=multinomial_labels,ax=ax)
        

        if dichotomy:
            if i==len(all_mean_feature_weights)-1:
                ax[i].set_xticklabels(ax[i].get_xticklabels(),rotation=35)
            else:
                ax[i].set_xticks([])
        else:
            ax.set_xticklabels(ax.get_xticklabels(),rotation=35)

    fig.savefig(save_path_prefix+"/weights/"+file_name_prefix+"_weights.pdf",bbox_inches="tight",pad_inches=0)
    plt.close()

    all_feature_ranks = []
    all_accuracy_curves = []
    for i,experiments in enumerate(all_experiments):
        
        feature_ranks = {name:[] for name in experiments[0]["sorted_feature_names"]}
        accuracy_curves = []

        for experiment in experiments:

            # print(experiment["original_accuracies"])
            accuracy_curves.append(experiment["original_accuracies"])
            
            for j,key in enumerate(experiment["sorted_feature_names"]):
                feature_ranks[key].append(j)
        
        all_feature_ranks.append(feature_ranks)
        all_accuracy_curves.append(accuracy_curves)
    #     mean_split_curves = experiment["mean_split_curves"]

    #     validation_curves.append(mean_split_curves)
    
    # mean_validations_curves = np.mean(validation_curves,axis=0)
    # for i,mean_curve in enumerate(mean_validations_curves):
    #     plt.plot(mean_curve,label=str(i)+" features removed")
    #     plt.legend()
    # plt.show()

    if dichotomy:
        fig_height = 7
    else:
        fig_height = 4.8


    fig, ax= plt.subplots(len(all_experiments),1,figsize=(12.8,fig_height),layout="constrained")
    for i , accuracy_curves in enumerate(all_accuracy_curves):
        curr_ax = ax[i] if dichotomy else ax

        mean_accuracy_curve = np.mean(accuracy_curves,axis=0)
        accuracy_curve_std = np.std(accuracy_curves,axis=0)
        curr_ax.errorbar(range(len(mean_accuracy_curve)),mean_accuracy_curve,
                       yerr=accuracy_curve_std,capsize=3)
        if dichotomy:
            curr_ax.set_ylabel(dich_labels[i])
        if i != len(all_accuracy_curves)-1:
            curr_ax.set_xticklabels([])
        
        
    fig.supylabel("Average accuracy")
    fig.supxlabel("Number of features removed")
    
    fig.savefig(save_path_prefix+"/curve/"+file_name_prefix+"_curve.pdf",bbox_inches="tight",pad_inches=0)
    plt.close()

        # for i, curve in enumerate(msc):
        #     plt.plot(curve,label=str(i))
        # plt.legend()
        # plt.show()
        # continue

        # plt.plot(moa)
        # plt.xlabel("# of removed features")
        # plt.ylabel("classification accuracy")
        # plt.show()
    # print("average ranks:")

    fig,ax = plt.subplots(len(all_feature_ranks),1,figsize=(12.8,fig_height),sharey=True,layout="constrained")

    for i,feature_ranks in enumerate(all_feature_ranks):
        curr_ax = ax[i] if dichotomy else ax
        rank_tuples = [(key,np.mean(feature_ranks[key])) for key in feature_ranks]
        rank_tuples.sort(key=lambda x:x[1])
        print(rank_tuples)
        barplot(dict(rank_tuples),ax=curr_ax)
        
        # curr_ax.xaxis.set_ticks(curr_ax.get_ticklocs(),labels=curr_ax.get_xticklabels(), )
        # curr_ax.set_xticklabels(curr_ax.get_xticklabels(),rotation=20)
        curr_ax.tick_params(axis="x",labelrotation=20)
        if dichotomy:
            ax[i].set_ylabel(dich_labels[i])
        
        
    fig.supylabel("Average elimination order")
    
    fig.savefig(save_path_prefix+"/ranking/"+file_name_prefix+"_rank.pdf",bbox_inches="tight",pad_inches=0)
    plt.close()
    
    if dichotomy:
        fig = plt.figure()
        ax = fig.add_subplot(111,projection="3d")
        bar3d_features = ["roofing_material","greenery","building_density"]
        xticks = np.arange(len(dich_labels))
        yticks = np.arange(len(bar3d_features))
        xx, yy = np.meshgrid(xticks,yticks)
        x = xx.ravel()
        y = yy.ravel()
        z = np.zeros(x.shape)
            
        dz = np.empty(xx.shape)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                dz[i,j] = np.mean(all_feature_ranks[j][bar3d_features[i]])

        dx = dy = 0.75
        
        ax.bar3d(x-dx/2,y-dy/2,z,dx,dy,dz.ravel())
        ax.set_xticks(xticks,dich_labels)
        ax.set_yticks(yticks,bar3d_features)
        fig.savefig(save_path_prefix+"/ranking/"+file_name_prefix+"_3D.pdf",bbox_inches="tight",pad_inches=0)
        plt.close()

    # for key, mean_rank in rank_tuples:
    #     print(key,mean_rank)

def get_feature_effects(model, test_X, test_Y, sample_weights=None, model_type=None):
    num_features = test_X.shape[1]

    accuracies=[]
    for i in range(num_features):#for loop to check the effect of each feature on performance
        feature_removed_inputs = test_X.copy()
        feature_removed_inputs[:,i] = 0

        outputs = predict(model,feature_removed_inputs,model_type)
        accuracies.append(get_model_accuracy(outputs,test_Y,sample_weights,model_type))

    return accuracies

def get_sklearn_model(input_shape,num_hiddens):
    return MLPClassifier(hidden_layer_sizes=(num_hiddens,), batch_size=input_shape[0])

def get_keras_model(input_shape,num_classes,num_hiddens):
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Dense(num_hiddens,activation="relu")(inputs)
    outputs = keras.layers.Dense(num_classes,activation="softmax")(x)

    return keras.Model(inputs=inputs, outputs=outputs)

