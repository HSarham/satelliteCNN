import sys
import matplotlib.pyplot as plt
from tensorflow import keras
import pandas
from keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import tensorflow as tf 
import os
from seaborn import heatmap
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import StandardScaler
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.utils.class_weight import compute_sample_weight
from utils import backward_elimination, visualize_backward_elimination_results,evaluate_linear_model,\
    get_confusion_matrix_plot, save_output_histogram, get_dichotomy_labels, reweight_classes_resample,\
    get_images_targets_folder_based, get_feature_extractor, get_group_models_and_predictions, read_survery_targets

def get_night_light_error(preprocessed_images, log_light_model, dataset_file_path):
    dataset = pandas.read_csv(dataset_file_path, sep=";", index_col=False)
    light_gt = np.log10(1+dataset["Light_values_sum"].to_numpy())
    log_light_outputs = log_light_model.predict(preprocessed_images)
    light_outputs = log_light_outputs
    light_mae = mean_absolute_error(light_gt,light_outputs)
    light_r2 = r2_score(light_gt,light_outputs)
    print(f"light mean absolute error:{light_mae}, light r2:{light_r2}")

def get_CNN_outputs(dataset_file_path, dataset_type, images_subfolder, model_path, targets, leave_one_out=False, act="classification", ordered_outputs={}):
    print("reading images")
    images, _ = get_images_targets_folder_based(dataset_file_path, dataset_type, images_subfolder, corrected=True)

    preprocessed_images = preprocess_input(images)
    log_light_model = keras.models.load_model(model_path)

    # get_night_light_error(preprocessed_images, log_light_model, dataset_file_path)
    
    print("loading the keras model")
    inner_model = log_light_model.get_layer("mobilenetv2_1.00_224")
    print("getting the feature extractor")
    feature_extractor = get_feature_extractor(inner_model, dataset_type)
    print("extracting features")
    features = feature_extractor.predict(preprocessed_images,batch_size=8)

    weighted_sample_flags = {"original_samples":False} if act=="regression" else {"original_samples":False, "weighted_samples":True} 
    weighted_sample_results = {flag:{} for flag in weighted_sample_flags}
    weighted_sample_targets = {flag:{} for flag in weighted_sample_flags}
    weighted_sample_best_alphas = {flag:{} for flag in weighted_sample_flags}
    weighted_sample_target_indices = {flag:{} for flag in weighted_sample_flags}
    ordered_outputs.update({flag:{} for flag in weighted_sample_flags})
    
    for flag in weighted_sample_flags:
        for key in targets:
            curr_targets = targets[key]
            _, best_alpha_outputs, output_targets, best_alphas, target_indices = \
                get_group_models_and_predictions(features, curr_targets, act=act,\
                                                 weighted_samples=weighted_sample_flags[flag],leave_one_out=leave_one_out)
            
            weighted_sample_best_alphas[flag][key] = best_alphas

            if act != "regression":
                best_alpha_outputs = np.minimum(5,np.maximum(1,np.round(best_alpha_outputs)))
            best_alpha_outputs = best_alpha_outputs.flatten()
            weighted_sample_results[flag][key] = best_alpha_outputs.tolist()
            weighted_sample_targets[flag][key] = output_targets.tolist()
            weighted_sample_target_indices[flag][key] = target_indices.tolist()
            ordered_outputs[flag][key] = best_alpha_outputs[np.argsort(target_indices)]
        
    return weighted_sample_results, weighted_sample_targets, weighted_sample_best_alphas, weighted_sample_target_indices

def plot_classifier_cm(classifier_results,targets):
    fig,ax = plt.subplots(2,4)
    for i,weight_flag in enumerate(classifier_results):
        for j,target in enumerate(targets):
            curr_outputs = classifier_results[weight_flag][target]
            curr_targets = targets[target]
            get_confusion_matrix_plot(curr_targets,curr_outputs,ax[i][j],target)
        
        ax[i][0].set_ylabel(f"{weight_flag}\nTrue label")
    plt.show()

def show_feature_correlations(features,output_prefix="plots/"):
    # correlation matrix plot
    feature_correlations = features.corr()
    print(feature_correlations.abs().sum(axis=0))
    fig,ax = plt.subplots()
    fig.set_size_inches(12.8,9.6)
    heatmap(feature_correlations,annot=True,fmt="0.2f",ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
    fig.savefig(output_prefix+"feature_correlation.pdf",bbox_inches="tight")

def get_np_features(df,feature_names):
    features = df[feature_names]
    np_features = features.to_numpy()
    features_mask = np.logical_not(np.isnan(np_features[:,0]))
    np_features = np.array([np_features[:,i][features_mask] for i in range(len(feature_names))]).astype(int).T
    return np_features, features_mask

def ordinal_p_values(features_dataframe, feature_names, human_ratings, sample_weights):
    np_features_no_colour, features_mask_no_colour = get_np_features(features_dataframe,feature_names)

    for i,name in enumerate(feature_names):
        print(i+1,name,end=" ")
    ss = StandardScaler()
    inputs = ss.fit_transform(np_features_no_colour)
    targets = human_ratings[features_mask_no_colour[:608]]
    # print(np.unique(targets))
    
    inputs,targets = reweight_classes_resample(inputs,targets,sample_weights)

    om = OrderedModel(targets,inputs,distr="logit").fit(method="lbfgs", disp=False, maxiter=1000)
    print(om.summary())
    
def expert_defined_evalutation(targets,np_features,features_mask,feature_names):
    model_types_num_experiments = {"dichotomies":100,
                                    "multinomial":1,
                                    "ordinal":1,
                                    "Random Forest":1
    }
    
    for model_type, num_experiments in model_types_num_experiments.items():
        print(f"{model_type=}")
        num_inner_experiments = 1

        evaluate_linear_model(inputs=np_features,targets=targets[features_mask[:608]],model_type=model_type,
                            path_prefix="plots/",feature_names=feature_names,num_experiments=num_experiments,
                            num_inner_experiments=num_inner_experiments)
        


def get_expert_results(targets,h_median_ratings,high_ratings_mask,target_key):
    for i, clusters in enumerate(["All Clusters","Clusters with >=10 raters"]):
        print("---------------------------------------------------")
        print(clusters)
        curr_targets = targets[target_key]
        
        curr_targs = curr_targets
        curr_ratings = h_median_ratings

        if i==1:
            curr_targs = curr_targs[high_ratings_mask]
            curr_ratings = curr_ratings[high_ratings_mask]
        
        plt.close()
        fig, ax = plt.subplots()
        get_confusion_matrix_plot(curr_targs,curr_ratings,ax,title="Domain Experts")
        
        plt.savefig(f"plots/CM_Expert_{clusters}.pdf",bbox_inches="tight",pad_inches=0)
        plt.close()


def get_CNN_results(targets,dataset_file_path,dataset_type,images_subfolder,model_path,target_key):
    
    with tf.device("/GPU:1"):
        classifier_outputs, classifier_targets, classifier_best_alphas, _ = \
            get_CNN_outputs(dataset_file_path, dataset_type, images_subfolder, model_path, {target_key:targets[target_key]})

    for i,weight_flag in enumerate(classifier_outputs):
        print("---------------------------------------------------")
        print(weight_flag)
        curr_outputs = classifier_outputs[weight_flag][target_key]
        curr_targets = classifier_targets[weight_flag][target_key]
        curr_best_alphas = classifier_best_alphas[weight_flag][target_key]
        print("curr_best_alphas")
        print(curr_best_alphas)
        print("curr_best_alphas_end")
        plt.close()
        save_output_histogram(curr_outputs,f"plots/hist_CNN_{weight_flag}.pdf",title="CNN")
        
        fig, ax = plt.subplots()
        get_confusion_matrix_plot(curr_targets,curr_outputs,ax,title="CNN")
        
        plt.savefig(f"plots/CM_CNN_{weight_flag}.pdf",bbox_inches="tight",pad_inches=0)
        

def get_backward_elimination_results(feature_names, feature_names_no_colour,feature_names_no_colour_no_rcondition,targets,hr,just_visualize=False):
    options = {
        "model_type":["multinomial","ordinal"],
        "feature_name":{"":feature_names,"_no_image_colour":feature_names_no_colour,"_no_image_colour_no_r_condition":feature_names_no_colour_no_rcondition},
        "training":{"":None,"_balanced_training":"balanced"},
        "testing":{"_balanced_testing":True,"":False}
        }

    # num_experiments = 100
    num_experiments = 1

    BE_path_prefix = "BE_results/"
    os.makedirs(BE_path_prefix,exist_ok=True)

    ##for non-dichotomies

    njobs = 5

    with ProcessPoolExecutor(max_workers=njobs) as ppe:
        futures=[]
        for model_type in options["model_type"]:
            model_type_text = "_"+model_type if model_type =="ordinal" else ""
            for f_text, f_names in options["feature_name"].items():
                np_features, features_mask = get_np_features(hr,f_names)
                for tr_weight_text,training_weight in options["training"].items():
                    for te_weight_text,balanced_testing in options["testing"].items():
                        print(f"BE{tr_weight_text}{te_weight_text}{f_text}_{model_type}")
                        backward_elimination_path = f"{BE_path_prefix}logistic{tr_weight_text}{te_weight_text}{f_text}{model_type_text}.pickle"
                        if not just_visualize:
                            futures.append(ppe.submit(backward_elimination,inputs=np_features,targets=targets["mean_combined"][features_mask[:608]],
                                            feature_names=f_names,output_path=backward_elimination_path,class_weight=training_weight,
                                            model_type=model_type,balanced_testing=balanced_testing,num_experiments=num_experiments))
                        else:
                            visualize_backward_elimination_results(backward_elimination_path,f_names,"plots/",f"BE{tr_weight_text}{te_weight_text}{f_text}{model_type_text}",model_type=model_type)
        for future in futures:
            future.result()

    #for dichotomies
    
    with ProcessPoolExecutor(max_workers=njobs) as ppe:
        
        futures=[]
        model_type="multinomial"
        for f_text, f_names in options["feature_name"].items():
            np_features, features_mask = get_np_features(hr,f_names)
            curr_targets = targets["mean_combined"][features_mask[:608]]
            dich_targets = get_dichotomy_labels(curr_targets)
            for dich_i,curr_dich_targets in enumerate(dich_targets):
                backward_elimination_path = f"{BE_path_prefix}logistic_dichotomy_{dich_i}{f_text}.pickle"
                if not just_visualize:
                    futures.append(ppe.submit(backward_elimination,inputs=np_features,targets=curr_dich_targets,
                                        feature_names=f_names,output_path=backward_elimination_path,class_weight="balanced"
                                        ,model_type=model_type,balanced_testing=True,num_experiments=num_experiments))
                else:
                    visualize_backward_elimination_results(backward_elimination_path,f_names,"plots/",f"BE_dichotomy{f_text}",model_type=model_type,dichotomy=True)
        for future in futures:
            future.result()
    

def main():
    
    model_path = sys.argv[1]
    images_subfolder = sys.argv[2]
    wealth_ratings_file_path = sys.argv[3]
    human_ratings_file_path = sys.argv[4]
    dataset_file_path = sys.argv[5]
    dataset_type = sys.argv[6]

    hr = pandas.read_csv(human_ratings_file_path)
    
    os.makedirs("plots",exist_ok=True)

    h_total_ratings = hr["Total_ratings"].to_numpy()
    h_median_ratings = hr["Median_rating"].to_numpy()
    feature_names = ["buildings_size", "roofing_material", "roofingcondition", "larger_buildings", \
                     "settlement_structure", "building_density", "greenery", "dominant_landuse", "image_colour",\
                     "road_surface_quality", "road_width", "roads_coverage", "vehicles_presence", "farm_sizes"]
    
    feature_names_no_colour = feature_names.copy()
    feature_names_no_colour.remove("image_colour")
    feature_names_no_colour_no_rcondition = feature_names_no_colour.copy()
    feature_names_no_colour_no_rcondition.remove("roofingcondition")

    features = hr[feature_names]
    
    #human ratings
    h_total_ratings = h_total_ratings[np.logical_not(np.isnan(h_total_ratings))]
    h_median_ratings = h_median_ratings[np.logical_not(np.isnan(h_median_ratings))]

    h_median_ratings = np.round(h_median_ratings)
    high_ratings_mask = np.nonzero(h_total_ratings>=10)

    #survey targets
    targets = read_survery_targets(wealth_ratings_file_path)

    np_features, features_mask = get_np_features(hr,feature_names)
    np_features_no_colour, features_mask_no_colour = get_np_features(hr,feature_names_no_colour)

    sample_weights = compute_sample_weight("balanced",y = targets["mean_combined"])
    
    target_key = "mean_combined"

    print("Feature correlation matrix ####################################################################################")
    show_feature_correlations(features)

    print("Making histograms for experts and DHS survey ##################################################################")
    save_output_histogram(targets["mean_combined"],"plots/hist_DHS.pdf",title="DHS Survey")
    save_output_histogram(h_median_ratings,"plots/hist_experts.pdf",title="Domain Experts")

    print("feature importance for random forest ##########################################################################")
    evaluate_linear_model(inputs=np_features_no_colour,targets=targets["mean_combined"][features_mask[:608]],model_type="Random Forest",
                          path_prefix="plots/",feature_names=feature_names_no_colour,num_experiments=100,
                          get_feature_importance=True)

    print("p_values using the ordinal classifier #########################################################################")
    ordinal_p_values(features_dataframe=hr, feature_names=feature_names_no_colour, human_ratings=h_median_ratings, sample_weights=sample_weights)
    
    print("evaluation using classifier with expert defined featuers ######################################################")
    expert_defined_evalutation(targets[target_key],np_features,features_mask,feature_names)

    print("results for domain experts #####################################################################################")
    get_expert_results(targets,h_median_ratings,high_ratings_mask,target_key)

    print("results for the CNN ############################################################################################")
    get_CNN_results(targets,dataset_file_path,dataset_type,images_subfolder,model_path,target_key)

    print("getting results for backward elimination #######################################################################")
    get_backward_elimination_results(feature_names, feature_names_no_colour,feature_names_no_colour_no_rcondition,targets,hr)
    get_backward_elimination_results(feature_names, feature_names_no_colour,feature_names_no_colour_no_rcondition,targets,hr,just_visualize=True)
    
if __name__ == "__main__":
    main()
