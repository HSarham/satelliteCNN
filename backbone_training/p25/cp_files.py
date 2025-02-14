import os
import pandas as pd
import shutil

trainset_file_list = ['TZA_rect_complete_TZA_surr_light_train_2023_09_lyon_mobilenetv2_2023-09-05--13-59-26_TZArs_Epochs_1_LR_0001_BS_100_L2reg_01_p25_TZA_rs_.parquet',
	'TZA_rect_complete_TZA_surr_light_validation_2023_09_lyon_mobilenetv2_2023-09-05--13-59-26_TZArs_Epochs_1_LR_0001_BS_100_L2reg_01_p25_TZA_rs_.parquet',
	'TZA_rect_complete_TZA_surr_light_test_2023_09_lyon_mobilenetv2_2023-09-05--13-59-26_TZArs_Epochs_1_LR_0001_BS_100_L2reg_01_p25_TZA_rs_.parquet']

data_folder = "/nfs/home/mixed_methods/Mixed_Methods/Sentinel_p25_224px_images/"
output_folder = "/nfs/home/mixed_methods/Mixed_Methods/Sentinel_p25_224px_images/output/"

os.makedirs(output_folder,exist_ok=True)

def main():
    for df_file_name in trainset_file_list:
        df = pd.read_parquet(df_file_name)
        print(len(df))
        continue
        for i,file_path in enumerate(df["Image_File_Name"]):
            print(i,end="\r")
            # file_name = file_path[file_path.rfind("/")+1:]
            dir = data_folder + "output/" + file_path[:file_path.rfind("/")+1]
            os.makedirs(dir,exist_ok=True)
            src = data_folder+file_path
            dst = output_folder+file_path
            shutil.copyfile(src,dst)

if __name__ == "__main__":
    main()