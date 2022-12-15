import os
import pandas as pd
import ast

pwd_path = "/media/ssd/data/Vitis-AI-2.0/Vitis-AI/andres_diss/21-22_CE901-SU_abundis_correa_andres/"
networks_df = pd.read_csv(pwd_path+"networks.csv")

depth_0 = []
depth_1 = []
depth_2 = []
depth_3 = []
depth_4 = []
image_s_list = []
different_len_depth = 0
for index, row in networks_df.iterrows():
    arch_nn = ast.literal_eval(row["Architecture"])
    depth_0.append(arch_nn['d'][0])
    depth_1.append(arch_nn['d'][1])
    depth_2.append(arch_nn['d'][2])
    depth_3.append(arch_nn['d'][3])
    depth_4.append(arch_nn['d'][4])
    image_s_list.append(arch_nn['image_size'])

print("Diff to len 5: {}".format(different_len_depth))

networks_df.drop(["Architecture"], axis=1, inplace=True)
networks_df["D0"] = depth_0
networks_df["D1"] = depth_1
networks_df["D2"] = depth_2
networks_df["D3"] = depth_3
networks_df["D4"] = depth_4
networks_df["ImageSize"] = image_s_list

import re

file_numbers_list = []
regex = re.compile(r'\d+')

for model in os.listdir(pwd_path+"final_models_1kval/"):
    nums = [int(x) for x in regex.findall(model)]
    file_numbers_list.append(nums[2])

missing_models = sorted(set(range(1,5001)) - set(file_numbers_list))

networks_df.drop(networks_df.loc[networks_df['Index'].isin(missing_models)].index, axis=0, inplace=True)

stats_list = []
for model in os.listdir(pwd_path+"final_models_1kval"):
    file_stats = os.stat(pwd_path+"final_models_1kval/"+model)
    stats_list.append(file_stats.st_size / (1024 * 1024))

networks_df["ModelSize"] = stats_list

# print(networks_df.head(20))

networks_df.to_csv(pwd_path+"final_info.csv")