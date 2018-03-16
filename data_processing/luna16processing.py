import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
import pandas as pd

luna_subset_path = '../data/luna16data/subset0/'
luna_path = '../data/CSVFILES/'

file_list = glob(luna_subset_path+"*.mhd")

#####################
#
# Helper function to get rows in data frame associated
# with each file


def get_filename(case):
    global file_list
    for f in file_list:
        if case in f:
            return f


# The locations of the nodes
df_node = pd.read_csv(luna_path+"annotations.csv")
df_node["file"] = df_node["seriesuid"].apply(get_filename)
df_node = df_node.dropna()

#####
#
# Looping over the image files
#
fcount = 0
for img_file in file_list:
    print("Getting mask for image file %s" % img_file.replace(luna_subset_path, ""))
    mini_df = df_node[df_node["file"] == img_file]  # get all nodules associate with file
    if len(mini_df) > 0:       # some files may not have a nodule--skipping those
        biggest_node = np.argsort(mini_df["diameter_mm"].values)[-1]   # just using the biggest node
        node_x = mini_df["coordX"].values[biggest_node]
        node_y = mini_df["coordY"].values[biggest_node]
        node_z = mini_df["coordZ"].values[biggest_node]
        diam = mini_df["diameter_mm"].values[biggest_node]