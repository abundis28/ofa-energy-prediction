import os
import subprocess as sp

directory = "/media/ssd/data/Vitis-AI-2.0/Vitis-AI/andres_diss/21-22_CE901-SU_abundis_correa_andres/imagenet_1k_128/"

i = 0
for folder in os.listdir(directory+"train"):
    if i < 128:
        f = os.path.join(directory+"train", folder)
        sp.run(["/media/ssd/data/Vitis-AI-2.0/Vitis-AI/andres_diss/21-22_CE901-SU_abundis_correa_andres/extract.sh", f, directory+"train"])
    i += 1

i = 0
for folder in os.listdir(directory+"val"):
    if i < 128:
        f = os.path.join(directory+"val", folder)
        sp.run(["/media/ssd/data/Vitis-AI-2.0/Vitis-AI/andres_diss/21-22_CE901-SU_abundis_correa_andres/extract.sh", f, directory+"val"])
    i += 1
