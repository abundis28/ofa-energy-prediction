import os
from ofa.utils import download_url
from subprocess import Popen

imagenet_data_path = "/workspace/andres_diss/21-22_CE901-SU_abundis_correa_andres/dataset"
download_url('https://hanlab.mit.edu/files/OnceForAll/ofa_cvpr_tutorial/imagenet_1k.zip', 
             model_dir=imagenet_data_path)
Popen("cd dataset && unzip imagenet_1k 1>/dev/null", shell=True).wait()
