# build ofa resnet50 design space
from ofa.model_zoo import ofa_net
from subprocess import call, Popen

ofa_network = ofa_net('ofa_resnet50', pretrained=True)

#  build accuracy predictor
import torch
from ofa.nas.accuracy_predictor import ResNetArchEncoder
from ofa.nas.accuracy_predictor import AccuracyPredictor 
from ofa.utils import download_url

image_size_list = [128, 144, 160, 176, 192, 224, 240, 256]
arch_encoder = ResNetArchEncoder(
	image_size_list=image_size_list, depth_list=ofa_network.depth_list, expand_list=ofa_network.expand_ratio_list,
    width_mult_list=ofa_network.width_mult_list, base_depth_list=ofa_network.BASE_DEPTH_LIST)

#ofa/utils/common_tools.py
acc_predictor_checkpoint_path = download_url(
    'https://hanlab.mit.edu/files/OnceForAll/tutorial/ofa_resnet50_acc_predictor.pth',
    model_dir='~/.ofa/',
)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
acc_predictor = AccuracyPredictor(arch_encoder, 400, 3,
                                  checkpoint_path=acc_predictor_checkpoint_path, device=device)

# print('The accuracy predictor is ready!')
# print(acc_predictor)

from ofa.nas.efficiency_predictor import ResNet50FLOPsModel

efficiency_predictor = ResNet50FLOPsModel(ofa_network)

import argparse
parser = argparse.ArgumentParser()

args = parser.parse_args(args=[])
args.arch_mutate_prob = 0.1 
args.resolution_mutate_prob = 0.5 
args.population_size = 528
args.max_time_budget = 50
args.parent_ratio = 0.25 
args.mutation_ratio = 0.5
starting_number = 4342

from ofa_altered.nas.search_algorithm import EvolutionFinder

evolution_finder = EvolutionFinder(efficiency_predictor, acc_predictor, **args.__dict__)

# get best subnet with constraint(Mflops)
# constraint : Mega flops
sorted_population = evolution_finder.run_evolution_search(constraint=2000, verbose=True)

print("CREATING pth AND pickle FILES")

path_pwd = "/workspace/andres_diss/21-22_CE901-SU_abundis_correa_andres/"

import pickle
from csv import writer

i = starting_number
pairings_list = []
with open('networks.csv', 'a') as file_obj:
    writer_object = writer(file_obj)
    for nn in sorted_population:
        net_name = "resnet50_fp2000_"+str(i)
        with open(path_pwd+'/models/net_config_'+net_name+'.pickle', 'wb') as handle:
            pickle.dump(nn[1], handle, protocol=pickle.HIGHEST_PROTOCOL)
        ofa_network.set_active_subnet(**nn[1])
        subnet = ofa_network.get_active_subnet(preserve_weight=True)
        pairings_list.append((net_name, nn[1]))
        torch.save(subnet.state_dict(), path_pwd+'/models/'+net_name+'.pth', _use_new_zipfile_serialization=False)
        nn = list(nn)
        nn.insert(0,i)
        writer_object.writerow(nn)
        i += 1
    file_obj.close()

print("EXPORTING")

import time
start_time = time.time()

#### SINGLE EXPORT TASK
# Popen(path_pwd+"export_pth.sh {} {} {}".format(pairings_list[0][0], str(pairings_list[0][1]["image_size"]), str(0)), shell=True).wait()

# #### SEQUENTIAL EXPORT
# for i, pair in enumerate(pairings_list):
#     Popen(path_pwd+"export_pth.sh {} {} {}".format(pair[0], str(pair[1]["image_size"]), str(i)), shell=True).wait()

#### PARALLEL EXPORT
from functools import partial
from multiprocessing.dummy import Pool

commands = []
i = starting_number
for pair in pairings_list:
    commands.append(path_pwd+"export_pth.sh {} {} {}".format(pair[0], str(pair[1]["image_size"]), str(i)))
    i+=1

parallel_tasks = 4
pool = Pool(parallel_tasks)
i = starting_number
for returncode in pool.imap(partial(call, shell=True), commands):
    if returncode != 0:
       print("%d command failed: %d" % (i, returncode))
    i+=1
    # if i % parallel_tasks == 0 and i != 0:
    #     print("\n\n\t\t &&&&&&&&&& DONE BATCH #{} &&&&&&&&&&\n\n".format(i/parallel_tasks)) 
    #     Popen("rm -r "+path_pwd+"compiled_models/*", shell=True)

end_time = time.time()
print("Total time elapsed: {}".format(end_time-start_time))

Popen("rm -r "+path_pwd+"models/*", shell=True)
Popen("rm -r "+path_pwd+"compiled_models/*", shell=True)
Popen("mv "+path_pwd+"compiled/* "+path_pwd+"final_models_1kval/", shell= True)
