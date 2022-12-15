from subprocess import call, Popen

pwd_path = "/media/ssd/data/Vitis-AI-2.0/Vitis-AI/andres_diss/21-22_CE901-SU_abundis_correa_andres/"

start_file_id = 5000
end_file_id = 5002

missing_models = [2509, 2510, 2511, 2512, 2513, 2514, 2515, 2516, 2517, 2519, 2520, 2521, 2522, 2523, 2524, 2525, 2526, 2527, 2528, 2529, 2530, 2531, 2532, 2533, 2534, 2537, 2538, 2539, 2540, 2541, 2542, 2543, 2544, 2545, 2546, 2547, 2548, 2550, 2551, 2552, 2553, 2554, 2560, 2561, 2593, 2597, 2638, 2640, 2641, 2643, 2644, 2645, 2646, 2647, 2649, 2662, 2663, 2664, 2665, 2666, 2667, 2668, 2669, 2670, 2671, 2672, 2673, 2674, 2675, 2676, 2678, 2679, 2681, 2682, 2683, 2684, 2685, 2686, 2687, 2688, 2689, 2690, 2691, 2692, 2693, 2695, 2696, 2698, 2699, 2700, 2701, 2702, 2703, 2789, 2793, 2795, 2797, 2800, 2801, 2804, 2807, 3148, 3151, 3153, 3155, 3156, 3157, 3158, 3162]
commands = []
for i in range(start_file_id, end_file_id):
    if i in missing_models:
        continue
    commands.append("cp "+pwd_path+"final_models_1kval/resnet50_fp2000_"+str(i)+".xmodel "+pwd_path+"temp_file_holder")

from functools import partial
from multiprocessing.dummy import Pool

parallel_tasks = 8
pool = Pool(parallel_tasks)
for returncode in pool.imap(partial(call, shell=True), commands):
    if returncode != 0:
       print("%d command failed: %d" % (i, returncode))

