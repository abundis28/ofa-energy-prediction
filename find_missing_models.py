import os
import re

file_numbers_list = []
regex = re.compile(r'\d+')

for model in os.listdir("/media/ssd/data/Vitis-AI-2.0/Vitis-AI/andres_diss/21-22_CE901-SU_abundis_correa_andres/temp_file_holder"):
    nums = [int(x) for x in regex.findall(model)]
    file_numbers_list.append(nums[2])
missing_models = [2509, 2510, 2511, 2512, 2513, 2514, 2515, 2516, 2517, 2519, 2520, 2521, 2522, 2523, 2524, 2525, 2526, 2527, 2528, 2529, 2530, 2531, 2532, 2533, 2534, 2537, 2538, 2539, 2540, 2541, 2542, 2543, 2544, 2545, 2546, 2547, 2548, 2550, 2551, 2552, 2553, 2554, 2560, 2561, 2593, 2597, 2638, 2640, 2641, 2643, 2644, 2645, 2646, 2647, 2649, 2662, 2663, 2664, 2665, 2666, 2667, 2668, 2669, 2670, 2671, 2672, 2673, 2674, 2675, 2676, 2678, 2679, 2681, 2682, 2683, 2684, 2685, 2686, 2687, 2688, 2689, 2690, 2691, 2692, 2693, 2695, 2696, 2698, 2699, 2700, 2701, 2702, 2703, 2789, 2793, 2795, 2797, 2800, 2801, 2804, 2807]
missing_models = sorted(set(range(3000,3500)) - set(file_numbers_list) - set(missing_models))

print(len(missing_models))
print(missing_models)