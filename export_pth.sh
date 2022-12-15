#! /bin/bash

python3 /workspace/examples/DPUCZDX8G/ofa_resnet50/ofa_quant.py --model_name $1 --image_size $2 --quant_mode calib --fast_finetune --model_dir /workspace/andres_diss/21-22_CE901-SU_abundis_correa_andres/models/ --data_dir /workspace/andres_diss/21-22_CE901-SU_abundis_correa_andres/imagenet_1k --nn_num=$3
python3 /workspace/examples/DPUCZDX8G/ofa_resnet50/ofa_quant.py --model_name $1 --image_size $2 --quant_mode test --fast_finetune --model_dir /workspace/andres_diss/21-22_CE901-SU_abundis_correa_andres/models/ --data_dir /workspace/andres_diss/21-22_CE901-SU_abundis_correa_andres/imagenet_1k --nn_num=$3
python3 /workspace/examples/DPUCZDX8G/ofa_resnet50/ofa_quant.py --model_name $1 --image_size $2 --quant_mode test --subset_len 1 --batch_size=1 --deploy --model_dir /workspace/andres_diss/21-22_CE901-SU_abundis_correa_andres/models/ --data_dir /workspace/andres_diss/21-22_CE901-SU_abundis_correa_andres/imagenet_1k --nn_num=$3
vai_c_xir -x /workspace/andres_diss/21-22_CE901-SU_abundis_correa_andres/compiled_models/quantize_result_$3/ResNets_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json -o compiled -n $1
