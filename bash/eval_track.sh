# export CUDA_VISIBLE_DEVICES="1"
# # detection
# python tools/train.py -c configs/rtdetr_lmot_det/rtdetr_r50vd_lmot_light_isp.yml -r output/rtdetr_lmot_det/rtdetr_r50vd_lmot_light_isp/checkpoint0019.pth --run_type det --eval_data val


# # tracking
# python tools/train.py -c configs/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_light_isp.yml -r output/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_light_isp/checkpoint0014_converted.pth --run_type track --eval_data val

# torchrun --nproc_per_node=2 --master_port=12345 tools/train.py -c configs/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_dark_bs2.yml -r output/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_dark_bs2_g2/checkpoint0009.pth --run_type track --eval_data val
# torchrun --nproc_per_node=2 --master_port=12345 tools/train.py -c configs/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_dark_bs2.yml -r output/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_dark_bs2_g2/checkpoint0019.pth --run_type track --eval_data val

# torchrun --nproc_per_node=2 --master_port=12345 tools/train.py -c configs/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_dark_ns_isp_bs2.yml -r output/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_dark_ns_isp_bs2_g2/checkpoint0009.pth --run_type track --eval_data val
# torchrun --nproc_per_node=2 --master_port=12345 tools/train.py -c configs/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_dark_ns_isp_bs2.yml -r output/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_dark_ns_isp_bs2_g2/checkpoint0019.pth --run_type track --eval_data val



# torchrun --nproc_per_node=2 --master_port=12345 tools/train.py -c configs/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_light_isp_bs2.yml -r output/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_light_isp_bs2_g2/checkpoint0019.pth --run_type track --eval_data val
# torchrun --nproc_per_node=2 --master_port=12345 tools/train.py -c configs/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_light_isp_bs2.yml -r output/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_light_isp_bs2_g2/checkpoint0009.pth --run_type track --eval_data val

# torchrun --nproc_per_node=2 --master_port=12345 tools/train.py -c configs/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_light_isp_TuneCoco_bs2.yml -r output/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_light_isp_TuneCoco_bs2_g2/checkpoint0019.pth --run_type track --eval_data val
# torchrun --nproc_per_node=2 --master_port=12345 tools/train.py -c configs/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_light_isp_TuneCoco_bs2.yml -r output/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_light_isp_TuneCoco_bs2_g2/checkpoint0009.pth --run_type track --eval_data val


# torchrun --nproc_per_node=2 --master_port=12345 tools/train.py -c configs/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_light_bs2.yml -r output/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_light_bs2_g2/checkpoint0019.pth --run_type track --eval_data val
# torchrun --nproc_per_node=2 --master_port=12345 tools/train.py -c configs/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_light_bs2.yml -r output/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_light_bs2_g2/checkpoint0009.pth --run_type track --eval_data val


# torchrun --nproc_per_node=2 --master_port=12345 tools/train.py -c configs/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_light_TuneCoco_bs2.yml -r output/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_light_TuneCoco_bs2_g2/checkpoint0019.pth --run_type track --eval_data val
# torchrun --nproc_per_node=2 --master_port=12345 tools/train.py -c configs/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_light_TuneCoco_bs2.yml -r output/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_light_TuneCoco_bs2_g2/checkpoint0009.pth --run_type track --eval_data val






# torchrun --nproc_per_node=2 --master_port=12345 tools/train.py -c configs/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_dark_bs2.yml -r output/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_dark_bs2_g2/checkpoint0009.pth --run_type track --eval_data test
# torchrun --nproc_per_node=2 --master_port=12345 tools/train.py -c configs/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_dark_bs2.yml -r output/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_dark_bs2_g2/checkpoint0019.pth --run_type track --eval_data test

# torchrun --nproc_per_node=2 --master_port=12345 tools/train.py -c configs/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_dark_ns_isp_bs2.yml -r output/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_dark_ns_isp_bs2_g2/checkpoint0009.pth --run_type track --eval_data test
# torchrun --nproc_per_node=2 --master_port=12345 tools/train.py -c configs/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_dark_ns_isp_bs2.yml -r output/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_dark_ns_isp_bs2_g2/checkpoint0019.pth --run_type track --eval_data test



# torchrun --nproc_per_node=2 --master_port=12345 tools/train.py -c configs/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_light_isp_bs2.yml -r output/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_light_isp_bs2_g2/checkpoint0019.pth --run_type track --eval_data test
# torchrun --nproc_per_node=2 --master_port=12345 tools/train.py -c configs/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_light_isp_bs2.yml -r output/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_light_isp_bs2_g2/checkpoint0009.pth --run_type track --eval_data test

# torchrun --nproc_per_node=2 --master_port=12345 tools/train.py -c configs/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_light_isp_TuneCoco_bs2.yml -r output/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_light_isp_TuneCoco_bs2_g2/checkpoint0019.pth --run_type track --eval_data test
# torchrun --nproc_per_node=2 --master_port=12345 tools/train.py -c configs/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_light_isp_TuneCoco_bs2.yml -r output/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_light_isp_TuneCoco_bs2_g2/checkpoint0009.pth --run_type track --eval_data test


# torchrun --nproc_per_node=2 --master_port=12345 tools/train.py -c configs/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_light_bs2.yml -r output/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_light_bs2_g2/checkpoint0019.pth --run_type track --eval_data test
# torchrun --nproc_per_node=2 --master_port=12345 tools/train.py -c configs/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_light_bs2.yml -r output/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_light_bs2_g2/checkpoint0009.pth --run_type track --eval_data test


# torchrun --nproc_per_node=2 --master_port=12345 tools/train.py -c configs/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_light_TuneCoco_bs2.yml -r output/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_light_TuneCoco_bs2_g2/checkpoint0019.pth --run_type track --eval_data test
# torchrun --nproc_per_node=2 --master_port=12345 tools/train.py -c configs/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_light_TuneCoco_bs2.yml -r output/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_light_TuneCoco_bs2_g2/checkpoint0009.pth --run_type track --eval_data test





# torchrun --nproc_per_node=2 --master_port=12345 tools/train.py -c configs/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_light_isp_bs2.yml -r output/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_light_isp_bs2_g2/checkpoint0019.pth --run_type track --eval_data test
# torchrun --nproc_per_node=2 --master_port=12345 tools/train.py -c configs/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_dark_bs2.yml -r output/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_dark_bs2_g2/checkpoint0019.pth --run_type track --eval_data test





torchrun --nproc_per_node=2 tools/train.py -c configs/rtdetr_lmot_tracking_v2/rtdetr_mot_r50vd_lmot_light_isp_bs2.yml -r output/rtdetr_lmot_tracking_v2/rtdetr_mot_r50vd_lmot_light_isp_bs2_g2/checkpoint0009.pth --run_type track --eval_data val
torchrun --nproc_per_node=2 tools/train.py -c configs/rtdetr_lmot_tracking_v2/rtdetr_mot_r50vd_lmot_light_isp_bs2.yml -r output/rtdetr_lmot_tracking_v2/rtdetr_mot_r50vd_lmot_light_isp_bs2_g2/checkpoint0004.pth --run_type track --eval_data val
torchrun --nproc_per_node=2 tools/train.py -c configs/rtdetr_lmot_tracking_v2/rtdetr_mot_r50vd_lmot_light_bs2.yml -r output/rtdetr_lmot_tracking_v2/rtdetr_mot_r50vd_lmot_light_bs2_g2/checkpoint0009.pth --run_type track --eval_data val
torchrun --nproc_per_node=2 tools/train.py -c configs/rtdetr_lmot_tracking_v2/rtdetr_mot_r50vd_lmot_light_bs2.yml -r output/rtdetr_lmot_tracking_v2/rtdetr_mot_r50vd_lmot_light_bs2_g2/checkpoint0004.pth --run_type track --eval_data val

torchrun --nproc_per_node=2 tools/train.py -c configs/rtdetr_lmot_tracking_v2/rtdetr_mot_r50vd_lmot_dark_ns_isp_bs2.yml -r output/rtdetr_lmot_tracking_v2/rtdetr_mot_r50vd_lmot_dark_ns_isp_bs2_g2/checkpoint0004.pth --run_type track --eval_data val
torchrun --nproc_per_node=2 tools/train.py -c configs/rtdetr_lmot_tracking_v2/rtdetr_mot_r50vd_lmot_dark_ns_isp_bs2.yml -r output/rtdetr_lmot_tracking_v2/rtdetr_mot_r50vd_lmot_dark_ns_isp_bs2_g2/checkpoint0009.pth --run_type track --eval_data val
torchrun --nproc_per_node=2 tools/train.py -c configs/rtdetr_lmot_tracking_v2/rtdetr_mot_r50vd_lmot_dark_bs2.yml -r output/rtdetr_lmot_tracking_v2/rtdetr_mot_r50vd_lmot_dark_bs2_g2/checkpoint0009.pth --run_type track --eval_data val
torchrun --nproc_per_node=2 tools/train.py -c configs/rtdetr_lmot_tracking_v2/rtdetr_mot_r50vd_lmot_dark_bs2.yml -r output/rtdetr_lmot_tracking_v2/rtdetr_mot_r50vd_lmot_dark_bs2_g2/checkpoint0004.pth --run_type track --eval_data val


torchrun --nproc_per_node=2 tools/train.py -c configs/rtdetr_lmot_tracking_v2/rtdetr_mot_r50vd_lmot_light_isp_bs2.yml -r output/rtdetr_lmot_tracking_v2/rtdetr_mot_r50vd_lmot_light_isp_bs2_g2/checkpoint0009.pth --run_type track --eval_data eval_train
torchrun --nproc_per_node=2 tools/train.py -c configs/rtdetr_lmot_tracking_v2/rtdetr_mot_r50vd_lmot_light_isp_bs2.yml -r output/rtdetr_lmot_tracking_v2/rtdetr_mot_r50vd_lmot_light_isp_bs2_g2/checkpoint0004.pth --run_type track --eval_data eval_train
torchrun --nproc_per_node=2 tools/train.py -c configs/rtdetr_lmot_tracking_v2/rtdetr_mot_r50vd_lmot_light_bs2.yml -r output/rtdetr_lmot_tracking_v2/rtdetr_mot_r50vd_lmot_light_bs2_g2/checkpoint0009.pth --run_type track --eval_data eval_train
torchrun --nproc_per_node=2 tools/train.py -c configs/rtdetr_lmot_tracking_v2/rtdetr_mot_r50vd_lmot_light_bs2.yml -r output/rtdetr_lmot_tracking_v2/rtdetr_mot_r50vd_lmot_light_bs2_g2/checkpoint0004.pth --run_type track --eval_data eval_train

torchrun --nproc_per_node=2 tools/train.py -c configs/rtdetr_lmot_tracking_v2/rtdetr_mot_r50vd_lmot_dark_ns_isp_bs2.yml -r output/rtdetr_lmot_tracking_v2/rtdetr_mot_r50vd_lmot_dark_ns_isp_bs2_g2/checkpoint0004.pth --run_type track --eval_data eval_train
torchrun --nproc_per_node=2 tools/train.py -c configs/rtdetr_lmot_tracking_v2/rtdetr_mot_r50vd_lmot_dark_ns_isp_bs2.yml -r output/rtdetr_lmot_tracking_v2/rtdetr_mot_r50vd_lmot_dark_ns_isp_bs2_g2/checkpoint0009.pth --run_type track --eval_data eval_train
torchrun --nproc_per_node=2 tools/train.py -c configs/rtdetr_lmot_tracking_v2/rtdetr_mot_r50vd_lmot_dark_bs2.yml -r output/rtdetr_lmot_tracking_v2/rtdetr_mot_r50vd_lmot_dark_bs2_g2/checkpoint0009.pth --run_type track --eval_data eval_train
torchrun --nproc_per_node=2 tools/train.py -c configs/rtdetr_lmot_tracking_v2/rtdetr_mot_r50vd_lmot_dark_bs2.yml -r output/rtdetr_lmot_tracking_v2/rtdetr_mot_r50vd_lmot_dark_bs2_g2/checkpoint0004.pth --run_type track --eval_data eval_train





