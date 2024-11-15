export CUDA_VISIBLE_DEVICES="1"
# detection
python tools/train.py -c configs/rtdetr_lmot_det/rtdetr_r50vd_lmot_light_isp.yml -r output/rtdetr_lmot_det/rtdetr_r50vd_lmot_light_isp/checkpoint0019.pth --run_type eval --eval_data val


# tracking
python tools/train.py -c configs/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_light_isp.yml -r output/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_light_isp/checkpoint0004.pth --run_type track --eval_data val

torchrun --nproc_per_node=2 --master_port=12345 tools/train.py -c configs/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_light_isp.yml -r output/rtdetr_lmot_tracking/rtdetr_mot_r50vd_lmot_light_isp/checkpoint0004.pth --run_type track --eval_data val