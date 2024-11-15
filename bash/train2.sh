# torchrun --nproc_per_node 2 --master_port 12346 tools/train.py -c configs/rtdetr_lmot_tracking_v2/rtdetr_mot_r50vd_lmot_dark_ns_isp_bs2.yml --auto_resume
torchrun --nproc_per_node 2 --master_port 12346 tools/train.py -c configs/rtdetr_lmot_tracking_v2/rtdetr_mot_r50vd_lmot_light_isp_bs2.yml --auto_resume
