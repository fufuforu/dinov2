
# torchrun --nproc_per_node 2 --master_port 12346 tools/train.py -c configs/rtdetr_lmot_det/rtdetr_r50vd_lmot_light_bs4.yml --auto_resume

# torchrun --nproc_per_node 2 --master_port 12346 tools/train.py -c configs/rtdetr_lmot_det/rtdetr_r50vd_lmot_dark_ns_isp_bs4.yml --auto_resume

# torchrun --nproc_per_node 2 --master_port 12346 tools/train.py -c configs/rtdetr_lmot_det/rtdetr_r50vd_lmot_dark_bs4.yml --auto_resume

# torchrun --nproc_per_node 2 --master_port 12346 tools/train.py -c configs/rtdetr_lmot_det/rtdetr_r50vd_lmot_dark_colorJit_bs4.yml --auto_resume

# torchrun --nproc_per_node 2 --master_port 12346 tools/train.py -c configs/rtdetr_lmot_det/rtdetr_r50vd_lmot_dark_ns_isp_colorJit_bs4.yml --auto_resume

# torchrun --nproc_per_node 2 --master_port 12346 tools/train.py -c configs/rtdetr_lmot_det/rtdetr_r50vd_lmot_dark_BackboneLr1e-3_bs4.yml --auto_resume

torchrun --nproc_per_node 2 --master_port 12346 tools/train.py -c configs/rtdetr_lmot_tracking_v2/rtdetr_mot_r50vd_lmot_dark_ns_isp_bs2.yml --auto_resume
torchrun --nproc_per_node 2 --master_port 12346 tools/train.py -c configs/rtdetr_lmot_tracking_v2/rtdetr_mot_r50vd_lmot_light_isp_bs2.yml --auto_resume


# torchrun --nproc_per_node 2 --master_port 12345 tools/train.py -c configs/rtdetr_lmot_tracking_v2/rtdetr_mot_r50vd_lmot_dark_bs2.yml --auto_resume
# torchrun --nproc_per_node 2 --master_port 12345 tools/train.py -c configs/rtdetr_lmot_tracking_v2/rtdetr_mot_r50vd_lmot_light_bs2.yml --auto_resume

