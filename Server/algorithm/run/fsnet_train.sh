#MODEL.DEVICE_ID can not update through config.yml,thus here update by opts
python ../main/train.py \
--config_file '../configs/fsnet.yml' \
MODEL.DEVICE "('cuda')" \
MODEL.DEVICE_ID "('0')" \
#SOLVER.LR_SCHEDULER_NAME "('CosineAnnealingLR')" \
#SOLVER.BASE_LR "(0.025)" \
#SOLVER.LR_MIN "(0.001)"
