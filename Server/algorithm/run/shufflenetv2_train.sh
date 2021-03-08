#MODEL.DEVICE_ID can not update through config.yml,thus here update by opts
python ../main/train.py \
--config_file '../configs/shufflenetv2.yml' \
MODEL.DEVICE "('cuda')" \
MODEL.DEVICE_ID "('1')" \
#SOLVER.BASE_LR "(0.015)" \
#SOLVER.LR_MIN "(0.0001)" 
#SOLVER.LR_SCHEDULER_NAME "('CosineAnnealingLR')" \
#SOLVER.BASE_LR "(0.025)"
