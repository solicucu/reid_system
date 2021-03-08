#MODEL.DEVICE_ID can not update through config.yml,thus here update by opts
python ../main/train.py \
--config_file '../configs/ssnet.yml' \
MODEL.DEVICE "('cuda')" \
MODEL.DEVICE_ID "('6')" \
#SOLVER.LR_SCHEDULR_NAME "('CosineAnnealingLR')" \
#SOLVER.BASE_LR "(0.025)" \
#SOLVER.LR_MIN "(0.001)"
