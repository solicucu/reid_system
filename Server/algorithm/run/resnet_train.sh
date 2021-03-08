#MODEL.DEVICE_ID can not update through config.yml,thus here update by opts
python ../main/train.py \
--config_file '../configs/resnet.yml' \
MODEL.DEVICE "('cuda')" \
MODEL.DEVICE_ID "('1')" \
#SOLVER.LR_SCHEDULER_NAME "('CosineAnnealingLR')" \
#SOLVER.BASE_LR "(0.025)"

