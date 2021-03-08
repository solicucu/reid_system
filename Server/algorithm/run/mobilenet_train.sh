#MODEL.DEVICE_ID can not update through config.yml,thus here update by opts
python ../main/train.py  --config_file '../configs/MobileNet.yml' MODEL.DEVICE "('cuda')" MODEL.DEVICE_ID "('0,1')"


