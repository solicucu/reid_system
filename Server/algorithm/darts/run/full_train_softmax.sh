python ../train_search.py \
--device 'cuda' \
--device_ids '0' \
--num_instance 4 \
--batch_size 32 \
--max_epochs 120 \
--in_planes 30 \
--model_name 'fsnet' \
--loss_name 'softmax' \
--log_name 'log_fsnet_search_softmax.txt' \
--eval_period 10 \
--ckpt_period 5 \
--dataset_dir '/home/share/solicucu/data/' \
--output_dir '/home/share/solicucu/data/ReID/MobileNetReID/fsnet/' \
--ckpt_dir "checkpoints/fsnet_search/" \
#--pretrained '/home/share/solicucu/data/ReID/MobileNetReID/darts/darts1/checkpoints/fsnet/'
