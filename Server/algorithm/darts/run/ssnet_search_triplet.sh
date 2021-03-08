python ../train_search.py \
--device 'cuda' \
--device_ids '1' \
--num_instance 4 \
--batch_size 32 \
--in_planes 60 \
--max_epochs 120 \
--base_lr 0.025 \
--lr_min 0.001 \
--model_name 'ssnet' \
--loss_name 'softmax_triplet' \
--log_name 'log_ssnet_search_atten_pcbneck_softmax_triplet.txt' \
--eval_period 40 \
--ckpt_period 40 \
--dataset_dir '/home/share/solicucu/data/' \
--output_dir '/home/share/solicucu/data/ReID/MobileNetReID/ssnet/' \
--ckpt_dir "checkpoints/search_atten_pcbneck_softmax_triplet/"  
#--pretrained '/home/share/solicucu/data/ReID/MobileNetReID/ssnet/checkpoints/search_atten_pcbneck_softmax_triplet/'
