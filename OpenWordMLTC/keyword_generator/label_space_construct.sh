CUDA_VISIBLE_DEVICES=0 python dynamic_GMM.py \
    --path ../../datasets \
    --keyphrase_dir deepseek_chat_label_50.txt \
    --task AAPD \
    --dynamic_iter 3000 \
    --cluster_size 84 \
    --model Qwen/Qwen3-8B \
    --batch_size 1 \
    --output_file init_labelspace.txt

CUDA_VISIBLE_DEVICES=0 python get_init_labelspace.py \
    --path ../../datasets \
    --data_dir deepseek_chat/init_labelspace.txt \
    --task AAPD \
    --lower_bound 0.80 \
    --model Qwen/Qwen3-8B \
    --output_dir deepseek_chat/init_label_space.txt
