# train
CUDA_VISIBLE_DEVICES=0 python zero-shot-AAPD.py \
    --path ../../datasets \
    --data_dir train_texts_split_50.txt \
    --keyphrase_dir deepseek_chat_label_50.txt \
    --task AAPD \
    --dynamic_iter 3000 \
    --model MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33

# test
CUDA_VISIBLE_DEVICES=0 python zero-shot_AAPD_test.py \
    --path ../../datasets \
    --data_dir deepseek_chat/init_label_space.txt \
    --keyphrase_dir keyphrase_candidate/deepseek_chat_label_test_50.txt \
    --task AAPD \
    --result_dir deepseek_chat/test_performance

CUDA_VISIBLE_DEVICES=0 python model_eval.py \
    --path ../../datasets \
    --data_dir deepseek_chat/test_performance \
    --keyphrase_dir deepseek_chat_label_test_50.txt \
    --task AAPD \
    --test_size 1000 \
    --output_dir deepseek_chat/test_performance/MLClass_result.txt
