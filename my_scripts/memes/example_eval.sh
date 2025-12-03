

export TOKENIZERS_PARALLELISM=false

#dataset_list=('harmc' 'harmp' 'multioff')
dataset_list=('PrideMM')
for dataset in ${dataset_list[@]}; do
    model="QWen/QWen2-VL-2B-Instruct"
    python ./src/llamafactory/custom/hm_inference.py --model_path "./checkpoints/qwen2_vl-2b/lora/pridemm/2025-01-17_baseline" \
        --base_model_path $model --processor_path $model --dataset $dataset  --data_split 'val test' --batch_size 1

    model="QWen/QWen2-VL-7B-Instruct"
    python ./src/llamafactory/custom/hm_inference.py --model_path "./checkpoints/qwen2_vl-7b/lora/pridemm/2025-01-17_baseline" \
        --base_model_path $model --processor_path $model --dataset $dataset  --data_split 'val test' --batch_size 1

done