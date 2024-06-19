#!/bin/bash
MODEL_PATH=/tmp/pretrain_model

echo $NUM_GPUS
echo $NUM_NODES 
echo $HOSTFILE
echo $MASTER_ADDR

python -m torch.distributed.run \
    --nproc_per_node $NUM_GPUS \
    --nnodes $NUM_NODES \
    --node_rank $NODE_INDEX \
    --master_addr $MASTER_ADDR \
    --master_port 9901 \
    ../../src/train_bash.py \
    --deepspeed ../deepspeed/ds_z3_offload_config.json \
    --stage sft \
    --do_train \
    --model_name_or_path $MODEL_PATH \
    --dataset train \
    --dataset_dir /opt/ml/input/data/train \
    --template llama3 \
    --finetuning_type full \
    --output_dir $OUTPUT_DIR \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --warmup_steps 100 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --evaluation_strategy steps \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --val_size 0.01 \
    --ddp_timeout 180000000 \
    --plot_loss \
    --bf16 \
    --max_seq_len 1024 \
    --report_to wandb
