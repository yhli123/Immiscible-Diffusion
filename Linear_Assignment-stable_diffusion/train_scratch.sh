export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="imagenet-1k"

accelerate config
accelerate launch --main_process_port 29600 --mixed_precision="fp16" --multi_gpu --num_processes 8 conditional_scratch_train_sd.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --resolution=256 --center_crop --random_flip \
  --train_batch_size=256 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --max_train_steps=20000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="scratch" \
  --enable_xformers_memory_efficient_attention \
  --caption_column="label" \
  --dataloader_num_workers=20 \
  --seed=42 \
  --checkpointing_steps=2500