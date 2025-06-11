source /h/junma/.mllm/bin/activate

CUDA_VISIBLE_DEVICES="0" accelerate launch --num_processes 1 --main_process_port 29500 infer.py \
  --model_name_or_path /home/jma/datasets/mohammed/FLARE-Task5-MLLM-3D/results/baseline  \
  --json_path /home/jma/datasets/mohammed/FLARE-Task5-MLLM-3D/validation/val_processed.json \
  --data_root /home/jma/datasets/mohammed/FLARE-Task5-MLLM-3D/validation/val_processed \
  --model_max_length 768 \
  --prompt "simple" \
  --proj_out_num 256