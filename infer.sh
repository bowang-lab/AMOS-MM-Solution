source /h/junma/.mllm/bin/activate
CUDA_VISIBLE_DEVICES="0" accelerate launch --num_processes 1 --main_process_port 29500 infer.py \
  --model_name_or_path /home/jma/datasets/mohammed/amosmm/models/phi3_medium  \
  --json_path Data/AMOSMM.json \
  --model_max_length 768 \
  --prompt "simple" \
  --proj_out_num 256