CUDA_VISIBLE_DEVICES="0" accelerate launch --num_processes 1 --main_process_port 29600 infer_vqa.py \
  --model_name_or_path /scratch/ssd004/datasets/med-img-data/amosmm/trained/paper/phi3_vqa_tt/ \
  --json_path Data/AMOSMM.json \
  --image_size 32 256 256 \
  --model_max_length 512 \
  --proj_out_num 256 \
  --with_acc True