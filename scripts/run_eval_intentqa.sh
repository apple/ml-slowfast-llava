#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
ROOT_DIR=${ROOT_DIR:-"$(dirname "$(dirname "$(readlink -f "$0")")")"}
CONFIG_NAME=${CONFIG_NAME:-"slowfast_llava_7b-resize-slow_10frms_spatial_1d_max_pool_fast_4x4-50_frms"}
DATA_DIR=${DATA_DIR:-"${ROOT_DIR}/playground/data/multiple_choice_qa"}
GT_QA_DIR=${GT_QA_DIR:-"${ROOT_DIR}/playground/gt_qa_files"}
MODEL_PATH=${MODEL_PATH:-"${ROOT_DIR}/liuhaotian/llava-v1.6-vicuna-7b/"}
OUTPUT_DIR=${OUTPUT_DIR:-"${ROOT_DIR}/outputs/artifacts"}
TEMP_DIR=${TEMP_DIR:-"${ROOT_DIR}/outputs/eval_save_dir"}
CONV_MODE=${CONV_MODE:-"multiple_choice_allvideo_v4"}
NUM_FRAMES=${NUM_FRAMES:-"50"}
INPUT_STRUCTURE=${INPUT_STRUCTURE:-"image_seq"}
TEMPORAL_AGGREGATION=${TEMPORAL_AGGREGATION:-"slowfast-slow_10frms_spatial_1d_max_pool-fast_4x4"}
IMAGE_ASPECT_RATIO=${IMAGE_ASPECT_RATIO:-"resize"}
ROPE_SCALING_FACTOR=${ROPE_SCALING_FACTOR:-"2"}

################################# Run ##################################

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 ${ROOT_DIR}/run_inference_multiple_choice_qa.py \
      --video_dir ${DATA_DIR}/IntentQA/video \
      --gt_file ${GT_QA_DIR}/IntentQA/val_qa.json \
      --output_dir ${OUTPUT_DIR}/IntentQA/${CONFIG_NAME} \
      --output_name ${CHUNKS}_${IDX} \
      --model_path ${MODEL_PATH} \
      --conv_mode ${CONV_MODE} \
      --num_chunks ${CHUNKS} \
      --chunk_idx ${IDX} \
      --num_frames ${NUM_FRAMES} \
      --temperature 0 \
      --input_structure ${INPUT_STRUCTURE} \
      --temporal_aggregation ${TEMPORAL_AGGREGATION} \
      --image_aspect_ratio ${IMAGE_ASPECT_RATIO} \
      --rope_scaling_factor ${ROPE_SCALING_FACTOR} &
done

wait

output_dir=${OUTPUT_DIR}/IntentQA/${CONFIG_NAME}
output_file=${output_dir}/merge.jsonl
temp_dir=${TEMP_DIR}/IntentQA/${CONFIG_NAME}

# Clear out the output file if it exists.
> "${output_file}"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${output_dir}/${CHUNKS}_${IDX}.json >> "${output_file}"
done

################################# Eval ##################################

python3 ${ROOT_DIR}/eval/eval_multiple_choice_qa.py \
    --pred_path ${output_file}
