#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
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
SAVE_DIR=${SAVE_DIR:-"${ROOT_DIR}/outputs/artifacts/logs"}

mkdir -p ${TEMP_DIR}
mkdir -p ${SAVE_DIR}

################################# Run ##################################

echo "evaluating nextqa ..."

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
CONFIG_NAME=${CONFIG_NAME} \
DATA_DIR=${DATA_DIR} \
GT_QA_DIR=${GT_QA_DIR} \
MODEL_PATH=${MODEL_PATH} \
OUTPUT_DIR=${OUTPUT_DIR} \
TEMP_DIR=${TEMP_DIR} \
CONV_MODE=${CONV_MODE} \
NUM_FRAMES=${NUM_FRAMES} \
INPUT_STRUCTURE=${INPUT_STRUCTURE} \
TEMPORAL_AGGREGATION=${TEMPORAL_AGGREGATION} \
IMAGE_ASPECT_RATIO=${IMAGE_ASPECT_RATIO} \
ROPE_SCALING_FACTOR=${ROPE_SCALING_FACTOR} \
bash scripts/run_eval_nextqa.sh >> ${SAVE_DIR}/${CONFIG_NAME}_nextqa.log

wait

echo "evaluating egoschema ..."

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
CONFIG_NAME=${CONFIG_NAME} \
DATA_DIR=${DATA_DIR} \
GT_QA_DIR=${GT_QA_DIR} \
MODEL_PATH=${MODEL_PATH} \
OUTPUT_DIR=${OUTPUT_DIR} \
TEMP_DIR=${TEMP_DIR} \
CONV_MODE=${CONV_MODE} \
NUM_FRAMES=${NUM_FRAMES} \
INPUT_STRUCTURE=${INPUT_STRUCTURE} \
TEMPORAL_AGGREGATION=${TEMPORAL_AGGREGATION} \
IMAGE_ASPECT_RATIO=${IMAGE_ASPECT_RATIO} \
ROPE_SCALING_FACTOR=${ROPE_SCALING_FACTOR} \
bash scripts/run_eval_egoschema.sh >> ${SAVE_DIR}/${CONFIG_NAME}_egoschema.log

wait

echo "evaluating intentqa ..."

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
CONFIG_NAME=${CONFIG_NAME} \
DATA_DIR=${DATA_DIR} \
GT_QA_DIR=${GT_QA_DIR} \
MODEL_PATH=${MODEL_PATH} \
OUTPUT_DIR=${OUTPUT_DIR} \
TEMP_DIR=${TEMP_DIR} \
CONV_MODE=${CONV_MODE} \
NUM_FRAMES=${NUM_FRAMES} \
INPUT_STRUCTURE=${INPUT_STRUCTURE} \
TEMPORAL_AGGREGATION=${TEMPORAL_AGGREGATION} \
IMAGE_ASPECT_RATIO=${IMAGE_ASPECT_RATIO} \
ROPE_SCALING_FACTOR=${ROPE_SCALING_FACTOR} \
bash scripts/run_eval_intentqa.sh >> ${SAVE_DIR}/${CONFIG_NAME}_intentqa.log
