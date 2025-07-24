MAMBA_MODEL=$1
PRED_OUTPUT_PATH="data/nnUNet_results/Dataset001_kvasir-seg/${MAMBA_MODEL}__nnUNetPlans__2d/pred_results"
EVAL_METRIC_PATH="data/nnUNet_results/Dataset001_kvasir-seg/${MAMBA_MODEL}__nnUNetPlans__2d"
GPU_ID="0"

CUDA_VISIBLE_DEVICES=${GPU_ID} nnUNetv2_train 001 2d all -tr ${MAMBA_MODEL} &&

echo "Predicting..." &&
CUDA_VISIBLE_DEVICES=${GPU_ID} nnUNetv2_predict \
    -i "data/nnUNet_raw/Dataset001_kvasir-seg/imagesTs" \
    -o "${PRED_OUTPUT_PATH}" \
    -d 001 \
    -c 2d \
    -tr "${MAMBA_MODEL}" \
    --disable_tta \
    -f all \
    -chk "checkpoint_final.pth" &&

echo "Computing dice..."
python evaluation/endoscopy_DSC_Eval.py \
    --gt_path "data/nnUNet_raw/Dataset001_kvasir-seg/labelsTs" \
    --seg_path "${PRED_OUTPUT_PATH}" \
    --save_path "${EVAL_METRIC_PATH}/metric_DSC.csv"  &&

echo "Computing NSD..."
python evaluation/endoscopy_NSD_Eval.py \
    --gt_path "data/nnUNet_raw/Dataset001_kvasir-seg/labelsTs" \
    --seg_path "${PRED_OUTPUT_PATH}" \
    --save_path "${EVAL_METRIC_PATH}/metric_NSD.csv" &&

echo "Done."