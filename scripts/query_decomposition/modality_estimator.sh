eval "$(conda shell.bash hook)"
conda activate qwen2.5

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/query_decomposer/modality_estimator.py