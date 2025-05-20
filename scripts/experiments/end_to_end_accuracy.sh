
# MM-Embed + Qwen2.5VL

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/top_modules/generator.py \
    --run_name mmembed_default \
    --target_dataset MP-DocVQA \
    --retrieval_results_path /root/LILaC/algorithm_results/LILaC/MP-DocVQA/retrieval/mmembed_default/mmembed_default.jsonl \
    --num_components 3 \
    --num_gpus 4 \
    --force_overwrite True \
    --images_subpath image_components

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/top_modules/generator.py \
    --run_name mmembed_default \
    --target_dataset SlideVQA \
    --retrieval_results_path /root/LILaC/algorithm_results/LILaC/SlideVQA/retrieval/mmembed_default/mmembed_default.jsonl \
    --num_components 3 \
    --num_gpus 4 \
    --force_overwrite True \
    --images_subpath image_components

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/top_modules/generator.py \
    --run_name mmembed_default \
    --target_dataset InfoVQA \
    --retrieval_results_path /root/LILaC/algorithm_results/LILaC/InfoVQA/retrieval/mmembed_default/mmembed_default.jsonl \
    --num_components 3 \
    --num_gpus 4 \
    --force_overwrite True \
    --images_subpath image_components

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/top_modules/generator.py \
    --run_name mmembed_default \
    --target_dataset MMCoQA \
    --retrieval_results_path /root/LILaC/algorithm_results/LILaC/MMCoQA/retrieval/mmembed_default/mmembed_default.jsonl \
    --num_components 9 \
    --num_gpus 4 \
    --force_overwrite True

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/top_modules/generator.py \
    --run_name mmembed_default_nonvllm \
    --target_dataset MMWebQA \
    --retrieval_results_path /root/LILaC/algorithm_results/LILaC/MMWebQA/retrieval/mmembed_default/mmembed_default.jsonl \
    --num_components 9 \
    --num_gpus 4 \
    --force_overwrite True








# UniME + Qwen2.5VL

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/top_modules/generator.py \
#     --run_name unime_default \
#     --target_dataset MP-DocVQA \
#     --retrieval_results_path /root/LILaC/algorithm_results/LILaC/MP-DocVQA/retrieval/unime_default/unime_default.jsonl \
#     --num_components 3 \
#     --num_gpus 4 \
#     --force_overwrite True \
#     --images_subpath image_components

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/top_modules/generator.py \
#     --run_name unime_default \
#     --target_dataset SlideVQA \
#     --retrieval_results_path /root/LILaC/algorithm_results/LILaC/SlideVQA/retrieval/unime_default/unime_default.jsonl \
#     --num_components 3 \
#     --num_gpus 4 \
#     --force_overwrite True \
#     --images_subpath image_components

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/top_modules/generator.py \
#     --run_name unime_default \
#     --target_dataset InfoVQA \
#     --retrieval_results_path /root/LILaC/algorithm_results/LILaC/InfoVQA/retrieval/unime_default/unime_default.jsonl \
#     --num_components 3 \
#     --num_gpus 4 \
#     --force_overwrite True \
#     --images_subpath image_components

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/top_modules/generator.py \
#     --run_name unime_default \
#     --target_dataset MMCoQA \
#     --retrieval_results_path /root/LILaC/algorithm_results/LILaC/MMCoQA/retrieval/unime_default/unime_default.jsonl \
#     --num_components 9 \
#     --num_gpus 4 \
#     --force_overwrite True


# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/top_modules/generator.py \
#     --run_name unime_default \
#     --target_dataset MMWebQA \
#     --retrieval_results_path /root/LILaC/algorithm_results/LILaC/MMWebQA/retrieval/unime_default/unime_default.jsonl \
#     --num_components 9 \
#     --num_gpus 4 \
#     --force_overwrite True




# MMe5 + Qwen2.5VL

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/top_modules/generator.py \
#     --run_name mme5_default \
#     --target_dataset MP-DocVQA \
#     --retrieval_results_path /root/LILaC/algorithm_results/LILaC/MP-DocVQA/retrieval/mme5_default/mme5_default.jsonl \
#     --num_components 3 \
#     --num_gpus 4 \
#     --force_overwrite True \
#     --images_subpath image_components

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/top_modules/generator.py \
#     --run_name mme5_default \
#     --target_dataset SlideVQA \
#     --retrieval_results_path /root/LILaC/algorithm_results/LILaC/SlideVQA/retrieval/mme5_default/mme5_default.jsonl \
#     --num_components 3 \
#     --num_gpus 4 \
#     --force_overwrite True \
#     --images_subpath image_components

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/top_modules/generator.py \
#     --run_name mme5_default \
#     --target_dataset InfoVQA \
#     --retrieval_results_path /root/LILaC/algorithm_results/LILaC/InfoVQA/retrieval/mme5_default/mme5_default.jsonl \
#     --num_components 3 \
#     --num_gpus 4 \
#     --force_overwrite True \
#     --images_subpath image_components

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/top_modules/generator.py \
#     --run_name mme5_default \
#     --target_dataset MMCoQA \
#     --retrieval_results_path /root/LILaC/algorithm_results/LILaC/MMCoQA/retrieval/mme5_default/mme5_default.jsonl \
#     --num_components 9 \
#     --num_gpus 4 \
#     --force_overwrite True

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/top_modules/generator.py \
#     --run_name mme5_default \
#     --target_dataset MMWebQA \
#     --retrieval_results_path /root/LILaC/algorithm_results/LILaC/MMWebQA/retrieval/mme5_default/mme5_default.jsonl \
#     --num_components 9 \
#     --num_gpus 4 \
    # --force_overwrite True




