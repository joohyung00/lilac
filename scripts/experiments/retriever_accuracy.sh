

# MM-Embed

CUDA_VISIBLE_DEVICES=0 python3 src/top_modules/retriever.py \
    --target_dataset MP-DocVQA \
    --run_name mmembed_default \
    --run_mode iterative_late_interaction \
    --embedding_model MM-Embed \
    --parameter_numiterations 1 \
    --parameter_beamwidth 30 \
    --force_overwrite True

CUDA_VISIBLE_DEVICES=0 python3 src/top_modules/retriever.py \
    --target_dataset SlideVQA \
    --run_name mmembed_default \
    --run_mode iterative_late_interaction \
    --embedding_model MM-Embed \
    --parameter_numiterations 1 \
    --parameter_beamwidth 30 \
    --force_overwrite True

CUDA_VISIBLE_DEVICES=0 python3 src/top_modules/retriever.py \
    --target_dataset InfoVQA \
    --run_name mmembed_default \
    --run_mode iterative_late_interaction \
    --embedding_model MM-Embed \
    --parameter_numiterations 1 \
    --parameter_beamwidth 30 \
    --force_overwrite True

CUDA_VISIBLE_DEVICES=0 python3 src/top_modules/retriever.py \
    --target_dataset MMWebQA \
    --run_name mmembed_default \
    --run_mode iterative_late_interaction \
    --embedding_model MM-Embed \
    --parameter_numiterations 1 \
    --parameter_beamwidth 30 \
    --force_overwrite True

CUDA_VISIBLE_DEVICES=0 python3 src/top_modules/retriever.py \
    --target_dataset MMCoQA \
    --run_name mmembed_default \
    --run_mode iterative_late_interaction \
    --embedding_model MM-Embed \
    --parameter_numiterations 1 \
    --parameter_beamwidth 30 \
    --force_overwrite True




# MMe5

# CUDA_VISIBLE_DEVICES=0 python3 src/top_modules/retriever.py \
#     --target_dataset MP-DocVQA \
#     --run_name mme5_default \
#     --run_mode iterative_late_interaction \
#     --embedding_model MMe5 \
#     --parameter_numiterations 1 \
#     --parameter_beamwidth 30 \
#     --force_overwrite True

# CUDA_VISIBLE_DEVICES=0 python3 src/top_modules/retriever.py \
#     --target_dataset SlideVQA \
#     --run_name mme5_default \
#     --run_mode iterative_late_interaction \
#     --embedding_model MMe5 \
#     --parameter_numiterations 1 \
#     --parameter_beamwidth 30 \
#     --force_overwrite True

# CUDA_VISIBLE_DEVICES=0 python3 src/top_modules/retriever.py \
#     --target_dataset InfoVQA \
#     --run_name mme5_default \
#     --run_mode iterative_late_interaction \
#     --embedding_model MMe5 \
#     --parameter_numiterations 1 \
#     --parameter_beamwidth 30 \
#     --force_overwrite True

# CUDA_VISIBLE_DEVICES=0 python3 src/top_modules/retriever.py \
#     --target_dataset MMWebQA \
#     --run_name mme5_default \
#     --run_mode iterative_late_interaction \
#     --embedding_model MMe5 \
#     --parameter_numiterations 1 \
#     --parameter_beamwidth 30 \
#     --force_overwrite True

# CUDA_VISIBLE_DEVICES=0 python3 src/top_modules/retriever.py \
#     --target_dataset MMCoQA \
#     --run_name mme5_default \
#     --run_mode iterative_late_interaction \
#     --embedding_model MMe5 \
#     --parameter_numiterations 1 \
#     --parameter_beamwidth 30 \
#     --force_overwrite True



# UniME

# CUDA_VISIBLE_DEVICES=0 python3 src/top_modules/retriever.py \
#     --target_dataset MP-DocVQA \
#     --run_name unime_default \
#     --run_mode iterative_late_interaction \
#     --embedding_model UniME \
#     --parameter_numiterations 1 \
#     --parameter_beamwidth 30 \
#     --force_overwrite True

# CUDA_VISIBLE_DEVICES=0 python3 src/top_modules/retriever.py \
#     --target_dataset SlideVQA \
#     --run_name unime_default \
#     --run_mode iterative_late_interaction \
#     --embedding_model UniME \
#     --parameter_numiterations 1 \
#     --parameter_beamwidth 30 \
#     --force_overwrite True

# CUDA_VISIBLE_DEVICES=0 python3 src/top_modules/retriever.py \
#     --target_dataset InfoVQA \
#     --run_name unime_default \
#     --run_mode iterative_late_interaction \
#     --embedding_model UniME \
#     --parameter_numiterations 1 \
#     --parameter_beamwidth 30 \
#     --force_overwrite True

# CUDA_VISIBLE_DEVICES=0 python3 src/top_modules/retriever.py \
#     --target_dataset MMWebQA \
#     --run_name unime_default \
#     --run_mode iterative_late_interaction \
#     --embedding_model UniME \
#     --parameter_numiterations 1 \
#     --parameter_beamwidth 30 \
#     --force_overwrite True

# CUDA_VISIBLE_DEVICES=0 python3 src/top_modules/retriever.py \
#     --target_dataset MMCoQA \
#     --run_name unime_default \
#     --run_mode iterative_late_interaction \
#     --embedding_model UniME \
#     --parameter_numiterations 1 \
#     --parameter_beamwidth 30 \
#     --force_overwrite True


