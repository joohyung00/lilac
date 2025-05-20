eval "$(conda shell.bash hook)"
conda activate qwen2.5

# CUDA_VISIBLE_DEVICES=0 python3 src/multimodal_document_manager/step5_add_subimages.py --target_data MMWebQA   --mode split
# CUDA_VISIBLE_DEVICES=0 python3 src/multimodal_document_manager/step5_add_subimages.py --target_data MMCoQA    --mode split
# CUDA_VISIBLE_DEVICES=1 python3 src/multimodal_document_manager/step5_add_subimages.py --target_data SlideVQA  --mode split
# CUDA_VISIBLE_DEVICES=2 python3 src/multimodal_document_manager/step5_add_subimages.py --target_data InfoVQA   --mode split
# CUDA_VISIBLE_DEVICES=3 python3 src/multimodal_document_manager/step5_add_subimages.py --target_data MP-DocVQA --mode split




# python3 src/multimodal_document_manager/step5_add_subimages.py --target_data MMWebQA   --mode add_to_parsed_documents
# python3 src/multimodal_document_manager/step5_add_subimages.py --target_data MMCoQA    --mode add_to_parsed_documents
# python3 src/multimodal_document_manager/step5_add_subimages.py --target_data SlideVQA  --mode add_to_parsed_documents
# python3 src/multimodal_document_manager/step5_add_subimages.py --target_data InfoVQA   --mode add_to_parsed_documents
# python3 src/multimodal_document_manager/step5_add_subimages.py --target_data MP-DocVQA --mode add_to_parsed_documents



