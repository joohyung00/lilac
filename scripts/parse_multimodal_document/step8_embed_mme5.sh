
eval "$(conda shell.bash hook)"
conda activate mme5





# MP-DocVQA

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mme5.py \
    --corpus_in_filepath  /root/LILaC/datasets/MP-DocVQA/embeddings/serializations/image.json \
    --out_embeddings_path /root/LILaC/datasets/MP-DocVQA/embeddings/MMe5/image.pt \
    --out_index_path      /root/LILaC/datasets/MP-DocVQA/embeddings/MMe5/image.json \
    --num_gpus 4

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mme5.py \
    --corpus_in_filepath  /root/LILaC/datasets/MP-DocVQA/embeddings/serializations/sentence.json \
    --out_embeddings_path /root/LILaC/datasets/MP-DocVQA/embeddings/MMe5/sentence.pt \
    --out_index_path      /root/LILaC/datasets/MP-DocVQA/embeddings/MMe5/sentence.json \
    --num_gpus 4

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mme5.py \
    --corpus_in_filepath  /root/LILaC/datasets/MP-DocVQA/embeddings/serializations/table_segment+summary.json \
    --out_embeddings_path /root/LILaC/datasets/MP-DocVQA/embeddings/MMe5/table.pt \
    --out_index_path      /root/LILaC/datasets/MP-DocVQA/embeddings/MMe5/table.json \
    --num_gpus 4

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mme5.py \
    --corpus_in_filepath  /root/LILaC/datasets/MP-DocVQA/embeddings/serializations/subimage.json \
    --out_embeddings_path /root/LILaC/datasets/MP-DocVQA/embeddings/MMe5/subimage.pt \
    --out_index_path      /root/LILaC/datasets/MP-DocVQA/embeddings/MMe5/subimage.json \
    --num_gpus 4

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mme5.py \
    --corpus_in_filepath  /root/LILaC/datasets/MP-DocVQA/embeddings/serializations/page.json \
    --out_embeddings_path /root/LILaC/datasets/MP-DocVQA/embeddings/MMe5/page.pt \
    --out_index_path      /root/LILaC/datasets/MP-DocVQA/embeddings/MMe5/page.json \
    --num_gpus 4 \
    --max_length 4096 




# SlideVQA


CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mme5.py \
    --corpus_in_filepath  /root/LILaC/datasets/SlideVQA/embeddings/serializations/image.json \
    --out_embeddings_path /root/LILaC/datasets/SlideVQA/embeddings/MMe5/image.pt \
    --out_index_path      /root/LILaC/datasets/SlideVQA/embeddings/MMe5/image.json \
    --num_gpus 4 \
    --max_length 1024

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mme5.py \
    --corpus_in_filepath  /root/LILaC/datasets/SlideVQA/embeddings/serializations/sentence.json \
    --out_embeddings_path /root/LILaC/datasets/SlideVQA/embeddings/MMe5/sentence.pt \
    --out_index_path      /root/LILaC/datasets/SlideVQA/embeddings/MMe5/sentence.json \
    --num_gpus 4 \
    --max_length 1024

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mme5.py \
    --corpus_in_filepath  /root/LILaC/datasets/SlideVQA/embeddings/serializations/table_segment+summary.json \
    --out_embeddings_path /root/LILaC/datasets/SlideVQA/embeddings/MMe5/table.pt \
    --out_index_path      /root/LILaC/datasets/SlideVQA/embeddings/MMe5/table.json \
    --num_gpus 4 \
    --max_length 2048

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mme5.py \
    --corpus_in_filepath  /root/LILaC/datasets/SlideVQA/embeddings/serializations/subimage.json \
    --out_embeddings_path /root/LILaC/datasets/SlideVQA/embeddings/MMe5/subimage.pt \
    --out_index_path      /root/LILaC/datasets/SlideVQA/embeddings/MMe5/subimage.json \
    --num_gpus 4 \
    --max_length 2048

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mme5.py \
    --corpus_in_filepath  /root/LILaC/datasets/SlideVQA/embeddings/serializations/page.json \
    --out_embeddings_path /root/LILaC/datasets/SlideVQA/embeddings/MMe5/page.pt \
    --out_index_path      /root/LILaC/datasets/SlideVQA/embeddings/MMe5/page.json \
    --num_gpus 4 \
    --max_length 4096 




# InfoVQA

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mme5.py \
    --corpus_in_filepath  /root/LILaC/datasets/InfoVQA/embeddings/serializations/image.json \
    --out_embeddings_path /root/LILaC/datasets/InfoVQA/embeddings/MMe5/image.pt \
    --out_index_path      /root/LILaC/datasets/InfoVQA/embeddings/MMe5/image.json \
    --num_gpus 4

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mme5.py \
    --corpus_in_filepath  /root/LILaC/datasets/InfoVQA/embeddings/serializations/sentence.json \
    --out_embeddings_path /root/LILaC/datasets/InfoVQA/embeddings/MMe5/sentence.pt \
    --out_index_path      /root/LILaC/datasets/InfoVQA/embeddings/MMe5/sentence.json \
    --num_gpus 4

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mme5.py \
    --corpus_in_filepath  /root/LILaC/datasets/InfoVQA/embeddings/serializations/table_segment+summary.json \
    --out_embeddings_path /root/LILaC/datasets/InfoVQA/embeddings/MMe5/table.pt \
    --out_index_path      /root/LILaC/datasets/InfoVQA/embeddings/MMe5/table.json \
    --num_gpus 4

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mme5.py \
    --corpus_in_filepath  /root/LILaC/datasets/InfoVQA/embeddings/serializations/subimage.json \
    --out_embeddings_path /root/LILaC/datasets/InfoVQA/embeddings/MMe5/subimage.pt \
    --out_index_path      /root/LILaC/datasets/InfoVQA/embeddings/MMe5/subimage.json \
    --num_gpus 4

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mme5.py \
    --corpus_in_filepath  /root/LILaC/datasets/InfoVQA/embeddings/serializations/page.json \
    --out_embeddings_path /root/LILaC/datasets/InfoVQA/embeddings/MMe5/page.pt \
    --out_index_path      /root/LILaC/datasets/InfoVQA/embeddings/MMe5/page.json \
    --num_gpus 4 \
    --max_length 4096 



# MMCoQA

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mme5.py \
    --corpus_in_filepath  /root/LILaC/datasets/MMCoQA/embeddings/serializations/text.json \
    --out_embeddings_path /root/LILaC/datasets/MMCoQA/embeddings/MMe5/text.pt \
    --out_index_path      /root/LILaC/datasets/MMCoQA/embeddings/MMe5/text.json \
    --num_gpus 4 \
    --max_length 2048

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mme5.py \
    --corpus_in_filepath  /root/LILaC/datasets/MMCoQA/embeddings/serializations/sentence.json \
    --out_embeddings_path /root/LILaC/datasets/MMCoQA/embeddings/MMe5/sentence.pt \
    --out_index_path      /root/LILaC/datasets/MMCoQA/embeddings/MMe5/sentence.json \
    --num_gpus 4 \
    --max_length 2048

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mme5.py \
    --corpus_in_filepath  /root/LILaC/datasets/MMCoQA/embeddings/serializations/table.json \
    --out_embeddings_path /root/LILaC/datasets/MMCoQA/embeddings/MMe5/table.pt \
    --out_index_path      /root/LILaC/datasets/MMCoQA/embeddings/MMe5/table.json \
    --num_gpus 4 \
    --max_length 4096

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mme5.py \
    --corpus_in_filepath  /root/LILaC/datasets/MMCoQA/embeddings/serializations/table.json \
    --out_embeddings_path /root/LILaC/datasets/MMCoQA/embeddings/MMe5/table.pt \
    --out_index_path      /root/LILaC/datasets/MMCoQA/embeddings/MMe5/table.json \
    --num_gpus 4 \
    --max_length 4096

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mme5.py \
    --corpus_in_filepath  /root/LILaC/datasets/MMCoQA/embeddings/serializations/image.json \
    --out_embeddings_path /root/LILaC/datasets/MMCoQA/embeddings/MMe5/image.pt \
    --out_index_path      /root/LILaC/datasets/MMCoQA/embeddings/MMe5/image.json \
    --num_gpus 4 \
    --max_length 4096

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mme5.py \
    --corpus_in_filepath  /root/LILaC/datasets/MMCoQA/embeddings/serializations/subimage.json \
    --out_embeddings_path /root/LILaC/datasets/MMCoQA/embeddings/MMe5/subimage.pt \
    --out_index_path      /root/LILaC/datasets/MMCoQA/embeddings/MMe5/subimage.json \
    --num_gpus 4 \
    --max_length 4096

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mme5.py \
    --corpus_in_filepath  /root/LILaC/datasets/MMCoQA/embeddings/serializations/page.json \
    --out_embeddings_path /root/LILaC/datasets/MMCoQA/embeddings/MMe5/page.pt \
    --out_index_path      /root/LILaC/datasets/MMCoQA/embeddings/MMe5/page.json \
    --num_gpus 4 \
    --max_length 4096 


# MMWebQA



CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mme5.py \
    --corpus_in_filepath  /root/LILaC/datasets/MMWebQA/embeddings/serializations/text.json \
    --out_embeddings_path /root/LILaC/datasets/MMWebQA/embeddings/MMe5/text.pt \
    --out_index_path      /root/LILaC/datasets/MMWebQA/embeddings/MMe5/text.json \
    --num_gpus 4 

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mme5.py \
    --corpus_in_filepath  /root/LILaC/datasets/MMWebQA/embeddings/serializations/sentence.json \
    --out_embeddings_path /root/LILaC/datasets/MMWebQA/embeddings/MMe5/sentence.pt \
    --out_index_path      /root/LILaC/datasets/MMWebQA/embeddings/MMe5/sentence.json \
    --num_gpus 4

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mme5.py \
    --corpus_in_filepath  /root/LILaC/datasets/MMWebQA/embeddings/serializations/table.json \
    --out_embeddings_path /root/LILaC/datasets/MMWebQA/embeddings/MMe5/table.pt \
    --out_index_path      /root/LILaC/datasets/MMWebQA/embeddings/MMe5/table.json \
    --num_gpus 4

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mme5.py \
    --corpus_in_filepath  /root/LILaC/datasets/MMWebQA/embeddings/serializations/table.json \
    --out_embeddings_path /root/LILaC/datasets/MMWebQA/embeddings/MMe5/table.pt \
    --out_index_path      /root/LILaC/datasets/MMWebQA/embeddings/MMe5/table.json \
    --num_gpus 4

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mme5.py \
    --corpus_in_filepath  /root/LILaC/datasets/MMWebQA/embeddings/serializations/image.json \
    --out_embeddings_path /root/LILaC/datasets/MMWebQA/embeddings/MMe5/image.pt \
    --out_index_path      /root/LILaC/datasets/MMWebQA/embeddings/MMe5/image.json \
    --num_gpus 4

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mme5.py \
    --corpus_in_filepath  /root/LILaC/datasets/MMWebQA/embeddings/serializations/subimage.json \
    --out_embeddings_path /root/LILaC/datasets/MMWebQA/embeddings/MMe5/subimage.pt \
    --out_index_path      /root/LILaC/datasets/MMWebQA/embeddings/MMe5/subimage.json \
    --num_gpus 4

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mme5.py \
    --corpus_in_filepath  /root/LILaC/datasets/MMWebQA/embeddings/serializations/page.json \
    --out_embeddings_path /root/LILaC/datasets/MMWebQA/embeddings/MMe5/page.pt \
    --out_index_path      /root/LILaC/datasets/MMWebQA/embeddings/MMe5/page.json \
    --num_gpus 4 \
    --max_length 4096 