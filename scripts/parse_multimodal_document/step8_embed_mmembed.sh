
# eval "$(conda shell.bash hook)"
# conda activate mmembed








# ## MP-DocVQA

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mmembed.py \
#     --corpus_in_filepath  /root/LILaC/datasets/MP-DocVQA/embeddings/serializations/text.json \
#     --out_embeddings_path /root/LILaC/datasets/MP-DocVQA/embeddings/MM-Embed/text.pt \
#     --out_index_path      /root/LILaC/datasets/MP-DocVQA/embeddings/MM-Embed/text.json \
#     --num_gpus 4 \
#     --max_length 256 \
#     --batch_size 32

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mmembed.py \
#     --corpus_in_filepath  /root/LILaC/datasets/MP-DocVQA/embeddings/serializations/sentence.json \
#     --out_embeddings_path /root/LILaC/datasets/MP-DocVQA/embeddings/MM-Embed/sentence.pt \
#     --out_index_path      /root/LILaC/datasets/MP-DocVQA/embeddings/MM-Embed/sentence.json \
#     --num_gpus 4 \
#     --max_length 128 \
#     --batch_size 32

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mmembed.py \
#     --corpus_in_filepath  /root/LILaC/datasets/MP-DocVQA/embeddings/serializations/table.json \
#     --out_embeddings_path /root/LILaC/datasets/MP-DocVQA/embeddings/MM-Embed/table.pt \
#     --out_index_path      /root/LILaC/datasets/MP-DocVQA/embeddings/MM-Embed/table.json \
#     --num_gpus 4 \
#     --max_length 256 \
#     --batch_size 32

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mmembed.py \
#     --corpus_in_filepath  /root/LILaC/datasets/MP-DocVQA/embeddings/serializations/table_segment+image.json \
#     --out_embeddings_path /root/LILaC/datasets/MP-DocVQA/embeddings/MM-Embed/table_segment+image.pt \
#     --out_index_path      /root/LILaC/datasets/MP-DocVQA/embeddings/MM-Embed/table_segment+image.json \
#     --num_gpus 4 \
#     --max_length 128 \
#     --batch_size 32

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mmembed.py \
#     --corpus_in_filepath  /root/LILaC/datasets/MP-DocVQA/embeddings/serializations/image.json \
#     --out_embeddings_path /root/LILaC/datasets/MP-DocVQA/embeddings/MM-Embed/image.pt \
#     --out_index_path      /root/LILaC/datasets/MP-DocVQA/embeddings/MM-Embed/image.json \
#     --num_gpus 4 \
#     --max_length 4096 \
#     --batch_size 1

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mmembed.py \
#     --corpus_in_filepath  /root/LILaC/datasets/MP-DocVQA/embeddings/serializations/subimage.json \
#     --out_embeddings_path /root/LILaC/datasets/MP-DocVQA/embeddings/MM-Embed/subimage.pt \
#     --out_index_path      /root/LILaC/datasets/MP-DocVQA/embeddings/MM-Embed/subimage.json \
#     --num_gpus 4 \
#     --max_length 256 \
#     --batch_size 8

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mmembed.py \
#     --corpus_in_filepath  /root/LILaC/datasets/MP-DocVQA/embeddings/serializations/page.json \
#     --out_embeddings_path /root/LILaC/datasets/MP-DocVQA/embeddings/MM-Embed/page.pt \
#     --out_index_path      /root/LILaC/datasets/MP-DocVQA/embeddings/MM-Embed/page.json \
#     --num_gpus 4 \
#     --max_length 4096 \
#     --batch_size 1






# # SlideVQA

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mmembed.py \
#     --corpus_in_filepath  /root/LILaC/datasets/SlideVQA/embeddings/serializations/text.json \
#     --out_embeddings_path /root/LILaC/datasets/SlideVQA/embeddings/MM-Embed/text.pt \
#     --out_index_path      /root/LILaC/datasets/SlideVQA/embeddings/MM-Embed/text.json \
#     --num_gpus 4 \
#     --max_length 256 \
#     --batch_size 32

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mmembed.py \
#     --corpus_in_filepath  /root/LILaC/datasets/SlideVQA/embeddings/serializations/sentence.json \
#     --out_embeddings_path /root/LILaC/datasets/SlideVQA/embeddings/MM-Embed/sentence.pt \
#     --out_index_path      /root/LILaC/datasets/SlideVQA/embeddings/MM-Embed/sentence.json \
#     --num_gpus 4 \
#     --max_length 128 \
#     --batch_size 32

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mmembed.py \
#     --corpus_in_filepath  /root/LILaC/datasets/SlideVQA/embeddings/serializations/table.json \
#     --out_embeddings_path /root/LILaC/datasets/SlideVQA/embeddings/MM-Embed/table.pt \
#     --out_index_path      /root/LILaC/datasets/SlideVQA/embeddings/MM-Embed/table.json \
#     --num_gpus 4 \
#     --max_length 256 \
#     --batch_size 32

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mmembed.py \
#     --corpus_in_filepath  /root/LILaC/datasets/SlideVQA/embeddings/serializations/table_segment+image.json \
#     --out_embeddings_path /root/LILaC/datasets/SlideVQA/embeddings/MM-Embed/table_segment+image.pt \
#     --out_index_path      /root/LILaC/datasets/SlideVQA/embeddings/MM-Embed/table_segment+image.json \
#     --num_gpus 4 \
#     --max_length 128 \
#     --batch_size 32

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mmembed.py \
#     --corpus_in_filepath  /root/LILaC/datasets/SlideVQA/embeddings/serializations/image.json \
#     --out_embeddings_path /root/LILaC/datasets/SlideVQA/embeddings/MM-Embed/image.pt \
#     --out_index_path      /root/LILaC/datasets/SlideVQA/embeddings/MM-Embed/image.json \
#     --num_gpus 4 \
#     --max_length 4096 \
#     --batch_size 1

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mmembed.py \
#     --corpus_in_filepath  /root/LILaC/datasets/SlideVQA/embeddings/serializations/subimage.json \
#     --out_embeddings_path /root/LILaC/datasets/SlideVQA/embeddings/MM-Embed/subimage.pt \
#     --out_index_path      /root/LILaC/datasets/SlideVQA/embeddings/MM-Embed/subimage.json \
#     --num_gpus 4 \
#     --max_length 256 \
#     --batch_size 8

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mmembed.py \
#     --corpus_in_filepath  /root/LILaC/datasets/SlideVQA/embeddings/serializations/page.json \
#     --out_embeddings_path /root/LILaC/datasets/SlideVQA/embeddings/MM-Embed/page.pt \
#     --out_index_path      /root/LILaC/datasets/SlideVQA/embeddings/MM-Embed/page.json \
#     --num_gpus 4 \
#     --max_length 4096 \
#     --batch_size 1





# # InfoVQA

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mmembed.py \
#     --corpus_in_filepath  /root/LILaC/datasets/InfoVQA/embeddings/serializations/text.json \
#     --out_embeddings_path /root/LILaC/datasets/InfoVQA/embeddings/MM-Embed/text.pt \
#     --out_index_path      /root/LILaC/datasets/InfoVQA/embeddings/MM-Embed/text.json \
#     --num_gpus 4 \
#     --max_length 256 \
#     --batch_size 32

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mmembed.py \
#     --corpus_in_filepath  /root/LILaC/datasets/InfoVQA/embeddings/serializations/sentence.json \
#     --out_embeddings_path /root/LILaC/datasets/InfoVQA/embeddings/MM-Embed/sentence.pt \
#     --out_index_path      /root/LILaC/datasets/InfoVQA/embeddings/MM-Embed/sentence.json \
#     --num_gpus 4 \
#     --max_length 128 \
#     --batch_size 32

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mmembed.py \
#     --corpus_in_filepath  /root/LILaC/datasets/InfoVQA/embeddings/serializations/table.json \
#     --out_embeddings_path /root/LILaC/datasets/InfoVQA/embeddings/MM-Embed/table.pt \
#     --out_index_path      /root/LILaC/datasets/InfoVQA/embeddings/MM-Embed/table.json \
#     --num_gpus 4 \
#     --max_length 256 \
#     --batch_size 32

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mmembed.py \
#     --corpus_in_filepath  /root/LILaC/datasets/InfoVQA/embeddings/serializations/table_segment+image.json \
#     --out_embeddings_path /root/LILaC/datasets/InfoVQA/embeddings/MM-Embed/table_segment+image.pt \
#     --out_index_path      /root/LILaC/datasets/InfoVQA/embeddings/MM-Embed/table_segment+image.json \
#     --num_gpus 4 \
#     --max_length 128 \
#     --batch_size 32

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mmembed.py \
#     --corpus_in_filepath  /root/LILaC/datasets/InfoVQA/embeddings/serializations/image.json \
#     --out_embeddings_path /root/LILaC/datasets/InfoVQA/embeddings/MM-Embed/image.pt \
#     --out_index_path      /root/LILaC/datasets/InfoVQA/embeddings/MM-Embed/image.json \
#     --num_gpus 4 \
#     --max_length 4096 \
#     --batch_size 1

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mmembed.py \
#     --corpus_in_filepath  /root/LILaC/datasets/InfoVQA/embeddings/serializations/subimage.json \
#     --out_embeddings_path /root/LILaC/datasets/InfoVQA/embeddings/MM-Embed/subimage.pt \
#     --out_index_path      /root/LILaC/datasets/InfoVQA/embeddings/MM-Embed/subimage.json \
#     --num_gpus 4 \
#     --max_length 256 \
#     --batch_size 8

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mmembed.py \
#     --corpus_in_filepath  /root/LILaC/datasets/InfoVQA/embeddings/serializations/page.json \
#     --out_embeddings_path /root/LILaC/datasets/InfoVQA/embeddings/MM-Embed/page.pt \
#     --out_index_path      /root/LILaC/datasets/InfoVQA/embeddings/MM-Embed/page.json \
#     --num_gpus 4 \
#     --max_length 4096 \
#     --batch_size 1




# ## MMCoQA

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mmembed.py \
#     --corpus_in_filepath  /root/LILaC/datasets/MMCoQA/embeddings/serializations/text.json \
#     --out_embeddings_path /root/LILaC/datasets/MMCoQA/embeddings/MM-Embed/text.pt \
#     --out_index_path      /root/LILaC/datasets/MMCoQA/embeddings/MM-Embed/text.json \
#     --num_gpus 4 \
#     --max_length 1024 \
#     --batch_size 16

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mmembed.py \
#     --corpus_in_filepath  /root/LILaC/datasets/MMCoQA/embeddings/serializations/sentence.json \
#     --out_embeddings_path /root/LILaC/datasets/MMCoQA/embeddings/MM-Embed/sentence.pt \
#     --out_index_path      /root/LILaC/datasets/MMCoQA/embeddings/MM-Embed/sentence.json \
#     --num_gpus 4 \
#     --max_length 128 \
#     --batch_size 64

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mmembed.py \
#     --corpus_in_filepath  /root/LILaC/datasets/MMCoQA/embeddings/serializations/table.json \
#     --out_embeddings_path /root/LILaC/datasets/MMCoQA/embeddings/MM-Embed/table.pt \
#     --out_index_path      /root/LILaC/datasets/MMCoQA/embeddings/MM-Embed/table.json \
#     --num_gpus 4 \
#     --max_length 2048 \
#     --batch_size 4

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mmembed.py \
#     --corpus_in_filepath  /root/LILaC/datasets/MMCoQA/embeddings/serializations/table.json \
#     --out_embeddings_path /root/LILaC/datasets/MMCoQA/embeddings/MM-Embed/table.pt \
#     --out_index_path      /root/LILaC/datasets/MMCoQA/embeddings/MM-Embed/table.json \
#     --num_gpus 4 \
#     --max_length 512 \
#     --batch_size 10

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mmembed.py \
#     --corpus_in_filepath  /root/LILaC/datasets/MMCoQA/embeddings/serializations/image.json \
#     --out_embeddings_path /root/LILaC/datasets/MMCoQA/embeddings/MM-Embed/image.pt \
#     --out_index_path      /root/LILaC/datasets/MMCoQA/embeddings/MM-Embed/image.json \
#     --num_gpus 4 \
#     --max_length 256 \
#     --batch_size 8

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mmembed.py \
#     --corpus_in_filepath  /root/LILaC/datasets/MMCoQA/embeddings/serializations/subimage.json \
#     --out_embeddings_path /root/LILaC/datasets/MMCoQA/embeddings/MM-Embed/subimage.pt \
#     --out_index_path      /root/LILaC/datasets/MMCoQA/embeddings/MM-Embed/subimage.json \
#     --num_gpus 4 \
#     --max_length 256 \
#     --batch_size 8

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mmembed.py \
#     --corpus_in_filepath  /root/LILaC/datasets/MMCoQA/embeddings/serializations/page.json \
#     --out_embeddings_path /root/LILaC/datasets/MMCoQA/embeddings/MM-Embed/page.pt \
#     --out_index_path      /root/LILaC/datasets/MMCoQA/embeddings/MM-Embed/page.json \
#     --num_gpus 4 \
#     --max_length 4096 \
#     --batch_size 1






# ## MMWebQA



# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mmembed.py \
#     --corpus_in_filepath  /root/LILaC/datasets/MMWebQA/embeddings/serializations/text.json \
#     --out_embeddings_path /root/LILaC/datasets/MMWebQA/embeddings/MM-Embed/text.pt \
#     --out_index_path      /root/LILaC/datasets/MMWebQA/embeddings/MM-Embed/text.json \
#     --num_gpus 4 \
#     --max_length 1024 \
#     --batch_size 16

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mmembed.py \
#     --corpus_in_filepath  /root/LILaC/datasets/MMWebQA/embeddings/serializations/sentence.json \
#     --out_embeddings_path /root/LILaC/datasets/MMWebQA/embeddings/MM-Embed/sentence.pt \
#     --out_index_path      /root/LILaC/datasets/MMWebQA/embeddings/MM-Embed/sentence.json \
#     --num_gpus 4 \
#     --max_length 128 \
#     --batch_size 64

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mmembed.py \
#     --corpus_in_filepath  /root/LILaC/datasets/MMWebQA/embeddings/serializations/table.json \
#     --out_embeddings_path /root/LILaC/datasets/MMWebQA/embeddings/MM-Embed/table.pt \
#     --out_index_path      /root/LILaC/datasets/MMWebQA/embeddings/MM-Embed/table.json \
#     --num_gpus 4 \
#     --max_length 2048 \
#     --batch_size 4

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mmembed.py \
#     --corpus_in_filepath  /root/LILaC/datasets/MMWebQA/embeddings/serializations/table.json \
#     --out_embeddings_path /root/LILaC/datasets/MMWebQA/embeddings/MM-Embed/table.pt \
#     --out_index_path      /root/LILaC/datasets/MMWebQA/embeddings/MM-Embed/table.json \
#     --num_gpus 4 \
#     --max_length 512 \
#     --batch_size 10

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mmembed.py \
#     --corpus_in_filepath  /root/LILaC/datasets/MMWebQA/embeddings/serializations/image.json \
#     --out_embeddings_path /root/LILaC/datasets/MMWebQA/embeddings/MM-Embed/image.pt \
#     --out_index_path      /root/LILaC/datasets/MMWebQA/embeddings/MM-Embed/image.json \
#     --num_gpus 4 \
#     --max_length 256 \
#     --batch_size 8

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mmembed.py \
#     --corpus_in_filepath  /root/LILaC/datasets/MMWebQA/embeddings/serializations/subimage.json \
#     --out_embeddings_path /root/LILaC/datasets/MMWebQA/embeddings/MM-Embed/subimage.pt \
#     --out_index_path      /root/LILaC/datasets/MMWebQA/embeddings/MM-Embed/subimage.json \
#     --num_gpus 4 \
#     --max_length 256 \
#     --batch_size 8

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/multimodal_document_manager/step8_embed_mmembed.py \
#     --corpus_in_filepath  /root/LILaC/datasets/MMWebQA/embeddings/serializations/page.json \
#     --out_embeddings_path /root/LILaC/datasets/MMWebQA/embeddings/MM-Embed/page.pt \
#     --out_index_path      /root/LILaC/datasets/MMWebQA/embeddings/MM-Embed/page.json \
#     --num_gpus 4 \
#     --max_length 4096 \
#     --batch_size 1