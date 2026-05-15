
DEVICENUM=0

# List of datasets you want to run on
DATASETS=("MultimodalQA" "MMCoQA" "MP-DocVQA" "SlideVQA" "InfoVQA")

########################################
# 1) Parameter sensitivity for beam_width
########################################

# Array of beam_widths to explore
BEAM_WIDTHS=(1 2 3 4 5 10 20 30)

for DS in "${DATASETS[@]}"; do
  for BW in "${BEAM_WIDTHS[@]}"; do
    # Generate a descriptive run name
    RUN_NAME="parametersensitivity_ni1_bw${BW}"
    echo "==== Running parameter sensitivity for beam_width=$BW on dataset=$DS ===="

    # Run the retriever
    CUDA_VISIBLE_DEVICES=$DEVICENUM python3 src/lilac/retriever/retriever.py \
      --target_dataset "$DS" \
      --run_mode late_interaction \
      --parameter_numiterations 1 \
      --parameter_beamwidth "$BW" \
      --run_name "$RUN_NAME" \
      --force_overwrite True

    echo "---- Completed for beam_width=$BW dataset=$DS ----"
    echo
  done
done

########################################
# 2) Parameter sensitivity for num_iterations
########################################

# Array of iteration counts to explore
NUM_ITERATIONS=(1 2)

for DS in "${DATASETS[@]}"; do
  
  echo "==== Running parameter sensitivity for num_iterations=0 on dataset=$DS ===="
  
  RUN_NAME="parametersensitivity_ni0_bw5"
  CUDA_VISIBLE_DEVICES=$DEVICENUM python3 src/lilac/retriever/retriever.py \
  --target_dataset "$DS" \
  --run_mode single_knn \
  --parameter_beamwidth 5 \
  --parameter_targetlevel top \
  --run_name "$RUN_NAME" \
  --force_overwrite True

  for NI in "${NUM_ITERATIONS[@]}"; do

    RUN_NAME="parametersensitivity_ni${NI}_bw5"
    echo "==== Running parameter sensitivity for num_iterations=$NI on dataset=$DS ===="

    CUDA_VISIBLE_DEVICES=$DEVICENUM python3 src/lilac/retriever/retriever.py \
      --target_dataset "$DS" \
      --run_mode late_interaction \
      --parameter_beamwidth 5 \
      --parameter_numiterations "$NI" \
      --run_name "$RUN_NAME" \
      --force_overwrite True
      
    echo "---- Completed for num_iterations=$NI dataset=$DS ----"
    echo
  done
done

echo "All experiments completed!"

