# Project Structure
`./llama-7b` store LLM (llama-2-7b-hf) Weights.

`./checkpoints` store SFT checkpoinys.

`./datasets` store datasets of ABR and CC, include raw time series csv files and pkl.

`./demo` store the api to evaluate performance on directly used LLMs without pretrain.

`./environment` store the network emulation env.

`./logs` store the train data.

`./plots_code` store the code to plot.

`./src` store the core llm model, include Inference,Real-time Planning,Training.

`./swap` is a temporary file during the interaction between env and llm.

# Run Inference
inference means use collected dataset to evaluate performance.
run: `python ./src/inference.py`

# Run Real-time Planning
Real-time Planning means real-time interraction between llm and modified env.

for ABR, run:
1. start `python ./src/Plan_precudure.py` , use `llmcc` python env.
2. Then start `python environments/adaptive_bitrate_streaming/run_baseline_llmcc.py`, use `abr_tf` python env. Note, start at the file root.

# Run Training
training means use collected dataset to train model.

# Collecting Dataset
