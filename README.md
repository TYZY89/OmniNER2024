# OmniNER2024

## Acknowledgments

**Our code utilizes the framework from the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). We extend our gratitude to the contributors of this project. Additionally, we declare that the code we use adheres to the original project's copyright and licensing agreements.**

## Getting Started

### Installation

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

### Using OmniNER2024

To use the OmniNER2024 benchmark with the LLaMA-Factory framework, follow these steps:

1. **Update `data/dataset_info.json`**: Add dataset information for OmniNER2024 to the `data/dataset_info.json` file. Ensure the format and details are consistent with existing entries.

2. **Add YAML Files**: Add the YAML files from this repository (e.g., examples/train_lora/qwen2_7b_instruct_OmniNER_benchmarks/qwen7b_lora_sft.yaml) to the `examples/train_lora` directory in the llama-factory framework. Ensure the paths and dataset names in the YAML files match the entries in `data/dataset_info.json`.

> **Note:**
> **The OmniNER2024 benchmark is not yet open-sourced. However, we will be open-sourcing it soon and will update the repository with the benchmark once it is available.**

### Quickstart

Use the following commands to run LoRA **fine-tuning** and **prediction** for the Qwen2-7B-Instruct model, respectively.

#### Fine-tuning

```bash
llamafactory-cli train examples/train_lora/qwen7b_lora_sft.yaml
```

#### Prediction

```bash
llamafactory-cli train examples/train_lora/qwen7b_lora_predict.yaml
```

#### Evaluation

To evaluate the model on the OmniNER2024 benchmark, use the evaluate.py script. Here is how you can run the evaluation:

```bash
python examples/evaluate.py --path <path_to_predicted_jsonl> 
```
Replace **<path_to_predicted_jsonl>** with the path to the JSONL file generated from the prediction step.

### OmniNER2024 using BERT

For instructions on using OmniNER2024 with BERT, please refer to the examples in the `examples/BERT-NER` directory in this repository. This directory contains detailed examples and scripts for fine-tuning and evaluating BERT models on the OmniNER2024 dataset.

### ERRTA

```bash
python ERRTA/badcase_analysis.py
python ERRTA/ERRTA_badcase_analysis.py
```

