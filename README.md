# OmniNER2024

## Acknowledgments

Our code utilizes the framework from the [LLaMA-Factory project on GitHub](https://github.com/hiyouga/LLaMA-Factory). We extend our gratitude to the contributors of this project. Additionally, we declare that the code we use adheres to the original project's copyright and licensing agreements.

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

2. **Add YAML Files**: Add YAML files (e.g. examples/train_lora/qwen2_7b_instruct_OmniNER_benchmarks/qwen7b_lora_sft.yaml) for training and evaluation into the `examples/train_lora` directory. Ensure the paths and dataset names in the YAML files match the entries in `data/dataset_info.json`.

**Note:** **OmniNER2024 benchmark is not yet open-sourced. We will update the repository with the benchmark once it is available.**

### Quickstart

Use the following commands to run LoRA **fine-tuning** and **prediction** for the Qwen2-7B-Instruct model, respectively.

#### Fine-tuning

```bash
llamafactory-cli train examples/train_lora/qwen7b_lora_sft.yaml
```

#### Prediction

```bash
llamafactory-cli predict examples/train_lora/qwen7b_lora_predict.yaml
```

### Evaluation

To evaluate the model on the OmniNER2024 benchmark, use the evaluate.py script. Here is how you can run the evaluation:

```bash
python evaluate.py --path <path_to_predicted_jsonl> 
```
Replace **<path_to_predicted_jsonl>** with the path to the JSONL file generated from the prediction step.
