## Chinese NER using Bert

BERT for Chinese NER. 

**update**：其他一些可以参考,包括Biaffine、GlobalPointer等:[examples](https://github.com/lonePatient/TorchBlocks/tree/master/examples)

### dataset list

1. cner: datasets/cner
2. CLUENER: https://github.com/CLUEbenchmark/CLUENER
3. **OmniNER2024: we will be open-sourcing it soon and will update the repository with the benchmark once it is available.**

### model list

1. BERT+Softmax
2. BERT+CRF
3. BERT+Span

### input format

Input format (prefer BIOS tag scheme), with each character its label for one line. Sentences are splited with a null line.

```text
美	B-LOC
国	I-LOC
的	O
华	B-PER
莱	I-PER
士	I-PER

我	O
跟	O
他	O
```

### run the code

1. Modify the configuration information in `run_ner_xxx.py` or `run_ner_xxx.sh` .
2. `cd example/BERT-NER`
3. `sh scripts/run_ner_xxx.sh`

**note**: file structure of the model

```
├── example
|  └──BERT-NER
|  |  └──prev_trained_model
|  |  |  └── bert_base
|  |  |  └── pytorch_model.bin
|  |  |  └── config.json
|  |  |  └── vocab.txt
|  |  |  └── ......
|  |  └──scripts
|  |  |  └── run_ner_softmax.sh
|  |  |  └── run_ner_span.sh
|  |  |  └── run_ner_crf
|  |  └──datasets
|  |  |  └── omniner
|  |  |  └── ......
```
