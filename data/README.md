# Data

This directory contains the prompt and metadata files used by the released training example and paper evaluation protocol.

The GenEval, OCR, DrawBench, and Pick-a-Pic split files are kept unchanged from the public DiffusionOPD evaluation release.

- `ocr/train.txt` and `ocr/test.txt` contain the text-rendering prompt splits. The default public training configuration uses `ocr/train.txt`.
- `geneval/train_metadata.jsonl` and `geneval/test_metadata.jsonl` contain GenEval prompts and structured requirements; `geneval/object_names.txt` contains the COCO detector labels. See the [GenEval project](https://github.com/djghosh13/geneval) for the benchmark and original licensing terms.
- `pickapic/train.txt` and `pickapic/test.txt` contain preference prompts derived from the public [Pick-a-Pic](https://huggingface.co/datasets/yuvalkirstain/pickapic_v1) captions. The test split is used for the separate-set HPSv2 evaluation.
- `drawbench/test.txt` contains the held-out DrawBench prompts used for the separate-set PickScore evaluation. See the [Imagen project](https://imagen.research.google/) for the benchmark source.

Text files contain one prompt per line. GenEval JSONL rows contain a prompt, a task tag, and structured inclusion/exclusion requirements. Only the OCR training recipe is enabled by the default configuration; the remaining files support evaluation and future public reward configurations.
