# Running Local LLMs

A comprehensive guide to running Large Language Models (LLMs) on your local machine using various frameworks and tools.

[![GitHub stars](https://img.shields.io/github/stars/yourusername/local-llms.svg)](https://github.com/yourusername/local-llms/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Frameworks and Tools](#frameworks-and-tools)
  - [llama.cpp](#llamacpp)
  - [Ollama](#ollama)
  - [HuggingFace Transformers](#huggingface-transformers)
  - [HuggingFace Transformers - Quantized (BitsAndBytes)](#huggingface-transformers---quantized-bitsandbytes)
  - [TorchAO](#torchao)
  - [vLLM](#vllm)
  - [GPT-NeoX] (#GPT-Neox)
  - [Triton] - (# Triton Inference Server TensorRT backend)
  - [LM Studio](#lm-studio)
- [Performance Comparison](#performance-comparison)
- [Contributing](#contributing)
- [License](#license)

## Overview

Running LLMs locally offers several advantages including privacy, offline access, and cost efficiency. This repository provides step-by-step guides for setting up and running LLMs using various frameworks, each with its own strengths and optimization techniques.

## Requirements

General requirements for running LLMs locally:

- **Hardware**:
  - **CPU**: Modern multi-core processor (8+ cores recommended)
  - **RAM**: 16GB minimum, 32GB+ recommended
  - **GPU**: NVIDIA GPU with 8GB+ VRAM (16GB+ recommended for larger models)
  - **Storage**: 20GB+ free space (varies by model size)
- **Software**:
  - Python 3.8+
  - CUDA 11.7+ and cuDNN (for GPU acceleration)
  - Git

Specific requirements are listed in each framework section.

## Frameworks and Tools

### llama.cpp

[llama.cpp](https://github.com/ggerganov/llama.cpp) is a C/C++ implementation of LLaMA that's optimized for CPU and GPU inference.

#### Installation

```bash
# Clone the repository
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build the project
make

# Download and convert a model (example with TinyLlama)
python3 -m pip install torch numpy sentencepiece
python3 scripts/convert.py <path_to_hf_model>

# Quantize the model (optional)
./quantize <path_to_model>/ggml-model-f16.bin <path_to_model>/ggml-model-q4_0.bin q4_0
```

#### Usage

```bash
# Run inference
./main -m <path_to_model>/ggml-model-q4_0.bin -n 512 -p "Write a short poem about programming:"
```

#### Advantages

- Extremely memory-efficient through quantization
- Works well on CPU-only setups
- Supports various model architectures (LLaMA, Mistral, Falcon, etc.)
- Available as a library for integration into other applications

### Ollama

[Ollama](https://github.com/ollama/ollama) provides an easy way to run open-source LLMs locally with a simple API.

#### Installation

```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# Download from https://ollama.com/download/windows
```

#### Usage

```bash
# Pull a model
ollama pull mistral

# Run a model
ollama run mistral

# API usage
curl -X POST http://localhost:11434/api/generate -d '{
  "model": "mistral",
  "prompt": "What is computational linguistics?"
}'
```

#### Advantages

- Simplified setup and usage
- Integrated model library with one-command downloads
- REST API for easy integration
- Cross-platform support
- No Python environment needed

### HuggingFace Transformers

[HuggingFace Transformers](https://github.com/huggingface/transformers) is a popular library that provides thousands of pre-trained models.

#### Installation

```bash
pip install transformers torch
```

#### Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"  # Requires HF auth token
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Generate text
inputs = tokenizer("Write a short story about:", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

#### Advantages

- Extensive model support
- Easy integration with PyTorch ecosystem
- Rich documentation and community support
- Seamless model switching

### HuggingFace Transformers - Quantized (BitsAndBytes)

Quantization with BitsAndBytes allows running larger models with reduced memory requirements.

#### Installation

```bash
pip install transformers torch bitsandbytes accelerate
```

#### Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Configure quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# Load model with quantization
model_name = "meta-llama/Llama-2-13b-hf"  # Requires HF auth token
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)

# Generate text
inputs = tokenizer("Explain quantum computing:", return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

#### Advantages

- Run larger models on consumer hardware
- Minimal performance impact despite compression
- Compatible with most HuggingFace models
- 4-bit and 8-bit quantization options

### TorchAO

[TorchAO](https://github.com/pytorch/ao) (PyTorch Ahead-of-Time Optimization) enables efficient inference through quantization and optimization techniques.

#### Installation

```bash
pip install torch torchao
```

#### Usage

```python
import torch
import torchao
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Apply TorchAO optimizations
optimized_model = torchao.optimize(
    model,
    quantization=True,
    dtype=torch.float16,
    inplace=False
)

# Generate text
inputs = tokenizer("Explain how solar panels work:", return_tensors="pt")
outputs = optimized_model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

#### Advantages

- Advanced quantization techniques
- Hardware-specific optimizations
- Compatible with PyTorch ecosystem
- Flexible configuration options


### GPT-NeoX - (#GPT-Neox)

GPT-NeoX is a series of models developed by EleutherAI and
built using a modified version of Megatron. It offers various
sizes, ranging from 20B to 100B parameters.

#### Installation

```bash
pip install gpt-neo
```

#### Usage

```python
from gpt_neo import GPTPredictor
from gpt_neo.utils import download_model

# Download the model (requires EleutherAI auth token)
download_model("gpt-neo-2.7B")  # Change the model size as
needed

# Load and initialize the model predictor
predictor = GPTPredictor.from_pretrained("gpt-neo-2.7B",
device="cuda:0")

# Generate text
inputs = "Write a short story about:"
outputs = predictor(inputs, max_length=100)
print(outputs["text"])
```

#### Advantages

- Large model capacity
- Open source and transparent development
- Supports multiple architectures (e.g., BERT, Megatron)
- Strong performance on various tasks


### Triton Inference Server (TensorRT backend)

Triton Inference Server is an open-source, extensible, and
customizable AI inferencing server developed by NVIDIA. It
supports a wide variety of models, including LLMs like BERT,
RoBERTa, DistilBert, and Electra, among others.


#### Installation 

1. Install Triton Inference Server: Follow the instructions
for [installing Triton Inference
Server](https://github.com/NVIDIA/Triton/blob/main/docs/quicksServer](https://github.com/NVDIA/Triton/blob/main/docs/quickstarts/quickstart-docker.md).


#### Usage

1. Prepare the model: Convert your Hugging Face Transformers
model to an ONNX format using [the ONNX model
converter](https://github.com/onnx/onnx/blob/master/tools/onnxconverter](https//github.com/onnx/onnx/blob/master/tools/onnxtools/README.md).
2. Export the ONNX model for TensorRT: Use the [TensorRT
optimizer](https://github.com/NVIDIA/TF-TRT) to convert your
ONNX model to a TensorRT engine.
3. Deploy the model in Triton Inference Server: Create and
configure a `model_config.pbtxt` file for your model. Then,
create and register a Triton model server with the
configuration file and the TensorRT engine.
4. Use the model in your application: Integrate Triton
Inference Server into your application using the Triton
Inference Client Library or any other method that suits your
needs.


#### Advantages

- High performance through the use of TensorRT optimizations
- Flexible deployment options (e.g., on-premises, in the
cloud)
- Supports a wide variety of models and customization
- Efficient handling of multiple models at scale


### vLLM

[vLLM](https://github.com/vllm-project/vllm) is a high-throughput and memory-efficient inference engine.

#### Installation

```bash
pip install vllm
```

#### Usage

```python
from vllm import LLM, SamplingParams

# Initialize the model
llm = LLM(model="meta-llama/Llama-2-7b-hf")  # Requires HF auth token

# Set sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=100
)

# Generate text
prompts = ["Write a short story about space exploration:"]
outputs = llm.generate(prompts, sampling_params)

# Print generated text
for output in outputs:
    print(output.text)
```

#### Command-line usage

```bash
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-hf
```

#### Advantages

- PagedAttention for efficient memory usage
- Multi-GPU support with tensor parallelism
- OpenAI-compatible API
- High throughput for batch processing
- Efficient continuous batching

### LM Studio

[LM Studio](https://lmstudio.ai/) is a desktop application for running local LLMs with a graphical interface.

#### Installation

1. Download the installer from [https://lmstudio.ai/](https://lmstudio.ai/)
2. Install and launch the application

#### Usage

1. Download models from the built-in model library
2. Configure inference parameters using the GUI
3. Chat with the model through the interface
4. Optionally expose an API server compatible with OpenAI

#### Advantages

- User-friendly GUI
- No coding required
- Built-in model discovery and management
- Visual parameter tuning
- Performance metrics visualization
- Compatible with various model formats

## Performance Comparison

| Framework             | Memory Usage | Inference Speed | Setup Complexity | GPU Support | CPU Support |
|-----------------------|--------------|-----------------|------------------|-------------|-------------|
| llama.cpp             | Very Low     | Moderate        | Moderate         | Good        | Excellent   |
| Ollama                | Low          | Good            | Very Low         | Good        | Good        |
| HF Transformers       | High         | Moderate        | Low              | Excellent   | Good        |
| HF - BitsAndBytes     | Moderate     | Good            | Low              | Excellent   | Limited     |
| TorchAO               | Moderate     | Good            | Moderate         | Excellent   | Good        |
| vLLM                  | Moderate     | Excellent       | Moderate         | Excellent   | Limited     |
| LM Studio             | Varies       | Good            | Very Low         | Good        | Good        |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
