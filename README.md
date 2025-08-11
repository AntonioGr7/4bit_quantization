# FP4 Quantization

High-Level Overview of FP4 Quantization for LLMs FP4 (4-bit floating-point) quantization is a technique used to significantly reduce the memory footprint and computational cost of Large Language Models (LLMs). The core idea is to represent the high-precision floating-point numbers (like FP16 or FP32) that make up a model's weights and activations with a much smaller 4-bit floating-point format. This allows you to store a massive model in a fraction of the memory and perform computations faster.
Here's a high-level breakdown of how it works:

- The Challenge: LLMs are huge. A model like LLaMA-7B has 7 billion parameters, each typically stored as a 16-bit floating-point number. This requires roughly 14 GB of VRAM. This is a lot, and it limits who can run these models. The goal of quantization is to reduce this number.

- The Idea: Instead of using 16 bits to represent each number, we'll use only 4 bits. A 4-bit floating-point number has a much smaller range of values it can represent. This is a trade-off: we save memory and compute, but we lose some precision. The challenge is to do this in a way that the model's performance doesn't degrade too much.

The Core Process: Quantization and De-Quantization:

- Quantization: When a model is loaded, its high-precision weights are converted to the 4-bit format. This involves a scaling factor and a data type conversion. The key is to find the right scaling factor that minimizes the loss of information.

- De-Quantization (on-the-fly): During a forward pass (inference), the 4-bit weights are loaded from memory. However, to perform the actual matrix multiplication (the core operation in a Transformer's linear layer), the GPU's hardware often requires higher precision (e.g., FP16). So, the 4-bit weights are de-quantized back to a higher precision on the fly. The matrix multiplication is then performed in this higher precision, and the result is stored.

- Handling Outliers: A major issue with quantizing LLMs is the presence of "outliers." These are a few values in the weight or activation tensors that are much larger than the rest. A naive quantization scheme would be dominated by these outliers, making the rest of the values lose all their precision. Solutions like bitsandbytes' FP4 and NF4 handle this by using a small, high-precision representation for these outliers while quantizing the majority of the values to 4-bit. This is a "mixed-precision" approach within the 4-bit quantization.
