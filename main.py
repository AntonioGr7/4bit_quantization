from quantizer.fp4 import FP4_Quantizer, FP4_Quantizer_Blockwise, NF4_Quantizer_Blockwise
import torch

if __name__ == "__main__":
    input_tensor = torch.randn((4, 512))
    
    print("Testing FP4 Quantizer")
    quantizer = FP4_Quantizer()
    print("Input Tensor:", input_tensor)
    quantized_tensor, scale = quantizer.quantize(input_tensor=input_tensor)
    print("Quantized Tensor:", quantized_tensor)
    dequantized_tensor = quantizer.dequantize(quantized_tensor, scale, input_tensor.shape)
    print("Dequantized Tensor:", dequantized_tensor)
    print("Errors:", input_tensor - dequantized_tensor)
    simple_f24_mean_error = (input_tensor - dequantized_tensor).abs().mean()
    print("Mean Error:", simple_f24_mean_error)

    print("--------------------------------------------")
    fp4_errors = []
    for block_size in [2, 8, 16, 32, 64,512, 512*4]:
        print("Testing FP4 Quantizer Blockwise")
        quantizer = FP4_Quantizer_Blockwise(block_size=block_size)
        print("Input Tensor:", input_tensor)
        quantized_tensor, scale = quantizer.quantize(input_tensor=input_tensor)
        print("Quantized Tensor:", quantized_tensor)
        dequantized_tensor = quantizer.dequantize(quantized_tensor, scale, input_tensor.shape)
        print("Dequantized Tensor:", dequantized_tensor)
        print("Errors:", input_tensor - dequantized_tensor)
        blockwise_mean_error = (input_tensor - dequantized_tensor).abs().mean()
        fp4_errors.append(blockwise_mean_error)
        print("Mean Error:", blockwise_mean_error)

    print("--------------------------------------------")
    nf4_errors = []
    for block_size in [2, 8, 16, 32, 64,512, 512*4]:
        print("Testing NF4 Quantizer Blockwise")
        quantizer = NF4_Quantizer_Blockwise(block_size=block_size)
        print("Input Tensor:", input_tensor)
        quantized_tensor, scale = quantizer.quantize(input_tensor=input_tensor)
        print("Quantized Tensor:", quantized_tensor)
        dequantized_tensor = quantizer.dequantize(quantized_tensor, scale, input_tensor.shape)
        print("Dequantized Tensor:", dequantized_tensor)
        print("Errors:", input_tensor - dequantized_tensor)
        blockwise_mean_error = (input_tensor - dequantized_tensor).abs().mean()
        nf4_errors.append(blockwise_mean_error)
        print("Mean Error:", blockwise_mean_error)

    print("Comparison of Mean Errors:")
    print("Simple FP4 Mean Error:", simple_f24_mean_error.item())
    print("----")
    print("FP4 Blockwise Mean Errors:")
    for block_size, mean_error in zip([2, 8, 16, 32,64,512, 512*4], fp4_errors):
        print(f"Block Size {block_size} Mean Error:", mean_error.item())
    print("NF4 Blockwise Mean Errors:")
    for block_size, mean_error in zip([2, 8, 16, 32,64,512, 512*4], nf4_errors):
        print(f"Block Size {block_size} Mean Error:", mean_error.item())


        
