from matmul import matmul
from quantizer.fp4 import FP4_Quantizer_Blockwise, NF4_Quantizer_Blockwise
import torch

if __name__ == "__main__":
    BLOCK_SIZE = 64

    quantizer = FP4_Quantizer_Blockwise(block_size=BLOCK_SIZE)

    in_features, out_features = 1024, 512
    weights = torch.randn(out_features, in_features).to(torch.bfloat16)
    input_tensor = torch.randn(1, in_features).to(torch.bfloat16)

    matmul_operation = matmul(quantizer = quantizer)
    base_matmul_result = matmul_operation(input_tensor, weights, weights_quantized=False)

    quantized_weight, scales = quantizer.quantize(weights)
    dequantized_matmul_result = matmul_operation(input_tensor, quantized_weight, weights_quantized=True,scales=scales, shape=weights.shape)
    
    mean_fp4_error = (base_matmul_result - dequantized_matmul_result).abs().mean()
    print("Matmul FP4 Error", mean_fp4_error)

    print("---------------------------------------------")

    BLOCK_SIZE = 64

    quantizer = NF4_Quantizer_Blockwise(block_size=BLOCK_SIZE)

    in_features, out_features = 1024, 512
    weights = torch.randn(out_features, in_features).to(torch.bfloat16)
    input_tensor = torch.randn(1, in_features).to(torch.bfloat16)

    matmul_operation = matmul(quantizer = quantizer)
    base_matmul_result = matmul_operation(input_tensor, weights, weights_quantized=False)

    quantized_weight, scales = quantizer.quantize(weights)
    dequantized_matmul_result = matmul_operation(input_tensor, quantized_weight, weights_quantized=True,scales=scales, shape=weights.shape)
    
    mean_nf4_error = (base_matmul_result - dequantized_matmul_result).abs().mean()
    print("Matmul NF4 Error", mean_nf4_error)