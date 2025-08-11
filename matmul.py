import torch

class matmul():
  def __init__(self,quantizer):
    self.quantizer = quantizer
  def __call__(self, input_tensor, weights, scales=None, weights_quantized=False, shape=None):
    if weights_quantized:
      if shape is None or scales is None:
        raise Exception("'shape' and 'scales' are required")
      weights = self.quantizer.dequantize(weights, scales, shape)
      weights = weights.to(torch.bfloat16)
    output = torch.matmul(input_tensor, weights.T)
    return output