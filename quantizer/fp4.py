from data_types.fp4 import FP4_E2M1
import torch

class FP4_Quantizer():
  def __init__(self):
    self.fp4_values = torch.tensor(FP4_E2M1().values)
  def quantize(self, input_tensor):
    block = input_tensor.view(-1) # Flatten
    scale = block.abs().max() # Get the max value of the block for the scale
    if scale == 0:
      return torch.zeros_like(block), scale
    scaled_block =block/scale # Scale the tensor 
    
  
    indices = torch.argmin(torch.abs(scaled_block.unsqueeze(1)-self.fp4_values),dim=1) # Find the nearest value from the range
    quantized_data = self.fp4_values[indices]
    return quantized_data, scale

  def dequantize(self,quantized_tensor,scale,original_shape):
    t = quantized_tensor*scale
    t = t.reshape(original_shape)
    return t 

class FP4_Quantizer_Blockwise():
  def __init__(self,block_size=8):
    self.fp4_values = torch.tensor(FP4_E2M1().values)
    self.block_size = block_size

  def quantize(self,input_tensor):
    data_flat = input_tensor.view(-1) # Flatten
    num_blocks = (data_flat.numel()+ self.block_size -1) // self.block_size
    quantized_data = torch.zeros(num_blocks * (self.block_size//2), dtype=torch.uint8) # Every 8 bit we'll pack together 2 tensor of 4 bit
    scales = torch.zeros(num_blocks)

    for i in range(num_blocks):
      start = i*self.block_size
      end = min((i+1)*self.block_size,data_flat.numel())
      block = data_flat[start:end]
      scale = block.abs().max() # Get the max value of the block for the scale
      if scale == 0:
        scale = 1.0
      scales[i] = scale # Saving the scale factor for the block

      scaled_block = block/scale # Scale the tensor 
      indices = torch.argmin(torch.abs(scaled_block.unsqueeze(1)-self.fp4_values),dim=1) # Find the nearest value
      # Combine two 4 bit indices in one uint8 value
      # This operation refactor the indices organizing it in group of two [[1,2],[3,4]...]
      # Then pack the values of the first column with the second column moving this one 4bit to the left (left bit shift operator)
      # For example if the index is 5 (0101) shifting it left will result in 0101 0000 (80)
      if indices.numel() % 2 != 0:
        # Pad with a dummy value to make the number of elements even
        indices = torch.cat((indices, torch.tensor([0], dtype=indices.dtype)))

      packed_indices = indices.view(-1,2) 
      packed_values = packed_indices[:, 0] | (packed_indices[:, 1] << 4) 
      quantized_data[i * (self.block_size // 2) : i * (self.block_size // 2) + packed_values.numel()] = packed_values
    return quantized_data, scales

  def dequantize(self,quantized_tensor,scales, original_shape):
    num_elements = torch.prod(torch.tensor(original_shape))
    dequantized_flat = torch.zeros(num_elements, dtype=torch.float32)

    num_blocks = scales.numel()
    current_index = 0
    for i in range(num_blocks):
      start = i * self.block_size
      end = min((i+1)*self.block_size, num_elements)
      current_block_size = end-start
      # How many 8-bit values to unpack for the current block
      packed_block_size = (current_block_size + 1) // 2
      packed_values = quantized_tensor[current_index:current_index+packed_block_size]
      # Unpack the values -> I need to do a bitwise operation the most signifanct bit will be the second index
      # The least significant bits will be the first index

      index_1 = packed_values & 0x0F
      index_2 = (packed_values >> 4) & 0x0F
      indices_unpacked =torch.stack([index_1,index_2], dim=1).view(-1)
      indices_unpacked = indices_unpacked[:current_block_size]
      fp4_block_value = self.fp4_values[indices_unpacked.long()]
      dequantized_flat[start:end] = fp4_block_value * scales[i]
      current_index += packed_block_size
    return dequantized_flat.view(original_shape)


class NF4_Quantizer_Blockwise():
  def __init__(self,block_size=8):
    self.nf4_values = torch.tensor([
        -1.0000, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0000,
         0.0796,  0.1609,  0.2461,  0.3379,  0.4407,  0.5626,  0.7229,  1.0000
    ], dtype=torch.float32) # Precomputed
    self.block_size = block_size

  def quantize(self,input_tensor):
    data_flat = input_tensor.view(-1) # Flatten
    num_blocks = (data_flat.numel()+ self.block_size -1) // self.block_size
    quantized_data = torch.zeros(num_blocks * (self.block_size//2), dtype=torch.uint8) # Every 8 bit we'll pack together 2 tensor of 4 bit
    scales = torch.zeros(num_blocks)

    for i in range(num_blocks):
      start = i*self.block_size
      end = min((i+1)*self.block_size,data_flat.numel())
      block = data_flat[start:end]
      scale = block.abs().max() # Get the max value of the block for the scale
      if scale == 0:
        scale = 1.0
      scales[i] = scale # Saving the scale factor for the block

      scaled_block = block/scale # Scale the tensor 
      indices = torch.argmin(torch.abs(scaled_block.unsqueeze(1)-self.nf4_values),dim=1) # Find the nearest value
      # Combine two 4 bit indices in one uint8 value
      # This operation refactor the indices organizing it in group of two [[1,2],[3,4]...]
      # Then pack the values of the first column with the second column moving this one 4bit to the left (left bit shift operator)
      # For example if the index is 5 (0101) shifting it left will result in 0101 0000 (80)
      if indices.numel() % 2 != 0:
        # Pad with a dummy value to make the number of elements even
        indices = torch.cat((indices, torch.tensor([0], dtype=indices.dtype)))

      packed_indices = indices.view(-1,2) 
      packed_values = packed_indices[:, 0] | (packed_indices[:, 1] << 4) 
      quantized_data[i * (self.block_size // 2) : i * (self.block_size // 2) + packed_values.numel()] = packed_values
    return quantized_data, scales

  def dequantize(self,quantized_tensor,scales, original_shape):
    num_elements = torch.prod(torch.tensor(original_shape))
    dequantized_flat = torch.zeros(num_elements, dtype=torch.float32)

    num_blocks = scales.numel()
    current_index = 0
    for i in range(num_blocks):
      start = i * self.block_size
      end = min((i+1)*self.block_size, num_elements)
      current_block_size = end-start
      # How many 8-bit values to unpack for the current block
      packed_block_size = (current_block_size + 1) // 2
      packed_values = quantized_tensor[current_index:current_index+packed_block_size]
      # Unpack the values -> I need to do a bitwise operation the most signifanct bit will be the second index
      # The least significant bits will be the first index

      index_1 = packed_values & 0x0F
      index_2 = (packed_values >> 4) & 0x0F
      indices_unpacked =torch.stack([index_1,index_2], dim=1).view(-1)
      indices_unpacked = indices_unpacked[:current_block_size]
      nf4_block_value = self.nf4_values[indices_unpacked.long()]
      dequantized_flat[start:end] = nf4_block_value * scales[i]
      current_index += packed_block_size
    return dequantized_flat.view(original_shape)