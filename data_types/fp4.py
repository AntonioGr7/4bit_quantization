class FP4_E2M1:
  '''
  class that represent the E2M1 format
  '''
  def __init__(self):
    self.values = []
    for sign in [0,1]:
      for exp in range(2**2):
        for mantissa in range(2):
          if exp==0 and mantissa == 0:
            value = 0
          else:
            exp_val = exp-1
            mantissa_val = 1+mantissa*0.5
            value = (1 if sign==0 else -1) * mantissa_val * (2**(exp_val))
         
          if value not in self.values:
            self.values.append(value)
    self.values = sorted(self.values)


