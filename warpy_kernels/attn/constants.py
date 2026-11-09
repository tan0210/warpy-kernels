import math

LOG2E = math.log2(math.e)
# E4M3 probabilities are scaled by `2^P_OFFSET` to stay above the subnormal range
P_OFFSET = 8.0
E4M3_ONE_X4 = 0x38383838  # four E4M3 ones packed in one register
