import torch
import torch.nn.functional as F

""" 
Simple implementation of Flash Attention in torch for my understanding. 
The core parts are the sum and max updates in the inner loop
At the end I calculate output directly to test if everything is working fine
"""

sequence_query_len = 256
sequence_kv_len = 256
head_dim = 512
block_size_kv = 64
block_size_query = 64

# Create random matrices representing query, key and value used in attention
matrix_query = torch.rand(sequence_query_len, head_dim)
matrix_key = torch.rand(sequence_kv_len,   head_dim)
matrix_value = torch.rand(sequence_kv_len, head_dim)

output_matrix = torch.zeros(sequence_query_len, head_dim)
max_row = torch.full((sequence_query_len,), float('-inf'))
total_exp_sum = torch.zeros(sequence_query_len)
num_blocks_query = (sequence_query_len + block_size_query - 1) // block_size_query

num_blocks_kv = sequence_kv_len//block_size_kv
num_blocks_kv = (sequence_kv_len + block_size_kv - 1) // block_size_kv

# Loop algorithm described in Flash Attention paper 
# Outer loop goes over key,value matrix block
# Inner loop goes over query matrix block
for j in range(0, num_blocks_kv):
    start_idx = j*block_size_kv
    end_idx = min((j+1)*block_size_kv, sequence_kv_len)
    block_key = matrix_key[start_idx:end_idx, :]
    block_value = matrix_value[start_idx:end_idx, :]

    for i in range(0, num_blocks_query):
        start_idx = i*block_size_query
        end_idx = min((i+1)*block_size_query, sequence_query_len)
        block_query = matrix_query[start_idx:end_idx, :]
        block_output = output_matrix[start_idx:end_idx, :]
        current_max_row = max_row[start_idx:end_idx]
        current_sum_row = total_exp_sum[start_idx:end_idx]

        block_result = torch.matmul(block_query, block_key.T)
        max_values, _ = torch.max(block_result, dim=1)
        block_exp = torch.exp(block_result - max_values.view(-1, 1))
        row_exp_sum = torch.sum(block_exp, dim=1)

        # calculate new max by comparing max values in block_result with max values stored in the max row tensor
        new_max = torch.where(max_values > current_max_row, max_values, current_max_row)
        # calculate exponential sums of rows in block_result
        new_exp_sum = torch.exp(current_max_row - new_max)*current_sum_row + torch.exp(max_values - new_max)*row_exp_sum

        term_1 = current_sum_row.view(-1, 1)*(torch.exp(current_max_row - new_max).view(-1, 1)*block_output)
        term_2 = torch.exp(max_values - new_max).view(-1, 1)*torch.matmul(block_exp, block_value)
        # update output matrix
        output_matrix[i*block_size_query:(i+1)*block_size_query, :] = (1/new_exp_sum).view(-1, 1)*(term_1 + term_2)

        max_row[i*block_size_query:(i+1)*block_size_query] = new_max
        total_exp_sum[i*block_size_query:(i+1)*block_size_query] = new_exp_sum


# output_matrix_direct computed by torch.softmax applied first should be equal to output_matrix  

softmax_matrix = F.softmax(torch.matmul(matrix_query, matrix_key.T), dim=1)
output_matrix_direct = torch.matmul(softmax_matrix, matrix_value)
