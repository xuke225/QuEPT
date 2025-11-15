import pickle

def find_optimal_bit_allocation(bit, save_path, num_blocks):
    n_blocks = num_blocks # 32/288
    bits_min = 2
    bits_max = 8
    avg_bit = bit
    total_bits = avg_bit * n_blocks 

    with open(save_path, 'rb') as file:

        SEN_TABLE_LLAMA7B_PER_LAYER =  pickle.load(file)



    dp = [{} for _ in range(n_blocks + 1)]
    dp[0][0] = 0.0

    path = [{} for _ in range(n_blocks + 1)]

    for k in range(1, n_blocks + 1):
        block_idx = k - 1
        prev_states = dp[k-1]
        for s_prev, sen_prev in prev_states.items():
            for b in range(bits_min, bits_max + 1):
                s_current = s_prev + b
                remaining_blocks = n_blocks - k
                lower_bound = max(bits_min * k, total_bits - remaining_blocks * bits_max)
                upper_bound = min(bits_max * k, total_bits - remaining_blocks * bits_min)
                if s_current < lower_bound or s_current > upper_bound:
                    continue
                current_sen = sen_prev + SEN_TABLE_LLAMA7B_PER_LAYER[b - 2][block_idx]
                if s_current not in dp[k] or current_sen < dp[k][s_current]:
                    dp[k][s_current] = current_sen
                    path[k][s_current] = (s_prev, b)

    if total_bits not in dp[n_blocks]:
        return None

    bits_config = []
    current_s = total_bits
    for k in range(n_blocks, 0, -1):
        s_prev, b = path[k][current_s]
        bits_config.append(b)
        current_s = s_prev
    bits_config.reverse()

    return bits_config, dp[n_blocks][total_bits]

def pickle_dump_model(model,filepath):
    with open (filepath, 'wb') as f: 
        pickle.dump(model, f)

