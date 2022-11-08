batch_list = [1 for _ in range(16)] # [4, 8, 10, 12, 16, 18, 20, 24]
allocation_list  = [1, 2] + [4 * i for i in range(1, 17)]
amp_list = [0, 1]
ckpt_list = [0, 1]
layer_list = [i for i in range(12)]


time_cost = 0
cnt = 0
for batch in batch_list: 
    for allocation in allocation_list: 
        for amp in amp_list: 
            for ckpt in ckpt_list: 
                for layer in layer_list: 
                    cnt += 1
                    time_cost += 20 * allocation / 64
print(time_cost / 3600)
print(cnt)