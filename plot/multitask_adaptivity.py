import pandas as pd 

df = pd.DataFrame(pd.read_csv('result/BM/titan_True.csv')).dropna() 
completition_list = df.completion_time.tolist() 
submission_list = df.submission_time.tolist() 
jct_list = [end - start for end, start in zip(completition_list, submission_list)]
name_list = df.name.tolist() 
jct_name_pair_mt = sorted([(jct, str(name)) for jct, name in zip(jct_list, name_list)])

df = pd.DataFrame(pd.read_csv('result/BM/titan_False.csv')).dropna() 
completition_list = df.completion_time.tolist() 
submission_list = df.submission_time.tolist() 
jct_list = [end - start for end, start in zip(completition_list, submission_list)]
name_list = df.name.tolist() 

mt_better = 0
no_mt_better = 0
jct_name_pair_no_mt = sorted([(jct, str(name)) for jct, name in zip(jct_list, name_list)])
job_cnt = 0
for jct_name in jct_name_pair_no_mt: # [-20:]: 
    jct, job_name = jct_name 
    for jct_name_mt in jct_name_pair_mt: 
        if jct_name_mt[1] == job_name: 
            print('job name {}, with mt {}, no mt {}'.format(job_name, jct_name_mt[0], jct))
            if jct / jct_name_mt[0] > 1.1:
                mt_better += 1
            if jct_name_mt[0] / jct > 1.1: 
                no_mt_better += 1 

print('mt_better {}, no_mt_better {}'.format(mt_better, no_mt_better))
# import pdb; pdb.set_trace() 
tot = [160, 320, 480, 720]
reduction = [30,61,97,140]
increase = [10, 28, 34, 83]
for t, r, i in zip(tot, reduction, increase): 
    print(r / t, " ", i / t)