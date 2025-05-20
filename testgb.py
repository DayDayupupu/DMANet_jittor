import jittor as jt
import numpy as np

# 加载两个不同时间点的检查点
checkpoint1 = jt.load('E:\\guobiao\\jittor\\log\\20250517-184532\\checkpoints\\model_step_0')
checkpoint2 = jt.load('E:\\guobiao\\jittor\\log\\20250517-184532\\checkpoints\\model_step_2')

# 比较参数差异
for (name1, param1), (name2, param2) in zip(checkpoint1['state_dict'].items(), checkpoint2['state_dict'].items()):
    # 确保参数是Jittor张量
    if isinstance(param1, np.ndarray):
        param1 = jt.array(param1)
    if isinstance(param2, np.ndarray):
        param2 = jt.array(param2)

    diff = (param1 - param2).abs().mean()
    print(f"{name1}: 平均参数变化 = {diff}")