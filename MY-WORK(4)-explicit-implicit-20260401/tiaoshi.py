# import pickle
# with open('/media/user/data/KITTI-Seg/semantickitti_infos_train.pkl', 'rb') as f:
#     data = pickle.load(f)
# print(data['data_list'][0].keys())
# # 看看有没有 'images' 或 'img_path' 相关的字段

import pickle
with open('/media/user/data/KITTI-Seg/semantickitti_infos_train.pkl', 'rb') as f:
    data = pickle.load(f)
print(data['data_list'][0].keys())
# 应该能看到 'images' 和 'calib_path' 字段了s