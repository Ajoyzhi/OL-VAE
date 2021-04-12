# main文件所在目录为F:/pycharm/OL_VAE/main
# kdd99训练集的文件路径
Train_Src_Path = '../dataset/KDD99/raw/kddcup.data_10_percent_corrected'
# 数值化的训练数据文件路径
Train_Number_Path = '../dataset/KDD99/raw/kddcup.data_10_percent_corrected_number.cvs'
# 选择特征后的训练数据文件路径
Train_Feature_Path =  '../dataset/KDD99/raw/kddcup.data_10_percent_corrected_feature.cvs'
# 按比例选择后的数据，标准化和归一化之后的训练数据（最终使用的训练数据）
Train_Des_Path = '../dataset/KDD99/raw/kddcup.data_10_percent_corrected_destination.cvs'

# 测试集的文件路径
Test_Src_Path = '../dataset/KDD99/raw/corrected'
# 数值化的训练数据文件路径
Test_Number_Path = '../dataset/KDD99/raw/corrected_number.cvs'
# 选择特征后的训练数据文件路径
Test_Feature_Path =  '../dataset/KDD99/raw/corrected_feature.cvs'
# 按比例选择后的数据，标准化和归一化之后的训练数据（最终使用的训练数据）
Test_Des_Path = '../dataset/KDD99/raw/corrected_destination.cvs'

# log输出位置
Train_Log_Path = '../other/log/train/'
Test_Log_Path = '../other/log/test/'
# 图像输出位置
Picture = '../other/pictures/'
# acc pre FPR detection_time train_time
Performance = '../other/performance/'

# 网络参数保存位置
Model = '../other/'

# SDN data local(src:raw data;dst:comput the increse rate)
raw_root = '../dataset/SDN/raw/rawraw/'
pro_root = '../dataset/SDN/raw/'