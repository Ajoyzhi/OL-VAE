import numpy as np
import csv

from sklearn import preprocessing

class Util():
    def __init__(self, src_path, number_path, feature_path, des_path, ratio: float, isTrain, FEATURES: int = 15):
        self.src_path = src_path
        self.number_path = number_path
        self.feature_path = feature_path
        self.des_path = des_path
        self.feature = FEATURES
        self.ratio = ratio
        self.isTrain = isTrain

        # 记录训练集中的正常数据数量和测试集中的数据数量
        self.train_count = 0
        self.test_count = 0

        self.data = None
        self.label = None

    """
        功能：读取src_path中的数据，数值化后，输出到number_path中
               如果是训练数据，就只将正常数据输出到number_path中；
               如果是测试数据，就将所有数据输出到number_path中
        运行一次即可
    """
    def get_data(self):
        data_file = open(self.number_path, 'w', newline='')
        with open(self.src_path, 'r') as data_source:
            csv_reader = csv.reader(data_source)
            csv_writer = csv.writer(data_file)

            for row in csv_reader:
                temp_line = np.array(row)
                temp_line[1] = handle_protocol(row[1])
                temp_line[2] = handle_service(row[2])
                temp_line[3] = handle_flag(row[3])
                temp_line[41] = handle_label(row[41])
                # 如果是训练数据，就只获取正常数据
                if self.isTrain:
                    if int(temp_line[41]) == 0:
                        self.train_count += 1
                        csv_writer.writerow(temp_line)
                # 否则将所有数据均输出到文件中。
                else:
                    self.test_count += 1
                    csv_writer.writerow(temp_line)
        data_file.close()

    """
        功能：从number_path中（41个特征）选择指定特征（15个），输出到feature_path中。
        如果不修改选择的特征，运行一次即可
    """
    def select_features(self):
        buffer_line = np.zeros(self.feature+1)
        data_file = open(self.feature_path, 'w', newline='')
        with open(self.number_path, 'r') as data_source:
            csv_reader = csv.reader(data_source)
            csv_writer = csv.writer(data_file)

            # 对每行数据进行特征选择
            for row in csv_reader:
                temp_line = np.array(row)
                buffer_line[0] = temp_line[1]
                buffer_line[1] = temp_line[2]
                buffer_line[2] = temp_line[3]
                buffer_line[3] = temp_line[4]
                buffer_line[4] = temp_line[5]
                buffer_line[5] = temp_line[11]
                buffer_line[6] = temp_line[22]
                buffer_line[7] = temp_line[23]
                buffer_line[8] = temp_line[28]
                buffer_line[9] = temp_line[30]
                buffer_line[10] = temp_line[31]
                buffer_line[11] = temp_line[32]
                buffer_line[12] = temp_line[33]
                buffer_line[13] = temp_line[35]
                buffer_line[14] = temp_line[36]
                buffer_line[15] = temp_line[41]
                # 将选择好的数据行输出到文件中
                csv_writer.writerow(buffer_line)

        data_file.close()

    """
        功能：从feature_path中获取已数值化且选择完特征（15个）的数据，实现按比例ratio加载、归一化和标准化
              可以获得归一化后的data数组和label数组；将数组合并输出到dst_path中
        输入：data_label（array）；ratio [0,1]
        每次都要运行
    """
    def normalizations(self):
        # 从文件中加载所有数据（15个特征+1个label）
        data_label = np.loadtxt(self.feature_path, delimiter=",", skiprows=0)
        # rows = 97278
        rows = data_label.shape[0]

        # 按比例随机选择数据
        number = int(rows * self.ratio)
        # array 97 * 16
        data_sample = sample(data_label, number)

        # 将样本数据分成数据data和标签label
        temp_data = np.hsplit(data_sample,(self.feature, self.feature+1))
        self.data = temp_data[0]
        self.label = temp_data[1]

        # 对数据部分进行标准化
        self.data = normalize(self.data)

        # 将采样的数据输出到文件中
        temp_data_label = np.array(np.hstack((self.data, self.label)))
        np.savetxt(self.des_path, temp_data_label, fmt='%.5f',delimiter=',')

# 中间代码
def handle_protocol(protocal):
    """ 数值化协议类型 """
    protocol_list = ['tcp', 'udp', 'icmp']
    return protocol_list.index(protocal)+1

def handle_service(service):
    """ 数值化70种网络服务类型 """
    service_list = ['aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain', 'domain_u',
                    'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest',
                    'hostnames',
                    'http', 'http_2784', 'http_443', 'http_8001', 'imap4', 'IRC', 'iso_tsap', 'klogin', 'kshell',
                    'ldap',
                    'link', 'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp',
                    'nntp',
                    'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje',
                    'shell',
                    'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time',
                    'urh_i', 'urp_i',
                    'uucp', 'uucp_path', 'vmnet', 'whois', 'X11', 'Z39_50',
                    'icmp']
    return service_list.index(service)+1

def handle_flag(flag):
    """ 数值化网络连接状态 """
    flag_list = ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH']
    return flag_list.index(flag)+1

def handle_label(label):
    """ 数值化标签 """
    label_list = ['normal.',
                  'buffer_overflow.', 'httptunnel.', 'loadmodule.', 'perl.', 'ps.', 'rootkit.', 'sqlattack.', 'xterm.',
                  'apache2.', 'back.', 'land.', 'mailbomb.', 'neptune.', 'pod.', 'processtable.', 'smurf.', 'teardrop.', 'udpstorm.',
                  'ftp_write.', 'guess_passwd.', 'imap.', 'multihop.', 'named.', 'phf.', 'sendmail.', 'snmpgetattack.', 'snmpguess.',
                  'spy.',  'warezmaster.', 'warezclient.', 'worm.', 'xlock.', 'xsnoop.',
                  'PROBE.', 'ipsweep.', 'mscan.', 'nmap.', 'portsweep.', 'saint.', 'satan.']
    return label_list.index(label)

def sample(array, number):
    # 将矩阵转化为list，随机采样，返回array
    rand_arr = np.arange(array.shape[0])
    np.random.shuffle(rand_arr)
    data_sample = array[rand_arr[0:number]]
    return data_sample

def normalize(x_data):
    # 选择所有特征进行最大最小化
    min_max_scaler = preprocessing.MinMaxScaler()
    x_data_min_max = min_max_scaler.fit_transform(x_data)
    # 97 * 15
    # print("\nafter min_max normalizing, the x_data is:", x_data_min_max)

    # 选择连续数据进行归一化(除了第1-3列)
    x_data_seq_T = x_data_min_max.T[3:-1]
    # 97 * 12
    x_data_seq_array = x_data_seq_T.T
    x_data_z_score = preprocessing.scale(x_data_seq_array)
    # print("\nafter z_score normalizing, the x_data is :", x_data_z_score)

    # 归一化和最大最小化的矩阵合并为一个array
    x_1to3_T = x_data_min_max.T[0:4]
    # 97 * 3
    x_1to3 = x_1to3_T.T
    # 在列上合并 97 * 15
    data = np.hstack((x_1to3, x_data_z_score))

    return data

