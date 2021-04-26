import numpy as np
import csv

class Util():
    def __init__(self, src_path, number_path, feature_path, des_path, ratio: float, isTrain: bool, FEATURES: int = 9):
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
               如果是测试数据，就将正常数据和dos攻击数据输出到number_path中
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
                # 否则将正常数据和DoS攻击数据均输出到文件中。
                else:
                    if int(temp_line[41]) in {0, 10, 13, 14, 16, 17}:
                        self.test_count += 1
                        csv_writer.writerow(temp_line)
        data_file.close()

    """
        功能：从number_path中（41个特征）选择指定特征（9个），输出到feature_path中。
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
                buffer_line[0] = temp_line[1]   # protocol_type -- 离散型
                buffer_line[1] = temp_line[4]   # src_bytes
                buffer_line[2] = temp_line[22]  # count
                buffer_line[3] = temp_line[23]  # srv_count
                buffer_line[4] = temp_line[30]  # srv_diff_host_rate
                buffer_line[5] = temp_line[31]  # dst_host_count
                buffer_line[6] = temp_line[32]  # dst_host_srv_count
                buffer_line[7] = temp_line[36]  # dst_host_src_diff_host_rate
                buffer_line[8] = temp_line[0]   # duration
                buffer_line[9] = temp_line[41]  # label
                # 将选择好的数据行输出到文件中
                csv_writer.writerow(buffer_line)
        data_file.close()

    """
        功能：从feature_path中获取已数值化且选择完特征（9个）的数据，实现按比例ratio加载、标准化和归一化
              可以获得归一化后的data数组和label数组；将数组合并输出到dst_path中
            标准化：
                连续型数据：l2 norm
                离散型数据：不处理
            归一化：
                连续数据：最大最小归一化
                离散数据：最大最小归一化
        输入：data_label（array）；ratio [0,1]
        每次都要运行
    """
    def normalizations(self):
        # 从文件中加载所有数据（9个特征+1个label）
        data_label = np.loadtxt(self.feature_path, dtype=float, delimiter=",", skiprows=0)
        # rows = 97278
        rows = data_label.shape[0]
        # 按比例随机选择数据
        number = int(rows * self.ratio)
        # array 97 * 10
        data_sample = sample(data_label, number)
        # 将样本数据分成数据data和标签label
        temp_data = np.hsplit(data_sample,(self.feature, self.feature+1))
        self.data = temp_data[0]
        self.label = temp_data[1]

        # 对数据部分进行标准化
        self.data = normalize(self.data)
        # 对数据部分进行归一化
        self.data = min_max(self.data)
        # 将采样的数据输出到文件中
        temp_data_label = np.array(np.hstack((self.data, self.label)))
        np.savetxt(self.des_path, temp_data_label,fmt='%.5f', delimiter=',')

        # 统计数据中的正常和异常数据
        normal = 0
        abnormal = 0
        for item in self.label:
            if item == 0:
                normal += 1
            else:
                abnormal += 1

        if not self.isTrain:
            print("normal data number:{},abnormal data number:{} in test set.".format(normal, abnormal))

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
    """
    数值化标签DoS:
        apache2 -- 9
        back -- 10 v
        land -- 11
        mailbomb -- 12
        neptune -- 13 v
        pod -- 14 v
        precesstable --15
        smurf -- 16 v
        teardrop -- 17 v
        udpstorm -- 18
    """
    label_list = ['normal.','buffer_overflow.', 'httptunnel.', 'loadmodule.', 'perl.', 'ps.', 'rootkit.', 'sqlattack.', 'xterm.',
                  'apache2.', 'back.', 'land.', 'mailbomb.', 'neptune.', 'pod.', 'processtable.', 'smurf.', 'teardrop.', 'udpstorm.',
                  'ftp_write.', 'guess_passwd.', 'imap.', 'multihop.', 'named.', 'phf.', 'sendmail.', 'snmpgetattack.', 'snmpguess.',
                  'spy.',  'warezmaster.', 'warezclient.', 'worm.', 'xlock.', 'xsnoop.',
                  'PROBE.', 'ipsweep.', 'mscan.', 'nmap.', 'portsweep.', 'saint.', 'satan.']
    return label_list.index(label)

def sample(array, number):
    rand_arr = np.arange(array.shape[0])
    # 打乱矩阵顺序
    np.random.shuffle(rand_arr)
    data_sample = array[rand_arr[0:number]]
    return data_sample

def z_score_normalization(x):
    x = (x - np.mean(x)) / np.std(x)
    return x

def min_max_normalization(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x

def normalize(x_data):
    """标准化数据 -- 只考虑连续数据，第一个特征proto为离散型"""
    for i in range(1, x_data.shape[1]):
        x_data[:, i] = z_score_normalization(x_data[:, i])
    return x_data

def min_max(x_data):
    """最大最小化数据"""
    for i in range(x_data.shape[1]):
        x_data[:, i] = min_max_normalization(x_data[:, i])
    return x_data