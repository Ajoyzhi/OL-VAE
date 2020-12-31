import numpy as np
import csv

from sklearn import preprocessing

class Util():
    def __init__(self, src_path, des_path, ratio: float, isTrain, FEATURES: int = 15):
        self.src_path = src_path
        self.des_path = des_path
        self.feature = FEATURES
        self.ratio = ratio
        self.isTrain = isTrain

        self.data = None
        self.label = None

    """
        功能：读取source_path中的数据，并输入到des_path中（实现数值化）
        其他：获取所有的训练集正常数据
    """
    def get_data(self):
        data_file = open(self.des_path, 'w', newline='')
        with open(self.src_path, 'r') as data_source:
            csv_reader = csv.reader(data_source)
            csv_writer = csv.writer(data_file)
            count = 0  # 行数

            for row in csv_reader:
                temp_line = np.array(row)
                temp_line[1] = handle_protocol(row[1])
                temp_line[2] = handle_service(row[2])
                temp_line[3] = handle_flag(row[3])
                temp_line[41] = handle_label(row[41])
                # 如果是训练数据，就只获取正常数据
                if self.isTrain:
                    if int(temp_line[41]) == 0:
                        count += 1
                        csv_writer.writerow(temp_line)
                # 否则将所有数据均输出到文件中。
                else:
                    csv_writer.writerow(temp_line)
        # print("the number of normal data: ", count)
        data_file.close()

    """
        功能：从des_path中获取已数值化的数据，实现按比例ratio加载、归一化和标准化
        输入：data_label（array）；ratio [0,1]
        return：归一化后的data数组和label数组
    """

    def normalizations(self):
        # 从文件中加载所有数据
        data_label = np.loadtxt(self.des_path, delimiter=",", skiprows=0)
        # print("data_label:", data_label)
        rows = data_label.shape[0]

        # 按比例选择数据
        number = int(rows * self.ratio)
        data_sample = sample(data_label, number)
        data_sample = np.array(data_sample)
        # 97 * 42
        # print("data_sample:", data_sample.shape)
        # print("\ndata_simple:", data_sample)

        # 特征选择(选择大部分不为0的数据) 15为选择的特征数量
        x_data, self.label = select_features(data_sample, self.feature)
        # 97 * 15
        # print("\nafter selecting the features, the x_data is :", x_data)

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
        self.data = np.hstack((x_1to3, x_data_z_score))

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
    """ 将矩阵随机采样，返回array """
    rand_arr = np.arange(array.shape[0])

    np.random.shuffle(rand_arr)
    data_sample = array[rand_arr[0:number]]

    return data_sample

def select_features(data_sample, features):
    rows = data_sample.shape[0]
    # print("rows:", rows)
    x_data = np.zeros((rows, features))
    label = np.zeros(rows)
    # 选择数组中的非零特征，共15个（col_index）
    for row in range(rows):
        x_data[row][0] = data_sample[row][1]
        x_data[row][1] = data_sample[row][2]
        x_data[row][2] = data_sample[row][3]
        x_data[row][3] = data_sample[row][4]
        x_data[row][4] = data_sample[row][5]
        x_data[row][5] = data_sample[row][11]
        x_data[row][6] = data_sample[row][22]
        x_data[row][7] = data_sample[row][23]
        x_data[row][8] = data_sample[row][28]
        x_data[row][9] = data_sample[row][30]
        x_data[row][10] = data_sample[row][31]
        x_data[row][11] = data_sample[row][32]
        x_data[row][12] = data_sample[row][33]
        x_data[row][13] = data_sample[row][35]
        x_data[row][14] = data_sample[row][36]
        label[row] = data_sample[row][41]

    # print("function_x_data:", x_data)
    # print("function_label:", label)
    return x_data,label


