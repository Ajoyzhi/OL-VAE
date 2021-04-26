import logging
import time


def init_log(filepath, name):
    # 获取本地时间
    real_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    logger = logging.getLogger(name)
    # 设置输出等级，debug<info<warning等
    logger.setLevel(logging.INFO)

    # 将log输出到文件中
    f_handler = logging.FileHandler(filepath + real_time + name + ".log")
    # 设置等级，不设置默认用上面的输出等级
    f_handler.setLevel(logging.INFO)
    # 设置格式 时间-输出内容
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    # 装载格式
    f_handler.setFormatter(formatter)
    # 添加
    logger.addHandler(f_handler)

    # 将log输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # 添加到logger中
    logger.addHandler(console)

    return logger