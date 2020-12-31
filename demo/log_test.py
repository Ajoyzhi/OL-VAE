
import logging
import time

def init_log(filepath):

    # 获取本地时间
    real_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    # 为你的log起个名字
    logger = logging.getLogger(__name__)
    # 设置输出等级，debug<info<warning等
    logger.setLevel(level=logging.DEBUG)

    # 将log输出到文件中
    handler = logging.FileHandler(filepath + real_time + ".log")
    # 设置等级，不设置默认用上面的输出等级
    # handler.setLevel(logging.INFO)
    # 设置格式 时间-输出内容
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    # 装载格式
    handler.setFormatter(formatter)
    # 将log输出到控制台
    console = logging.StreamHandler()
    # console.setLevel(logging.DEBUG)

    # 将整个两个添加到进logger中
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

if __name__ == '__main__':
    # log文件在E盘
    logger = init_log("E:/")
    logger.debug("aaaa")
    logger.info("sfda")


