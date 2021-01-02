a = [1,2,3,4,5,6]
# 应该是按照key-value对应得到的
for i,j in enumerate(a):
    print(i,j)

print(a.index(5))

# 得到协议的对应数值
def handle_protocol(protocal):
    # 数值化协议类型 
    protocol_list = ['tcp', 'udp', 'icmp']
    return protocol_list.index(protocal)

print(handle_protocol('tcp'))

# 测试%d占位符
a = 1.0
print("The data is %s."%a)

