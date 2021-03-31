import torch

"""
# compute the median of data in list composing with tensor
def list_median(list):
    sum_tensor = []
    for item in list:
        tmp = torch.sum(item)
        sum_tensor.append((tmp, item))

    sort_tensor1 = sorted(sum_tensor, key=lambda x:x[0])
    print("sort_list:", sort_tensor1)
    half = len(sort_tensor1) // 2
    print("half:", half)
    print("list[half]：", sort_tensor1[half][1])
    print("list[~half]：", sort_tensor1[~half][1])
    return (sort_tensor1[half][1] + sort_tensor1[~half][1]) / 2
"""
# compute the min of data in list composing with tensor
def list_min(list):
    sum_tensor = []
    for item in list:
        tmp = torch.min(item)
        print("the min in the tensor:", tmp) # 0.2,0.3,0.4,0.1,2.0
        sum_tensor.append((tmp, item))

    sort_tensor1 = sorted(sum_tensor, key=lambda x: x[0]) #
    print("sort_tensor1:", sort_tensor1)
    return sort_tensor1[0][1]

list = []
tensor1 = torch.Tensor([0.2, 2.0, 6.0])
tensor2 = torch.Tensor([0.3, 2.0, 5.0])
tensor3 = torch.Tensor([0.4, 2.0, 1.0])
tensor4 = torch.Tensor([0.4, 2.0, 2.0])
tensor5 = torch.Tensor([0.1, 2.0, 3.0])
tensor6 = torch.Tensor([10, 2.0, 4.0])

list.append(tensor1)
list.append(tensor2)
list.append(tensor3)
list.append(tensor4)
list.append(tensor5)
list.append(tensor6)

#tensorx = list_median(list)
tensorx = list_min(list)
print("tensorx:", tensorx)
