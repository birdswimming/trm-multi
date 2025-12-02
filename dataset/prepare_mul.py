from os import makedirs
import random
from typing import List, Dict
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from tqdm import tqdm

stoi:dict = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    '*': 10,
    '_': 11,
    '#': 12,
    '.': 15,
}
number_tokens = range(10)
empty_token = 11
new_line_token = 12
start_token = 13
end_token = 14
mask_token = 15
ignore_token = -100

itos: dict = {v: k for k,v in stoi.items()}
itos[start_token] = 'schratch_pad:\n'
itos[end_token] = '\n'
itos[ignore_token] = ''
itos[new_line_token] = '#\n' 
# decode = lambda l: ''.join([itos[i] for i in l])
def decode(l: list[int]):
    valid_l = []
    for i in l:
        if i != end_token:
            valid_l.append(i)
        else:
            break
    return ''.join([itos[i] for i in valid_l])

def decode_to_result(l: list[int]):
    valid_l = []
    for i in l:
        if i != end_token:
            valid_l.append(i)
        else:
            break
    last_line = []
    cur_line = []
    for i in valid_l:
        if i in number_tokens:
            cur_line.append(i)
        elif i == new_line_token:
            if len(cur_line) > 0:
                last_line = cur_line
            cur_line = []
    if len(last_line) == 0:
        return -1
    final_result = ''.join([itos[i] for i in reversed(last_line)])
    final_result = int(final_result)
    return final_result

def render_vertical_multiplication(a: int, b: int) -> tuple[list[str], int, int]:
    """
    将 a * b 以竖式乘法格式渲染为 width x height 的字符块。
    - 不含分割线；
    - 空格全部用 '_'；
    - 每行末尾额外加 '#'（不在画布内）。
    """
    sa = str(a)
    sb = str(b)
    # 从低位到高位生成部分积
    partials = [str(a * int(d)) for d in reversed(sb)]
    final = str(a * b)

    # 需要的总行数（不含空行）：被乘数 + 乘数 + 每个部分积 + 最终积
    used_rows = 2 + len(partials) + 1
    used_cols = len(final)
    
    results = []
    padded_sa = '_' * (used_cols - len(sa)) + sa
    results.append(padded_sa)
    padded_sb = '_' * (used_cols - len(sb)) + sb
    results.append(padded_sb)
    for i, partial in enumerate(partials):
        padded_partial = partial + i * '_'
        padded_partial = '_' * (used_cols - len(padded_partial)) + padded_partial
        results.append(padded_partial)
    padded_final = '_' * (used_cols - len(final)) + final
    results.append(padded_final)
    return results, used_rows, used_cols

def render_vertical_multiplication_with_signal(a: int, b: int) -> tuple[list[str], int, int]:
    """
    将 a * b 以竖式乘法格式渲染为 width x height 的字符块。
    - 不含分割线；
    - 空格全部用 '_'；
    - 每行末尾额外加 '#'（不在画布内）。
    """
    sa = str(a)
    sb = str(b)
    # 从低位到高位生成部分积
    partials = [str(a * int(d)) for d in reversed(sb)]
    final = str(a * b)

    # 需要的总行数（不含空行）：被乘数 + 乘数 + 每个部分积 + 最终积
    used_rows = 2 + len(partials) + 1
    used_cols = max(len(final), len(sb) + 2)
    
    results = []
    padded_sa = '_' * (used_cols - len(sa)) + sa
    results.append(padded_sa)
    padded_sb = '_' * (used_cols - len(sb) - 2) + '*_' +sb
    results.append(padded_sb)
    for i, partial in enumerate(partials):
        padded_partial = partial + i * '_'
        padded_partial = '_' * (used_cols - len(padded_partial)) + padded_partial
        results.append(padded_partial)
    padded_final = '_' * (used_cols - len(final)) + final
    results.append(padded_final)
    return results, used_rows, used_cols


def render_vertical_multiplication_with_addition(a: int, b: int) -> tuple[list[str], int, int]:
    """
    将 a * b 以竖式乘法格式渲染为 width x height 的字符块。
    - 不含分割线；
    - 空格全部用 '_'；
    - 每行末尾额外加 '#'（不在画布内）。
    """
    sa = str(a)
    sb = str(b)
    # 从低位到高位生成部分积
    mul_partials_nums = [a * int(d) for d in reversed(sb)]
    mul_partials = [str(p) for p in mul_partials_nums]
    add_partials_nums = []
    sum = 0
    for i, p in enumerate(mul_partials_nums):
        sum += p * (10**i)
        add_partials_nums.append(sum)
    final = str(a * b)
    add_partials = [str(p) for p in add_partials_nums]

    # 需要的总行数（不含空行）：被乘数 + 乘数 + 每个部分积
    used_rows = 2 + len(mul_partials) + len(add_partials)
    used_cols = len(final)
    
    results = []
    padded_sa = '_' * (used_cols - len(sa)) + sa
    results.append(padded_sa)
    padded_sb = '_' * (used_cols - len(sb)) + sb
    results.append(padded_sb)
    for i, partial in enumerate(mul_partials):
        padded_partial = partial + i * '_'
        padded_partial = '_' * (used_cols - len(padded_partial)) + padded_partial
        results.append(padded_partial)
    for partial in add_partials:
        padded_partial = '_' * (used_cols - len(partial)) + partial
        results.append(padded_partial)
    # results.append(padded_final)
    return results, used_rows, used_cols

# render_vertical_multiplication(1234, 5432)


def list_to_tensor(lst: list[str]) -> torch.Tensor:
    """
    lst: List[str], 每个str长度相同，只包含stoi中的字符
    return: LongTensor, 形状 (len(lst), str_len)
    """
    # 每个字符映射为stoi中的整数
    data = [[stoi[c] for c in s] for s in lst]
    # 转为tensor
    return torch.tensor(data, dtype=torch.int)

def prepare_data(a:int , b:int, height:int, width:int, random: random.Random = random.Random(0), with_add: bool = False, for_test:bool=False, reverse: bool = False):
    data_str_list, n_row, n_col = render_vertical_multiplication_with_signal(a, b)
    assert n_col <= width and n_row <= height
    data_tensor = list_to_tensor(data_str_list)
    pad_rows = height - n_row
    pad_cols = width - n_col
    
    front_pad_cols = random.randint(0, pad_cols)
    front_pad_rows = random.randint(0, pad_rows)

    back_pad_cols = pad_cols - front_pad_cols
    back_pad_rows = pad_rows - front_pad_rows
    
    
    data_tensor = F.pad(data_tensor, (front_pad_cols, back_pad_cols, front_pad_rows, back_pad_rows), value=11)
    data_tensor = data_tensor.reshape(-1)
    answer_tensor = data_tensor
    answer = decode(answer_tensor.tolist())
    input_tensor = data_tensor
    input_tensor[(2+front_pad_rows)*(width):] = empty_token
    input = decode(input_tensor.tolist())
    return input, answer

def prepare_data_masked(a:int , b:int, height:int, width:int, random: random.Random = random.Random(0)):
    num_mask_rows = 1
    
    data_str_list, n_row, n_col = render_vertical_multiplication_with_signal(a, b)
    assert n_col <= width and n_row <= height
    raw_data_tensor = list_to_tensor(data_str_list)
    pad_rows = height - n_row
    pad_cols = width - n_col
    
    front_pad_cols = random.randint(0, pad_cols)
    front_pad_rows = random.randint(0, pad_rows)

    back_pad_cols = pad_cols - front_pad_cols
    back_pad_rows = pad_rows - front_pad_rows
    raw_data_tensor = F.pad(raw_data_tensor, (front_pad_cols, back_pad_cols, front_pad_rows, back_pad_rows), value=empty_token)
    raw_data_tensor = raw_data_tensor.reshape(-1)
    
    # mask_rows = random.randint(front_pad_rows + 2, height - num_mask_rows + 1)
    answers = []
    inputs = []
    
    for i in range(front_pad_rows + 2, front_pad_rows + n_row, num_mask_rows):
        mask_start_rows = i
        mask_end_rows = min(height, mask_start_rows + num_mask_rows)
        data_tensor = raw_data_tensor.clone()
        answer_tensor = data_tensor
        answer_tensor[mask_end_rows*width:] = empty_token
        answer = decode(answer_tensor.tolist())
        answers.append(answer)
        input_tensor = data_tensor
        input_tensor[mask_start_rows*width:mask_end_rows*width] = mask_token
        input = decode(input_tensor.tolist())
        inputs.append(input)

    return inputs, answers



class VerticalMultiplicationDataset(Dataset):
    """
    每个样本返回一个 60×60 的字符块（字符串），包含一个 10位×10位 的竖式乘法。
    格式：
      - 无分割线；
      - 空格→'_'；
      - 每行结尾加 '#'
    """
    def __init__(self, 
                 length: int = 10000, 
                 seed: int = None, 
                 width: int = 60, 
                 height: int = 60, 
                 min_num_len_1: int = 1,
                 max_num_len_1: int = 10,
                 min_num_len_2: int = 1,
                 max_num_len_2: int = 10,
                 with_add: bool = False,
                 for_test: bool = False,
                 reverse: bool = False):
        self.length = length
        self.width = width
        self.height = height
        self.max_num_len_1 = max_num_len_1
        self.min_num_len_1 = min_num_len_1
        self.max_num_len_2 = max_num_len_2
        self.min_num_len_2 = min_num_len_2
        self.random = random.Random(seed)
        self.with_add = with_add
        self.for_test = for_test
        self.reverse = reverse

    def __len__(self):
        return self.length
    
    def _sample_digit_positive(self, min_num_len, max_num_len) -> int:
        # 均匀选择一个位数
        num_digits = self.random.randint(min_num_len, max_num_len)
        # 生成该位数的正整数
        low = 10 ** (num_digits - 1)
        high = 10 ** num_digits - 1
        return self.random.randint(low, high)

    def __getitem__(self, idx: int = 0) -> Dict[str, torch.Tensor]:
        a = self._sample_digit_positive(self.min_num_len_1, self.max_num_len_1)
        b = self._sample_digit_positive(self.min_num_len_2, self.max_num_len_2)
        inputs, answers = prepare_data_masked(a = a, 
                                      b = b, 
                                      height = self.height, 
                                      width=self.width,
                                      random=self.random)
        result = {
            "input": inputs,
            "answer": answers,
        }
        return result


dataset_train = VerticalMultiplicationDataset(
    length=1000000,
    seed=36, 
    width=20, 
    height=20, 
    min_num_len_1=1,
    max_num_len_1=7,
    min_num_len_2=1,
    max_num_len_2=7, 
    with_add=False,
    for_test=False,
    reverse=False
)

dataset_test = VerticalMultiplicationDataset(
    length=1000000,
    seed=36, 
    width=20, 
    height=20, 
    min_num_len_1=8,
    max_num_len_1=10,
    min_num_len_2=8,
    max_num_len_2=10, 
    with_add=False,
    for_test=False,
    reverse=False
)



def split_k(s, k):
    return "\n".join(s[i:i+k] for i in range(0, len(s), k))
result = dataset_train.__getitem__()

for input, answer in zip(result["input"], result["answer"]):
    print("input")
    print(split_k(input, 20))
    print("answer")
    print(split_k(answer, 20))
# print("input")
# print(split_k(result["input"], 20))
# print("answer")
# print(split_k(result["answer"], 20))

# data_path = "multi_data/20×20_masked"
# import os
# os.makedirs(data_path, exist_ok=True)


# train_data_len = 50000
# with open(f"{data_path}/train.csv", "w") as f:
#     print("source,question,answer,rating", file=f)
#     for i in tqdm(range(train_data_len)):
#         result = dataset.__getitem__()
#         input=result["input"]
#         answer=result["answer"]
#         print(f",{input},{answer},", file=f)

# test_data_len = 10000
# with open(f"{data_path}/test.csv", "w") as f:
#     print("source,question,answer,rating", file=f)
#     for i in tqdm(range(test_data_len)):
#         result = dataset.__getitem__()
#         input=result["input"]
#         answer=result["answer"]
#         print(f",{input},{answer},", file=f)

import os
data_path = "multi_data/20×20-masked-less-data"
os.makedirs(data_path, exist_ok=True)

train_data_len = 3000
with open(f"{data_path}/train.csv", "w") as f:
    print("source,question,answer,rating", file=f)
    for i in tqdm(range(train_data_len)):
        result = dataset_train.__getitem__()
        for input, answer in zip(result["input"], result["answer"]):
            print(f",{input},{answer},", file=f)

test_data_len = 1000
with open(f"{data_path}/test.csv", "w") as f:
    print("source,question,answer,rating", file=f)
    for i in tqdm(range(test_data_len)):
        result = dataset_test.__getitem__()
        for input, answer in zip(result["input"], result["answer"]):
            print(f",{input},{answer},", file=f)