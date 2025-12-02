import random
from typing import List, Dict
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter
import torch
import torch.nn.functional as F

stoi:dict = {
    '0': 1,
    '1': 2,
    '2': 3,
    '3': 4,
    '4': 5,
    '5': 6,
    '6': 7,
    '7': 8,
    '8': 9,
    '9': 10,
    '_': 11,
    'x': 12,
    '.': 13,
}
number_tokens = range(10)
pad_token = 0
empty_token = 11
mask_token = 13

itos: dict = {v: k for k,v in stoi.items()}

def split_k(s, k):
    return "\n".join(s[i:i+k] for i in range(0, len(s), k))

def decode_tensor(t: torch.Tensor):
    bytes_flat = (t + ord('0') - 1).to(torch.uint8).reshape(-1)
    s = bytes(bytes_flat.tolist()).decode()
    s = s.replace(':', '_')
    s = s.replace(';', 'x')
    s = s.replace('<', '.')
    return s

def decode(t: torch.Tensor, width: int):
    return split_k(decode_tensor(t), width)

def replace_character(s: str):
    s = s.replace('_', ':')
    s = s.replace('x', ';')
    s = s.replace('.', '<')
    return s

def encode_to_tensor(s: str, height: int, width: int):
    s = replace_character(s)
    t = torch.tensor(list(s.encode()), dtype=torch.uint8)
    t = t.view(height, width) - ord('0') + 1
    return t

def encode_to_numpy(s: str, height: int, width: int):
    s = replace_character(s)
    arr = np.frombuffer(s.encode(), dtype=np.uint8)
    arr = arr.reshape(height, width) - ord('0') + 1
    return arr.astype(np.uint8)

def render_vertical_multiplication_with_operation(a: int, b: int) -> tuple[list[str], int, int]:
    sa = str(a)
    sb = str(b)
    partials = [str(a * int(d)) for d in reversed(sb)]
    final = str(a * b)

    used_rows = 2 + len(partials) + 1
    used_cols = max(len(final), len(sb) + 2)
    
    results = []
    padded_sa = '_' * (used_cols - len(sa)) + sa
    results.append(padded_sa)
    padded_sb = '_' * (used_cols - len(sb) - 2) + 'x_' +sb
    results.append(padded_sb)
    for i, partial in enumerate(partials):
        padded_partial = partial + i * '_'
        padded_partial = '_' * (used_cols - len(padded_partial)) + padded_partial
        results.append(padded_partial)
    padded_final = '_' * (used_cols - len(final)) + final
    results.append(padded_final)
    result = ''.join(results)
    return result, used_rows, used_cols

def list_to_tensor(lst: list[str]) -> torch.Tensor:
    """
    lst: List[str], 每个str长度相同，只包含stoi中的字符
    return: LongTensor, 形状 (len(lst), str_len)
    """

    data = [[stoi[c] for c in s] for s in lst]

    return torch.tensor(data, dtype=torch.int)

def prepare_data_masked(a:int , b:int, height:int, width:int, random: random.Random = random.Random(0)):
    num_mask_rows = 1
    
    data_str, n_row, n_col = render_vertical_multiplication_with_operation(a, b)
    assert n_col <= width and n_row <= height
    raw_data_tensor = encode_to_tensor(data_str, n_row, n_col)
    pad_rows = height - n_row
    pad_cols = width - n_col
    
    front_pad_cols = random.randint(0, pad_cols)
    front_pad_rows = random.randint(0, pad_rows)

    back_pad_cols = pad_cols - front_pad_cols
    back_pad_rows = pad_rows - front_pad_rows
    raw_data_tensor = F.pad(raw_data_tensor, (front_pad_cols, back_pad_cols, front_pad_rows, back_pad_rows), value=empty_token)
    raw_data_tensor = raw_data_tensor.reshape(-1)

    answers = []
    inputs = []
    
    for i in range(front_pad_rows + 2, front_pad_rows + n_row, num_mask_rows):
        mask_start_rows = i
        mask_end_rows = min(height, mask_start_rows + num_mask_rows)
        answer_tensor = raw_data_tensor.clone()
        answer_tensor[mask_end_rows*width:] = empty_token
        answers.append(answer_tensor)
        input_tensor = answer_tensor.clone()
        input_tensor[mask_start_rows*width:mask_end_rows*width] = mask_token
        inputs.append(input_tensor)

    return inputs, answers

class HeatmapDataset(Dataset):
    def __init__(self, 
                 length: int = 10000, 
                 seed: int = None, 
                 width: int = 60, 
                 height: int = 60, 
                 min_num_len_1: int = 1,
                 max_num_len_1: int = 10,
                 num_len_2: int = 1,
                 with_add: bool = False,
                 for_test: bool = False,
                 reverse: bool = False):
        self.length = length
        self.width = width
        self.height = height
        self.num1_len_range = range(min_num_len_1, max_num_len_1 + 1)
        self.num2_len = num_len_2
        self.random = random.Random(seed)
        self.with_add = with_add
        self.for_test = for_test
        self.reverse = reverse
        self.count = 0

    def __len__(self):
        return self.length
    
    def _sample_digit_positive(self, num_digits) -> tuple[int, int]:
        low = 10 ** (num_digits - 1)
        high = 10 ** num_digits - 1
        return self.random.randint(low, high), num_digits

    def __getitem__(self, idx: int = 0) -> list[Dict[str, torch.Tensor]]:
        num1_len = self.num1_len_range[self.count % len(self.num1_len_range)]
        a, a_len = self._sample_digit_positive(num1_len)
        b, b_len = self._sample_digit_positive(self.num2_len)
        inputs, answers = prepare_data_masked(a = a, 
                                      b = b, 
                                      height = self.height, 
                                      width=self.width,
                                      random=self.random)

        inputs_tensor = torch.stack(inputs, dim=0)
        answers_tensor = torch.stack(answers, dim=0)

        result = {
            "input": inputs_tensor,
            "label": answers_tensor,
            "num1_len": a_len,
            "num2_len": b_len
        }
        self.count += 1
        return result
    



def get_batch(dataloader_iter: _BaseDataLoaderIter):
    raw_data = next(dataloader_iter)
    raw_data["input"] = raw_data["input"].transpose(0, 1)
    raw_data["label"] = raw_data["label"].transpose(0, 1)
    
    batch_datas = []
    for i in range(raw_data["input"].size(0)):
        batch_datas.append(
            {
                "inputs": raw_data["input"][i],
                "labels": raw_data["label"][i],
                "puzzle_identifiers": torch.zeros(raw_data["input"].size(1), dtype=torch.int32)
            }
        )
    return batch_datas


def dihedral_transform_torch(arr: torch.Tensor, tid: int) -> torch.Tensor:
    """8 dihedral symmetries by rotate, flip and mirror (torch version)"""

    if tid == 0:
        return arr  # identity
    elif tid == 1:
        return torch.rot90(arr, k=1, dims=(0, 1))
    elif tid == 2:
        return torch.rot90(arr, k=2, dims=(0, 1))
    elif tid == 3:
        return torch.rot90(arr, k=3, dims=(0, 1))
    elif tid == 4:
        return torch.flip(arr, dims=[1])   # fliplr: flip width
    elif tid == 5:
        return torch.flip(arr, dims=[0])   # flipud: flip height
    elif tid == 6:
        return arr.transpose(0, 1)         # main diagonal reflection
    elif tid == 7:
        return torch.flip(torch.rot90(arr, k=1, dims=(0, 1)), dims=[1])
    else:
        return arr


def prepare_data_masked_multiviews(a:int , b:int, height:int, width:int, random: random.Random = random.Random(0), num_views = 16):
    num_mask_rows = 1
    
    data_str, n_row, n_col = render_vertical_multiplication_with_operation(a, b)
    assert n_col <= width and n_row <= height
    raw_data_tensor = encode_to_tensor(data_str, n_row, n_col)
    pad_rows = height - n_row
    pad_cols = width - n_col
    input_tensors = []
    answer_tensors = []
    
    for i in range(num_views):
        answers = []
        inputs = []
        front_pad_cols = random.randint(0, pad_cols)
        front_pad_rows = random.randint(0, pad_rows)
        back_pad_cols = pad_cols - front_pad_cols
        back_pad_rows = pad_rows - front_pad_rows
        padded_data_tensor = F.pad(raw_data_tensor, (front_pad_cols, back_pad_cols, front_pad_rows, back_pad_rows), value=empty_token)
        idx = random.randint(0, 7)
        padded_data_tensor = dihedral_transform_torch(padded_data_tensor, idx)

        padded_data_tensor = padded_data_tensor.reshape(-1)
        for i in range(front_pad_rows + 2, front_pad_rows + n_row, num_mask_rows):
            mask_start_rows = i
            mask_end_rows = min(height, mask_start_rows + num_mask_rows)
            answer_tensor = padded_data_tensor.clone()
            answer_tensor[mask_end_rows*width:] = empty_token
            answers.append(answer_tensor)
            input_tensor = answer_tensor.clone()
            input_tensor[mask_start_rows*width:mask_end_rows*width] = mask_token
            inputs.append(input_tensor)
        input_tensor = torch.stack(inputs, dim=0)
        answer_tensor = torch.stack(answers, dim=0)
        input_tensors.append(input_tensor)
        answer_tensors.append(answer_tensor)
    return input_tensors, answer_tensors


class HeatmapDatasetMultiViews(Dataset):
    def __init__(self, 
                 length: int = 10000, 
                 seed: int = None, 
                 width: int = 60, 
                 height: int = 60, 
                 min_num_len_1: int = 1,
                 max_num_len_1: int = 10,
                 num_len_2: int = 1,
                 num_views: int = 16,
                 with_add: bool = False,
                 for_test: bool = False,
                 reverse: bool = False):
        self.length = length
        self.width = width
        self.height = height
        self.num1_len_range = range(min_num_len_1, max_num_len_1 + 1)
        self.num2_len = num_len_2
        self.random = random.Random(seed)
        self.with_add = with_add
        self.for_test = for_test
        self.reverse = reverse
        self.count = 0
        self.num_views = num_views

    def __len__(self):
        return self.length
    
    def _sample_digit_positive(self, num_digits) -> int:
        low = 10 ** (num_digits - 1)
        high = 10 ** num_digits - 1
        return self.random.randint(low, high)

    def __getitem__(self, idx: int = 0) -> list[Dict[str, torch.Tensor]]:
        num1_len = self.num1_len_range[self.count % len(self.num1_len_range)]
        a = self._sample_digit_positive(num1_len)
        b = self._sample_digit_positive(self.num2_len)
        inputs, answers = prepare_data_masked_multiviews(a = a, 
                                      b = b, 
                                      height = self.height, 
                                      width=self.width,
                                      random=self.random,
                                      num_views=self.num_views)

        inputs_tensor = torch.stack(inputs, dim=0)
        answers_tensor = torch.stack(answers, dim=0)

        result = {
            "input": inputs_tensor,
            "label": answers_tensor,
        }
        self.count += 1
        return result

def get_batch_multiviews(dataloader_iter: _BaseDataLoaderIter):
    raw_data = next(dataloader_iter)
    print(raw_data["input"].shape)
    print(raw_data["label"].shape)
    
    
    raw_data["input"] = raw_data["input"].permute(2, 1, 0, 3)
    raw_data["label"] = raw_data["label"].permute(2, 1, 0, 3)
    
    batch_datas = []
    for i in range(raw_data["input"].size(0)):
        batch_datas.append(
            {
                "inputs": raw_data["input"][i],
                "labels": raw_data["label"][i],
            }
        )
    return batch_datas


dataset = HeatmapDatasetMultiViews(
    length=1000,
    seed=36,
    height=20,
    width=20,
    min_num_len_1=5,
    max_num_len_1=10,
    num_len_2=5,
)
dataloader = DataLoader(
    dataset=dataset,
    batch_size=10,
    shuffle=False,
    num_workers=1,
)
dataloader_iter = iter(dataloader)
results = get_batch_multiviews(dataloader_iter)
for result in results:
    for k, v in result.items():
        # print(f"\n{k}: \n{decode(v, 20)}")
        print(f"\n{k}: {v.shape}")
    print()