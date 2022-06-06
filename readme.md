#  Calculate n-gram similarity with Pytorch 
This is a [torch](https://pytorch.org/ "torch") (version >= 1.4) tool that can simulate some text relevance metrics such as ROUGE, BLEU on GPU.  The similariity score is calculated with the product of system output and reference. Given a system output $x$
 and a reference $y$, the returned result is $ |x \cap y| / |x| + |y|$

#### Usage of torch_ngram
- We require the system output and reference represented as `Long Tensor`
- Input parameters of `get_score( system_tensor, reference_tensor, pad_id, n_gram)`: 
  * **system_tensor** (batch_size, sys_seq_len) , 
  * **reference_tensor**: (batch_size, reference_num ,ref_seq_len)  # supporting multiple references, 
  * **pad_id** (int, the pad value in batching), 
  * **n_gram** (int, usually set to 1,2,3,4). 

```python
# An example of using torch_ngram, '=>' means roberta-tokenizer
sys = "leave that up to us"  =>  [38457, 14, 62, 7, 201]
ref1 = "leave that up quiet to us"  =>  [38457, 14, 62, 5128, 7, 201]
ref2 = "leave that quiet down"  => [38457, 14, 5128, 159]
ref3 = "this is frightening"  => [9226, 16, 21111]

import torch
from torch_ngram import pad2max
from torch_ngram import get_score
sys_tensor = torch.tensor( [38457, 14, 62, 7, 201])
ref1_tensor = torch.tensor( [38457, 14, 62, 5128, 7, 201])
ref2_tensor = torch.tensor( [38457, 14, 5128, 159])
ref3_tensor = torch.tensor( [9226, 16, 21111])

ref_tensors = pad2max([ref1_tensor, ref2_tensor, ref3_tensor], pad_id=1)
ref_tensors: tensor([[38457,    14,    62,  5128,     7,   201],
          [38457,    14,  5128,   159,     1,     1],
          [ 9226,    16, 21111,     1,     1,     1]])
# batch_size = 1  if you already has the batch dimension, remove unsqueeze(0)
print(get_score(sys_tensor.unsqueeze(0), ref_tensors.unsqueeze(0), pad_id=1, n_gram=1))
1-gram sim: tensor([[0.9091, 0.4444, 0.0000]]), which means sys is most similar to ref1 based on 1-gram overlap
print(get_score(sys_tensor.unsqueeze(0), ref_tensors.unsqueeze(0), pad_id=1, n_gram=2))
2-gram sim: tensor([[0.6667, 0.2857, 0.0000]])
print(get_score(sys_tensor.unsqueeze(0), ref_tensors.unsqueeze(0), pad_id=1, n_gram=3))
3-gram sim: tensor([[0.2857, 0.0000, 0.0000]])

# test self-similarity
print(get_score(sys_tensor.unsqueeze(0), sys_tensor.unsqueeze(0).unsqueeze(0), pad_id=1, n_gram=1))
1-gram sim: tensor([[1.]])

# test batch != 1
sys_tensor_batch = sys_tensor.unsqueeze(0).repeat(2, 1)
tensor([[38457,    14,    62,     7,   201],
        [38457,    14,    62,     7,   201]])
ref_tensor_batch = ref_tensors.unsqueeze(0).repeat(2, 1, 1)
tensor([[[38457,    14,    62,  5128,     7,   201],
         [38457,    14,  5128,   159,     1,     1],
         [ 9226,    16, 21111,     1,     1,     1]],

        [[38457,    14,    62,  5128,     7,   201],
         [38457,    14,  5128,   159,     1,     1],
         [ 9226,    16, 21111,     1,     1,     1]]])

print(get_score(sys_tensor_batch, ref_tensor_batch, pad_id=1, n_gram=1))
batch_size =2,  1-gram sim :  tensor([[0.9091, 0.4444, 0.0000], [0.9091, 0.4444, 0.0000]])
batch_size =2,  2-gram sim :  tensor([[0.6667, 0.2857, 0.0000],  [0.6667, 0.2857, 0.0000]])
```
