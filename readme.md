#  Calculate n-gram similarity with Pytorch 
This is a [torch](https://pytorch.org/ "torch") (version >= 1.4) tool that can simulate some text relevance metrics such as ROUGE, BLEU on GPU.  The similariity score is calculated with the product of system output and reference. Given a system output $x$
 and a reference $y$, the returned result is:   $ 2*|x \cap y| / (|x| + |y|)$

### Install
``` 
pip install torch-ngram
```

### Usage of torch_ngram
- Input parameters of `get_score(reference_tensor, system_tensor, pad_id, n_gram)`: 
  * **reference_tensor**: LongTensor (batch_size, seq_len1), 
  * **system_tensor**: LongTensor (batch_size, hypo_num, ref_seq_len)  # supporting multiple hypotheses, 
  * **pad_id**: int # the pad value, 
  * **n_gram**: int # usually set to 1,2,3,4. 

```python
# An example of using torch_ngram, '->' means roberta-tokenizer
ref = "leave that up to us"  ->  [38457, 14, 62, 7, 201]
sys1 = "leave that up quiet to us"  ->  [38457, 14, 62, 5128, 7, 201]
sys2 = "leave that quiet down"  -> [38457, 14, 5128, 159]
sys3 = "this is frightening"  -> [9226, 16, 21111]

# Code
import torch
from torch_ngram.torch_ngram import pad2max
from torch_ngram.torch_ngram import get_score
ref_tensor = torch.tensor([38457, 14, 62, 7, 201])
sys1_tensor = torch.tensor([38457, 14, 62, 5128, 7, 201])
sys2_tensor = torch.tensor([38457, 14, 5128, 159])
sys3_tensor = torch.tensor([9226, 16, 21111])

sys_tensors = pad2max([sys1_tensor, sys2_tensor, sys3_tensor], pad_id=1)
==> sys_tensors: tensor([[38457,    14,    62,  5128,     7,   201],
                       [38457,    14,  5128,   159,     1,     1],
                       [9226,    16, 21111,     1,     1,     1]])
# batch_size = 1  if you already has the batch dimension, remove unsqueeze(0)
print(get_score(ref_tensor.unsqueeze(0), sys_tensors.unsqueeze(0), pad_id=1, n_gram=1))
==> 1-gram overlap: tensor([[0.9091, 0.4444, 0.0000]]), which means sys1 is most similar to ref based on 1-gram overlap
print(get_score(ref_tensor.unsqueeze(0), sys_tensors.unsqueeze(0), pad_id=1, n_gram=2))
==> 2-gram overlap: tensor([[0.6667, 0.2857, 0.0000]])
print(get_score(ref_tensor.unsqueeze(0), sys_tensors.unsqueeze(0), pad_id=1, n_gram=3))
==> 3-gram overlap: tensor([[0.2857, 0.0000, 0.0000]])

# test self-similarity
print(get_score(ref_tensor.unsqueeze(0), ref_tensor.unsqueeze(0).unsqueeze(0), pad_id=1, n_gram=1))
==> 1-gram overlap: tensor([[1.]])

# test batch != 1
ref_tensor_batch = ref_tensor.unsqueeze(0).repeat(2, 1)
==> tensor([[38457,    14,    62,     7,   201],
        [38457,    14,    62,     7,   201]])
sys_tensor_batch = sys_tensors.unsqueeze(0).repeat(2, 1, 1)
==> tensor([[[38457,    14,    62,  5128,     7,   201],
         [38457,    14,  5128,   159,     1,     1],
         [ 9226,    16, 21111,     1,     1,     1]],

        [[38457,    14,    62,  5128,     7,   201],
         [38457,    14,  5128,   159,     1,     1],
         [ 9226,    16, 21111,     1,     1,     1]]])

print(get_score(ref_tensor_batch, sys_tensor_batch, pad_id=1, n_gram=1))
==> batch_size =2,  1-gram overlap :  tensor([[0.9091, 0.4444, 0.0000], [0.9091, 0.4444, 0.0000]])
print(get_score(ref_tensor_batch, sys_tensor_batch, pad_id=1, n_gram=2))
==> batch_size =2,  2-gram overlap :  tensor([[0.6667, 0.2857, 0.0000],  [0.6667, 0.2857, 0.0000]])
```
