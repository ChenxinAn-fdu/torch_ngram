import torch


def pad2max(input_tensors, pad_id):
    """
    :param input_tensors: A List of tensors
    :param pad_id: the pad value
    :return:
    """
    assert type(pad_id) == int
    assert isinstance(input_tensors, list)
    assert isinstance(input_tensors[0], torch.LongTensor)
    max_len = 0
    for input_tensor in input_tensors:
        max_len = max(max_len, input_tensor.shape[-1])
    after_padding = []
    for input_tensor in input_tensors:
        pad_size = max_len - input_tensor.shape[-1]
        pad_tensor = torch.full([*input_tensor.shape[:-1], pad_size], pad_id,
                                device=input_tensor.device).long()
        after_padding.append(torch.cat([input_tensor, pad_tensor], dim=-1))
    return torch.stack(after_padding)


def form_ngram(input_tensor, n=2):
    """
    input_tensor: batch x sample_num x seq_len
    return: batch x seq_len-3 x 4
    """
    bsz, cand_num, seq_len = input_tensor.size(0), input_tensor.size(1), input_tensor.size(2)
    seq_len_clip = seq_len - n + 1
    input_tensor_repeated = input_tensor[:, :, None, :].repeat(1, 1, seq_len_clip, 1)
    help_matrix_1 = torch.triu(torch.ones(seq_len, seq_len))
    help_matrix_2 = torch.triu(torch.ones(seq_len, seq_len), diagonal=n)
    help_matrix = (help_matrix_1 - help_matrix_2)[:seq_len_clip].bool()[None, None, :, :]
    ret_tensor = torch.masked_select(input_tensor_repeated, help_matrix.to(input_tensor.device))
    return ret_tensor.view(bsz, cand_num, seq_len_clip, n)


def overlap_1gram(ref_tensor, sys_tensor):
    ref_tensor = ref


def get_score(ref_tensor, sys_tensor, pad_id, n_gram):
    """
    ref_tensor: batch x seq_len1
    sys_tensor: batch x sample_num x seq_len2
    """
    sys_padding = (~(sys_tensor == pad_id)).float()
    ref_padding = (~(ref_tensor == pad_id)).float()
    # 将 ref 和 sys的pad_id 换成不一样的 防止pad_id 的影响
    n = min(min(n_gram, ref_tensor.size(-1)), sys_tensor.size(-1))
    ref_lengths = torch.sum(ref_padding, dim=-1) - n + 1
    ref_ones = torch.ones_like(ref_lengths, device=ref_lengths.device)
    ref_lengths = torch.where(ref_lengths > 0, ref_lengths, ref_ones)
    sys_lengths = torch.sum(sys_padding, dim=-1) - n + 1
    sys_ones = torch.ones_like(sys_lengths, device=sys_lengths.device)
    sys_lengths = torch.where(sys_lengths > 0, sys_lengths, sys_ones)
    ref_tensor = ref_tensor * ref_padding
    bsz, sample_num = sys_tensor.size(0), sys_tensor.size(1)
    ref_tensor = ref_tensor[:, None, :].repeat(1, sample_num, 1)
    if n_gram == 1:
        ref_tensor_1gram = ref_tensor.unsqueeze(-1).repeat(1, 1, 1, sys_tensor.shape[-1])
        sys_tensor_1gram = sys_tensor.unsqueeze(-1).permute(0, 1, 3, 2).repeat(1, 1, ref_tensor.shape[-1], 1)
        sim_matrix = (ref_tensor_1gram == sys_tensor_1gram).sum(dim=-1).sum(dim=-1)
    else:
        input_tensor1_ngram = form_ngram(ref_tensor, n).float()
        input_tensor2_ngram = form_ngram(sys_tensor, n).float()  # batch x sample_num x seq_len-3 x 4
        sim_matrix = torch.cosine_similarity(input_tensor2_ngram.unsqueeze(3), input_tensor1_ngram.unsqueeze(2),
                                             dim=-1) >= 1.0
        sim_matrix = torch.sum(torch.max(sim_matrix, dim=-1).values, dim=-1)
    length = sys_lengths + ref_lengths.unsqueeze(1)
    score = 2 * sim_matrix / length
    upper = torch.ones_like(score)
    score = torch.where(score < 1, score, upper)
    return score  # batch x sample_num
