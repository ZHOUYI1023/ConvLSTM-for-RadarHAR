import torch
import torch.nn.functional as F


def smoothSeq(seq):
    cumulative_sum = torch.cumsum(seq, dim=1)
    accumulated_time = torch.arange(1, seq.size(1) + 1, dtype=seq.dtype, device=seq.device)
    smoothed_seq = cumulative_sum / accumulated_time.view(1, seq.size(1), 1)
    return smoothed_seq

def softplus(x):
    return torch.log(1 + torch.exp(x))

def rank_loss(f_st, f, beta):
    loss = 0
    _, length, feature_size = f.shape
    f_smooth = smoothSeq(f)
    for i in range(length-1):
        theta = torch.sum(f_st.squeeze() * f_smooth[:, i+1, :].squeeze(), dim=1) - torch.sum(f_st.squeeze() * f_smooth[:, i, :].squeeze(), dim=1) + beta
        time_loss = softplus(theta) 
        #print(time_loss)
        #print(loss)
        loss += time_loss
    loss /=  length-1
    return torch.mean(loss)


def rank_loss_normal(f_st, f, beta):
    loss = 0
    _, length, feature_size = f.shape
    f = smoothSeq(f)
    for i in range(length-1):
        # Normalize input features
        f_st_normalized = F.normalize(f_st, p=2, dim=1)
        f_i_normalized = F.normalize(f[:, i, :], p=2, dim=1)
        f_iplus1_normalized = F.normalize(f[:, i+1, :], p=2, dim=1)
        
        # Calculate dot products
        dot_product_i = torch.sum(f_st_normalized * f_i_normalized, dim=1)
        dot_product_iplus1 = torch.sum(f_st_normalized * f_iplus1_normalized, dim=1)
        # Calculate loss
        theta = dot_product_iplus1 - dot_product_i + beta # 64-dimensional
        loss += softplus(theta) 
    return torch.mean(loss)/(length-1)