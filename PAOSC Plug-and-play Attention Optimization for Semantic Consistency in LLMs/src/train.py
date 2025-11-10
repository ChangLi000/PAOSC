import torch
import random

def epsilon_greedy_unique_topk(length, batchsize, probs, epsilon=0.3, topk=5, device='cpu'):
    """
    每个样本返回 topk 个 index，如果随机则是从未选中过的 index 中随机选，
    否则是从未选中过的注意力分布中选最大 topk。
    """
    sampled_indices = []

    for b in range(batchsize):
        if random.random() < epsilon:
            idx = torch.randperm(length)[:topk].to(device)
        else:
            probs_copy = probs[b].clone()
            selected = []
            for _ in range(topk):
                max_idx = torch.argmax(probs_copy).item()
                selected.append(max_idx)
                probs_copy[max_idx] = -float('inf') 
            idx = torch.tensor(selected, device=device)
        sampled_indices.append(idx.unsqueeze(0))  # [1, topk]

    sampled_indices = torch.cat(sampled_indices, dim=0)  # [batchsize, topk]
    return sampled_indices


def train_generator(x, G, D, C, LLM, optimizer, criterion, device, topk=3, lambda_mask=0.3, withoutBCEloss=False, withoutConfidenceloss=False):
    G.train()
    D.eval()
    LLM.eval()
    if withoutBCEloss and withoutConfidenceloss:
        raise ValueError("Error: BCEloss and Confidence loss cannot be disabled at the same time!")
        
    
    input = {k: v.clone().to(LLM.device) for k, v in x['input'].items()}
    label = torch.tensor(x['label'], dtype=torch.float).to(device)  # [B]
    attention_mask = input['attention_mask'].to(device)
    output = LLM(**input)
    current_input = output.last_hidden_state.detach().to(device)
    B, L, E = output.last_hidden_state.shape

    with torch.no_grad():
        original_pred = D(current_input,attention_mask).squeeze(1)  # [B]

    total_log_probs = []
    
    logits, attn_weights = G(current_input, attention_mask)
        # Drop token from input
    probs = attn_weights[-1][:,:,-1,:].mean(dim=1)
    sampled_idx = epsilon_greedy_unique_topk(L, B, probs, epsilon=lambda_mask, topk=topk, device=device)

    for i, idx in enumerate(sampled_idx):
        for j in idx:
            attention_mask[i][j] = 0

    
    row_all_zero = torch.all(attention_mask == 0, dim=1)
    if torch.any(row_all_zero):
        attention_mask = input['attention_mask'].to(device).clone()
        
    
    with torch.no_grad():
        pred = D(current_input, attention_mask).squeeze(1)  # [B]

    log_prob = torch.log(torch.gather(probs, 1, sampled_idx).squeeze(1) + 1e-8).mean(dim=1)  # [B]
    total_log_probs.append(log_prob)  # list of [B]

    # 计算奖励
    delta_conf = (original_pred - pred).mean(dim=1)  # [B]
    label = label.long()
    cls = criterion(original_pred, label)  # scalar or [B], 
    cls_ = criterion(pred, label)  # scalar or [B], 

    reward = []

    if withoutBCEloss:
        for i in range(B):
            r = (delta_conf[i]).detach()
            reward.append(r)
    elif withoutConfidenceloss:
        for i in range(B):
            r = (torch.abs(cls[i]-cls_[i])).detach()
            reward.append(r)
    else:
        for i in range(B):
            r = (delta_conf[i] + torch.abs(cls[i]-cls_[i])).detach()
            reward.append(r)
        
    reward = torch.stack(reward)  # [B]
    reward = reward-reward.mean(dim=0)

    policy_loss = -(log_prob * reward).mean()

    logits = C(logits)
    
    loss = criterion(logits, label).mean() + policy_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.detach(), attention_mask, total_log_probs, logits


def train_discriminator(x, D, LLM, optimizer, criterion, device,mask=None):
    D.train()
    LLM.eval()
    input = {k: v.clone().to(LLM.device) for k, v in x['input'].items()}
    label = torch.tensor(x['label'], dtype=torch.float).to(device)  # [B]
    attention_mask = input['attention_mask'].to(device)
    output = LLM(**input)
    input_ids = output.last_hidden_state.detach().to(device)

    label = label.long()
    if mask is not None:
        mask = mask.to(device)
        x_ = D(input_ids,mask)
        x = D(input_ids,attention_mask)
        D_loss = criterion(x_.squeeze(1), label) + criterion(x.squeeze(1), label)
    else:
        x_label = D(input_ids,attention_mask)
    
        D_loss = criterion(x_label.squeeze(1), label)

    D_loss = D_loss.mean()
    # 反向传播
    optimizer.zero_grad()
    D_loss.backward()
    optimizer.step()
    
    return D_loss.detach()


def trainer(train_loader, G, D, C, model, tokenizer, optimizer_GC, optimizer_D, criterion_D, device, save_path=".", topk=3, max_steps=100, lambda_mask=0.3,epoch=3):

    count = 0
    for epoch in range(epoch):
        for x in train_loader:
            if min( [len(i[:i.index(tokenizer.pad_token)]) if tokenizer.pad_token in i else len(i) for i in x['words']]) <= topk :
                continue
            G_loss, mask, total_log_probs, \
            logits = train_generator(x, G, D, C, model, optimizer_GC, criterion_D, device, topk=topk, lambda_mask=lambda_mask, withoutBCEloss=False, withoutConfidenceloss=False)
            
            D_loss = train_discriminator(x, D, model, optimizer_D, criterion_D, device,mask.detach())
            #if count%100 == 0:
            print(f"Epoch {epoch}, Steps:{count}, D_loss = {D_loss:.4f}, G_loss = {G_loss:.4f}")
            print("==================")
            count += 1
            if count >max_steps:
                break
        if count >max_steps:
            break 

    torch.save(C.state_dict(), f"{save_path}/C.pth")
    torch.save(G.state_dict(), f"{save_path}/G.pth")
    torch.save(D.state_dict(), f"{save_path}/D.pth")
