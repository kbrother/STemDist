import argparse
import torch
import random
import numpy as np
import util
from model.mtgnn import gtnet
from tqdm import tqdm
from torch.func import functional_call, grad, vmap
import heapq
import sys
import math
import copy


def test_syn(args, data, device):
    num_total = data['train_loader'].xs.shape[0]
    num_nodes = data['train_loader'].xs.shape[2]
    in_dim = data['train_loader'].xs.shape[3]
    seq_len = data['train_loader'].xs.shape[1]  
    scaler = data['scaler']
    num_elems = round(args.reduction_rate * num_total)
    out_dim = 1

    _model = gtnet(True, True, 2, num_nodes, 
                      device, predefined_A=None, use_static_feat=False,
                      dropout=0.3, subgraph_size=20,
                      node_dim=10, dilation_exponential=1,             
                      seq_length=seq_len, in_dim=in_dim, out_dim=out_dim,
                      layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True)   
    _model = _model.to(device)
    optimizer = torch.optim.Adam(params=_model.parameters(), lr=args.lr)

    grads = get_grad_fast(_model, data, device)
    selected, gamma_selected, _ = craig(grads.to(device), num_elems, device)
    
    synx = data['train_loader'].xs[selected]
    syny = data['train_loader'].ys[selected]
    synx = torch.tensor(synx, dtype=torch.float, device=device)
    syny = torch.tensor(syny, dtype=torch.float, device=device)
    syny = scaler.transform(syny)
    gamma_selected = gamma_selected / torch.sum(gamma_selected) * num_elems
    
    min_val_loss = sys.float_info.max
    for i in tqdm(range(args.epochs)):
        if (i+1)%args.check == 0:
            grads = get_grad_fast(_model, data, device)
            selected, gamma_selected, _ = craig(grads.to(device), num_elems, device)
    
            synx = data['train_loader'].xs[selected]
            syny = data['train_loader'].ys[selected]
            synx = torch.tensor(synx, dtype=torch.float, device=device)
            syny = torch.tensor(syny, dtype=torch.float, device=device)
            syny = scaler.transform(syny)
            gamma_selected = gamma_selected / torch.sum(gamma_selected) * num_elems
        
        _model.train()
        output_syn = _model(synx.transpose(1, 3)).squeeze()
        loss_syn = torch.square(output_syn - syny) * gamma_selected.unsqueeze(-1).unsqueeze(-1)
        loss_syn = torch.mean(loss_syn)
        optimizer.zero_grad()
        loss_syn.backward()
        optimizer.step()
    
        _model.eval()
        if (i+1)%args.check == 0:
            with torch.no_grad():
                if args.ae:
                    val_loss = _model.test_model(data['val_loader'], scaler, device, args.ae)
                else:
                    val_loss = math.sqrt(_model.test_model(data['val_loader'], scaler, device))

            print(f"epoch :{i}, train loss: {loss_syn}, val loss: {val_loss}")
            if min_val_loss > val_loss:
                min_i = i
                min_val_loss = val_loss
                min_params = copy.deepcopy(_model.state_dict())

    _model.load_state_dict(min_params)
    _model.eval()
    with torch.no_grad():       
        if args.ae:
            test_loss =_model.test_model(data['test_loader'], scaler, device, args.ae)
        else:
            test_loss = math.sqrt(_model.test_model(data['test_loader'], scaler, device))
    print(f"min i: {min_i}, val loss: {min_val_loss}, test loss: {test_loss}")      
    with open(args.save_path, "a") as f:
        f.write(f"min i: {min_i}, val loss: {min_val_loss}, test loss: {test_loss}\n")

    
def craig(grads, budget, device, eps=1e-12):
    num_series, num_feat = grads.shape
    G = grads / (grads.norm(dim=1, keepdim=True) + eps)
    
    best_sim = torch.full((num_series,), -1e9, device=device)
    best_rep = torch.full((num_series,), -1, dtype=torch.long, device=device)
    sumvec = G.sum(dim=0)  # [D]
    init_score = (G @ sumvec)  # [N]  == sum_i cos(i,e)

    heap = [(-init_score[e].item(), int(e), 0) for e in range(num_series)]
    heapq.heapify(heap)

    selected = []
    it = 0

    def recompute_gain(e: int):
        # sim = (G @ G[e] + 1)/2
        sim = (G @ G[e] + 1.0) * 0.5  # [N]
        gain = torch.relu(sim - best_sim).sum().item()
        return gain, sim

    in_selected = torch.zeros(num_series, dtype=torch.bool, device=device)
    while len(selected) < budget:
        it += 1
        while True:
            neg_ub, e, _ = heapq.heappop(heap)
            if in_selected[e]:
                continue

            true_gain, sim_vec = recompute_gain(e)
            heapq.heappush(heap, (-true_gain, e, it))
            neg_top, e_top, _ = heap[0]
            if e_top == e:
                heapq.heappop(heap)
                selected.append(e)
                in_selected[e] = True

                improve = sim_vec > best_sim
                best_sim[improve] = sim_vec[improve]
                best_rep[improve] = e
                break

    gamma = torch.bincount(best_rep.clamp_min(0), minlength=num_series)  # [N], long
    gamma_selected = gamma[torch.tensor(selected, device=device)]

    return selected, gamma_selected, best_rep

    
def get_grad_fast(_model, _data, device):
    _model.train()  # 필요에 따라 eval()로 고정해도 됨 (dropout 등)

    scaler = _data['scaler']

    params = dict(_model.named_parameters())
    buffers = dict(_model.named_buffers())

    last_params = {k: v for k, v in params.items() if k.startswith("end_conv_2.")}
    if len(last_params) == 0:
        raise ValueError("end_conv_2.* 파라미터를 찾지 못했습니다.")

    def single_sample_loss(last_params_local, x_i, y_i):
        merged_params = params.copy()
        merged_params.update(last_params_local)

        x_i = x_i.unsqueeze(0).transpose(1, 3)
        y_i = y_i.unsqueeze(0)

        pred_i = functional_call(_model, (merged_params, buffers), (x_i,))
        pred_i = scaler.inverse_transform(pred_i).squeeze(-1)

        loss_i, num_i = util.masked_se(pred_i, y_i, 0.)
        loss_i = loss_i / num_i

        # grad를 위해 반드시 scalar여야 안정적입니다.
        # masked_se가 (1,) 형태를 주는 경우가 있어 안전하게 sum/mean 처리 권장
        return loss_i.sum()

    grad_fn = grad(single_sample_loss)

    all_grads = []  # (batch, P)들을 모았다가 마지막에 cat

    for x, y in _data['train_loader'].get_iterator():
        # torch.tensor(...)는 매번 새 메모리를 강제 할당합니다. as_tensor 권장
        x = torch.as_tensor(x, device=device, dtype=torch.float32)
        y = torch.as_tensor(y, device=device, dtype=torch.float32)

        grads_struct = vmap(
            grad_fn,
            in_dims=(None, 0, 0),
            randomness="different"
        )(last_params, x, y)

        grads_flat_list = []
        for name in sorted(grads_struct.keys()):
            g = grads_struct[name]                # (B, *shape)
            grads_flat_list.append(g.reshape(g.shape[0], -1))

        grad_flat_curr = torch.cat(grads_flat_list, dim=1)

        # 매우 중요: 그래프 끊기
        grad_flat_curr = grad_flat_curr.detach()

        # 보통 CRAIG에서 전체 train-set grads는 매우 큽니다.
        # GPU에 계속 쌓지 말고 CPU에 저장하는 것을 강력 권장
        all_grads.append(grad_flat_curr.cpu())

        # 명시적으로 레퍼런스 제거 (파이썬 GC가 더 빨리 회수하도록)
        del grads_struct, grads_flat_list, grad_flat_curr, x, y

    grads_flat = torch.cat(all_grads, dim=0)  # CPU tensor
    return grads_flat
    

# python -m coreset.craig -de 1 -d ../data/GBA -s 0 -b 32 -rr 5e-3 -lr 1e-2 -e 400 -c 10 -sp result_craig/gba_1e-2
if __name__ == "__main__":
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('-de', '--device', type=int, default=0, help='')
    parser.add_argument('-d', '--data', type=str, default='../data/METR-LA', help='data path')
    parser.add_argument('-s', '--seed', type=int, default=0, help='')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('-rr', '--reduction_rate',type=float,default=1e-3,help='learning rate')
    parser.add_argument('-lr', '--lr',type=float,default=1e-3,help='learning rate')
    parser.add_argument('-e', '--epochs',type=int,default=100,help='')
    parser.add_argument('-c', '--check', type=int, default=5, help='')
    parser.add_argument('-a', '--ae', action='store_true')
    parser.add_argument('-sp', '--save_path', type=str, default='results/') 
    args = parser.parse_args()

    # random seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.device}")
    dataloader =  util.load_dataset(args.data, args.batch_size)
    print("load finish")

    test_syn(args, dataloader, device)