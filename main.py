# main.py — A3C Netlist Optimiser, with first‐segment heavy workers
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import pathlib
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import netlist_processor   # 같은 폴더에 있어야 함

# ── 경로 설정 (iccad/release 구조) ─────────────────────────
BASE_DIR = pathlib.Path(__file__).resolve().parent
REL_DIR  = BASE_DIR / "release"
CE_DIR   = REL_DIR / "cost_estimators"
for f in os.listdir(CE_DIR):
    os.chmod(CE_DIR / f, 0o755)

DESIGN_TPL = REL_DIR / "netlists/design{}.v"
COST_TPL   = REL_DIR / "cost_estimators/cost_estimator_{}"
LIB_PATH   = REL_DIR / "lib/lib1.json"

# ── 라이브러리 데이터 로드 ─────────────────────────────────
netlist_processor.LIBRARY_CELLS = netlist_processor.load_library(str(LIB_PATH))
lib_data = netlist_processor.library_datas(str(LIB_PATH))

# ── 하이퍼파라미터 & 로깅 설정 ─────────────────────────────
GATE_MAP    = {"and":0,"buf":1,"nand":2,"nor":3,"not":4,"or":5,"xnor":6,"xor":7}
LEARNING_LR = 3e-4
H1, H2 = 128, 64
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Cost 평가용 프로세스 풀 ─────────────────────────────────
_cost_executor = ProcessPoolExecutor(max_workers=4)

def _eval_cost(net_state, cost_path, lib_path):
    return netlist_processor.cost(net_state, cost_path, lib_path)

# ── 전체 게이트 특성 텐서 캐싱 ─────────────────────────────
_full_tensors = {}
def build_full_tensor(net):
    key = id(net)
    if key not in _full_tensors:
        rows = []
        for g in net.gates:
            base = GATE_MAP[g.base_type.lower()]
            sub  = g.sub_num
            feat = lib_data[f"{g.base_type.lower()}_{sub}"]
            rows.append([base, sub, *feat])
        _full_tensors[key] = torch.tensor(rows, dtype=torch.float32, device=DEVICE)
    return _full_tensors[key]

# ── Actor-Critic 네트워크 (LSTM 제거) ────────────────────────
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim=14):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, H1)
        self.fc2 = nn.Linear(H1, H2)
        self.pi  = nn.Linear(H2, action_dim)
        self.v   = nn.Linear(H2, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.pi(h), self.v(h)

class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=LEARNING_LR):
        super().__init__(params, lr=lr)

# ── Segment 환경 정의 ─────────────────────────────────────
class SegmentEnv:
    def __init__(self, base_net, full_tensor, seg_idx, cost_path):
        self.base_net    = copy.deepcopy(base_net)
        self.full_tensor = full_tensor
        self.seg_idx     = seg_idx
        self.cost_path   = cost_path
        self.reset()

    def reset(self):
        self.net = copy.deepcopy(self.base_net)
        self.ptr = 0
        self.tens = self.full_tensor[self.seg_idx]
        return self.tens

    def allowed(self):
        g = self.net.gates[self.seg_idx[self.ptr]]
        idx = GATE_MAP[g.base_type.lower()]
        if idx in [0,2,3,5]: return list(range(1,9))
        if idx in [1,4]:     return list(range(1,15))
        if idx in [6,7]:     return list(range(1,7))
        raise RuntimeError("Unsupported gate type")

    def step(self, action):
        tens = self.tens.clone()
        gidx = self.seg_idx[self.ptr]
        gate = self.net.gates[gidx]
        gate.sub_num = action
        key = f"{gate.base_type.lower()}_{action}"
        with torch.no_grad():
            new_feat = torch.tensor(
                [GATE_MAP[gate.base_type.lower()], action, *lib_data[key]],
                dtype=torch.float32, device=DEVICE
            )
            tens[self.ptr] = new_feat
        self.ptr += 1
        if self.ptr == len(self.seg_idx):
            cost = _eval_cost(self.net, self.cost_path, str(LIB_PATH))
            return None, -cost, True, {}
        return tens.clone(), 0.0, False, {}

# ── A3C 워커 ─────────────────────────────────────────────
def worker(wid, global_net, optimizer,
           global_ep, max_ep, lock, best_info,
           base_net, seg_idx, cost_path, input_dim):
    local_net = ActorCritic(input_dim).to(DEVICE)
    local_net.load_state_dict(global_net.state_dict())
    env = SegmentEnv(base_net, build_full_tensor(base_net), seg_idx, cost_path)

    while True:
        with lock:
            if global_ep.value >= max_ep: break
            global_ep.value += 1

        state = env.reset()
        logps, vals, ents = [], [], []
        R = 0.0
        done = False

        while not done:
            # 현재 pointer 위치의 feature만 사용
            ptr = env.ptr
            feat = state[ptr].unsqueeze(0)           # shape (1, input_dim)
            logits, value = local_net(feat)          # logits: (1,action_dim), value: (1,1)

            acts = env.allowed()
            mask = torch.zeros_like(logits)
            mask[0, torch.tensor(acts)-1] = 1
            logits = logits.masked_fill(mask==0, torch.finfo(logits.dtype).min)

            probs = F.softmax(logits, dim=-1)
            dist  = torch.distributions.Categorical(probs)
            ai    = dist.sample()                    # tensor([k])
            logp  = dist.log_prob(ai).unsqueeze(1)   # shape (1,1)
            ent   = dist.entropy().unsqueeze(1)

            action = acts[ai.item()]
            next_state, reward, done, _ = env.step(action)
            R += reward
            if next_state is not None:
                state = next_state.clone().detach()

            logps.append(logp)
            vals.append(value)
            ents.append(ent)

        cost = -R
        adv = ( -cost - vals[-1] ).detach()
        loss = -(torch.cat(logps)*adv).mean() \
             + 0.5*adv.pow(2).mean() \
             - 0.01*torch.cat(ents).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(local_net.parameters(), 40)
        for lp, gp in zip(local_net.parameters(), global_net.parameters()):
            if lp.grad is not None:
                gp.grad = lp.grad.cpu()
        optimizer.step()
        local_net.load_state_dict(global_net.state_dict())

        with lock:
            if cost < best_info['cost']:
                best_info.update(cost=cost, net=copy.deepcopy(env.net))

# ── Run Segment Optimization ───────────────────────────────
def run_segment(base_net, seg_idx, episodes, workers, cost_path, input_dim):
    global_net = ActorCritic(input_dim)
    global_net.share_memory()
    optimizer = SharedAdam(global_net.parameters())
    manager = mp.Manager()
    best_info = manager.dict(cost=float('inf'), net=None)
    global_ep = mp.Value('i', 0)
    lock = mp.Lock()

    procs = []
    for w in range(workers):
        p = mp.Process(target=worker, args=(
            w, global_net, optimizer,
            global_ep, episodes, lock, best_info,
            base_net, seg_idx, cost_path, input_dim
        ))
        p.start()
        procs.append(p)

    pbar = tqdm(total=episodes, desc="Episodes")
    last = 0
    while True:
        time.sleep(0.5)
        with lock:
            curr = global_ep.value
        inc = curr - last
        if inc > 0:
            pbar.update(inc)
            last = curr
        if curr >= episodes:
            break
    pbar.close()
    for p in procs:
        p.join()

    return best_info['net'], best_info['cost']

# ── Full Design Optimization & CLI ─────────────────────────
def optimise_design(design_idx, cost_idx, seg_size, first_ep, other_ep, workers):
    dpath = str(DESIGN_TPL).format(design_idx)
    cpath = str(COST_TPL).format(cost_idx)
    base = netlist_processor.parse(dpath)
    init_cost = netlist_processor.cost(base, cpath, str(LIB_PATH))
    logging.warning(f"Design {design_idx} init cost={init_cost:.4f}")

    G = len(base.gates)
    num_seg = max(1, math.ceil(G / seg_size))
    seg_sz  = math.ceil(G / num_seg)
    SEGMENTS = [list(range(i*seg_sz, min((i+1)*seg_sz, G))) for i in range(num_seg)]
    input_dim = len(lib_data['and_1']) + 2
    net       = base

    for i, seg in enumerate(SEGMENTS):
        ep = first_ep if i == 0 else other_ep
        wk = workers if i == 0 else 1   # 첫 segment만 다중 worker, 나머지는 1개
        best_net, best_cost = run_segment(
            net, seg, ep, wk, cpath, input_dim
        )
        for idx in seg:
            net.gates[idx] = best_net.gates[idx]
        logging.warning(f"Seg {i+1}/{num_seg} done | cost={best_cost:.3f}")

    final_cost = netlist_processor.cost(net, cpath, str(LIB_PATH))
    logging.warning(f"Final cost={final_cost:.4f} Δ={init_cost-final_cost:.4f}")
    out_file = f"final_design{design_idx}_cost{cost_idx}.v"
    with open(out_file, 'w') as f:
        f.write(netlist_processor.generate_verilog_forCE(net))
    logging.warning(f"Saved → {out_file}")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--design', required=True)
    parser.add_argument('-c','--cost',   required=True)
    parser.add_argument('--seg_size', type=int, default=200, help='target gates per segment')
    parser.add_argument('--first_ep', type=int, default=3000)
    parser.add_argument('--other_ep', type=int, default=20)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    for d in map(int, args.design.split(',')):
        for c in map(int, args.cost.split(',')):
            optimise_design(d, c, args.seg_size, args.first_ep, args.other_ep, args.workers)
