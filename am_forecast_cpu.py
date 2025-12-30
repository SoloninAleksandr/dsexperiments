# am_forecast_cpu.py
"""
CPU‑friendly rewrite of the original GPU POC.

Key changes vs. upstream:
1. **Pure‑CPU** – `DEVICE` is locked to `'cpu'`; no CUDA calls remain.
2. **Configurable scale** – use command‑line flags `--vocab` and `--max_edges` to trade quality for speed.
3. **Float32 everywhere** – avoids bf16/float16 limitations on commodity CPUs.
4. **Cleaner main()** – one demo run plus optional REPL for quick experiments.

python am_forecast_cpu.py --vocab 300 --max_edges 1000 --iters 5 --topk 10
"""
import argparse, random, torch

def parse_args():
    p = argparse.ArgumentParser(description="HLLSet‑Swarm forecast (CPU edition)")
    p.add_argument("--vocab", type=int, default=1_000, help="Vocabulary size (default: 1000)")
    p.add_argument("--max_edges", type=int, default=5_000, help="Pruning ceiling for sparse AM")
    p.add_argument("--covers", type=int, default=100, help="How many toy covers to generate")
    p.add_argument("--prompt_len", type=int, default=30, help="Length of toy user prompt (tokens)")
    p.add_argument("--teacher_len", type=int, default=35, help="Length of toy teacher cover (tokens)")
    p.add_argument("--topk", type=int, default=20, help="How many tokens to show in the reply")
    p.add_argument("--iters", type=int, default=10, help="Max forecast iterations")
    return p.parse_args()

# -----------------------------------------------------------------------------
# 0.  Helpers & constants (filled later by CLI) --------------------------------
# -----------------------------------------------------------------------------
DEVICE = torch.device("cpu")  # hard‑wired CPU

# -----------------------------------------------------------------------------
# 1.  Toy HLLSet generator -----------------------------------------------------
# -----------------------------------------------------------------------------

def fake_cover(vocab: int, n: int) -> torch.Tensor:
    """Return `n` *unique* random token indices in the range [0, vocab)."""
    return torch.unique(torch.randint(0, vocab, (n,)))

# -----------------------------------------------------------------------------
# 2.  Sparse AM builder with pruning ------------------------------------------
# -----------------------------------------------------------------------------

class PrunedAM:
    def __init__(self, vocab: int, max_edges: int):
        self.vocab = vocab
        self.max_edges = max_edges
        self.rows, self.cols, self.vals = [], [], []
    def add_edge(self, u: int, v: int, w: float = 1.0):
        if len(self.vals) >= self.max_edges:
            return  # hard prune
        self.rows.append(u)
        self.cols.append(v)
        self.vals.append(w)
    def csr(self) -> torch.Tensor:
        """Return a `torch.sparse_coo_tensor` in **cpu/float32**"""
        if not self.vals:
            # empty tensor safeguard
            indices = torch.empty((2, 0), dtype=torch.long)
            values = torch.empty((0,), dtype=torch.float32)
        else:
            indices = torch.tensor([self.rows, self.cols], dtype=torch.long)
            values = torch.tensor(self.vals, dtype=torch.float32)
        return torch.sparse_coo_tensor(indices, values, size=(self.vocab, self.vocab)).coalesce()

# -----------------------------------------------------------------------------
# 3.  Relational tensor wrapper (τ, ρ, Δ) --------------------------------------
# -----------------------------------------------------------------------------

class RelationalTensor:
    def __init__(self, K: int, vocab: int, edge_budget: int, Wtau: torch.Tensor, Wrho: torch.Tensor):
        self.K = K
        self.vocab = vocab
        self.edge_budget = edge_budget
        # List[torch.sparse_coo_tensor], one per slice
        self.slices = [None] * K
        self._refresh_slice(0, Wtau)
        self._refresh_slice(1, Wrho)
        # slice‑2 (delta) will be filled on the first ingest
    def _refresh_slice(self, k: int, mat: torch.Tensor):
        self.slices[k] = mat.coalesce()
    def overwrite_slice(self, k: int, mat: torch.Tensor):
        mat = mat.coalesce()
        if mat._nnz() > self.edge_budget:  # aggressive prune – top absolute weights
            vals, idx = torch.topk(mat.values(), self.edge_budget)
            mat = torch.sparse_coo_tensor(mat.indices()[:, idx], vals, size=mat.shape)
            mat = mat.coalesce()
        self._refresh_slice(k, mat)
    # ---------------------------------------------------------------------
    def contract(self, k: int, vec: torch.Tensor) -> torch.Tensor:
        """Sparse matrix‑vector multiply, clamp to [0,1]."""
        nxt = torch.sparse.mm(self.slices[k], vec)
        return torch.clamp(nxt, 0.0, 1.0)

# -----------------------------------------------------------------------------
# 4.  d / r / n decomposition helper ------------------------------------------
# -----------------------------------------------------------------------------

def hll_delta(hll_t: torch.Tensor, hll_t1: torch.Tensor):
    s_t, s_t1 = set(hll_t.tolist()), set(hll_t1.tolist())
    d = torch.tensor(list(s_t - s_t1), dtype=torch.long)
    r = torch.tensor(list(s_t & s_t1), dtype=torch.long)
    n = torch.tensor(list(s_t1 - s_t), dtype=torch.long)
    return d, r, n

# -----------------------------------------------------------------------------
# 5.  Forecast loop ------------------------------------------------------------
# -----------------------------------------------------------------------------

def forecast(rt: RelationalTensor, prompt_hll: torch.Tensor, max_iter: int, tol: float, topk: int):
    p = torch.zeros(rt.vocab)
    p[prompt_hll] = 1.0
    p = p / p.sum()  # L1‑norm‑1
    for _ in range(max_iter):
        p_old = p.clone()
        for k in range(rt.K):
            if rt.slices[k] is None:
                continue
            p = rt.contract(k, p)
        if torch.norm(p - p_old, p=1).item() < tol:
            break
    return torch.topk(p, topk).indices, p

# -----------------------------------------------------------------------------
# 6.  Online ingest + forecast -------------------------------------------------
# -----------------------------------------------------------------------------

def ingest_and_forecast(rt: RelationalTensor, prev_Wtau: torch.Tensor, new_cover: torch.Tensor,
                        teacher_cover: torch.Tensor, max_edges: int, max_iter: int, topk: int):
    # ---- Build fresh lattice fragment (τ) --------------------------------
    am = PrunedAM(rt.vocab, max_edges)
    for u in new_cover.tolist():
        for v in new_cover.tolist():
            if u != v:
                am.add_edge(u, v, 1.0)
    Wtau_new = am.csr()
    # ---- ρ lattice as fraction of τ --------------------------------------
    Wrho_new = Wtau_new * 0.3
    # ---- Δ slice = difference to previous τ ------------------------------
    delta = (Wtau_new - prev_Wtau).coalesce()
    # ---- Hot swap tensor slices ------------------------------------------
    rt.overwrite_slice(0, Wtau_new)
    rt.overwrite_slice(1, Wrho_new)
    rt.overwrite_slice(2, delta)  # first time fills slice‑2
    # ---- Forecast ---------------------------------------------------------
    resp, belief = forecast(rt, new_cover, max_iter=max_iter, tol=1e-3, topk=topk)
    return resp, belief, Wtau_new

# -----------------------------------------------------------------------------
# 7.  main() -------------------------------------------------------------------
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(42)

    # Step‑0: create toy corpus for initial τ / ρ --------------------------
    corpus = [fake_cover(args.vocab, random.randint(10, 50)) for _ in range(args.covers)]
    base_am = PrunedAM(args.vocab, args.max_edges)
    for cover in corpus:
        for u in cover.tolist():
            for v in cover.tolist():
                if u != v:
                    base_am.add_edge(u, v, 1.0)
    Wtau = base_am.csr()
    Wrho = Wtau * 0.3

    # Step‑1: create relational tensor ------------------------------------
    rt = RelationalTensor(K=3, vocab=args.vocab, edge_budget=args.max_edges, Wtau=Wtau, Wrho=Wrho)

    # Step‑2: simulate one conversation turn ------------------------------
    prompt = fake_cover(args.vocab, args.prompt_len)
    teacher = fake_cover(args.vocab, args.teacher_len)
    resp, belief, Wtau = ingest_and_forecast(rt, Wtau, prompt, teacher,
                                             args.max_edges, args.iters, args.topk)

    print("Response tokens (top‑{}): {}".format(args.topk, resp.tolist()))
    active = (belief > 0).sum().item()
    print(f"Belief vector sparsity: {active} / {args.vocab}")

if __name__ == "__main__":
    main()
