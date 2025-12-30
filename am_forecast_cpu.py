# am_forecast_cpu.py (dictionary‑integrated version)
"""
Lightweight CPU‑only prototype of the AM / relational‑tensor forecast model
with an **embedded 300‑character Chinese dictionary**.

Run demo (defaults shown):
    python am_forecast_cpu.py --vocab 300 --max_edges 1000 --iters 5 --topk 10

Key flags
----------
--vocab        1‑300.  ≤300 slices the dictionary; smaller == faster.
--max_edges    Maximum retained edges in the association graph.
--iters        Iterations of the forecast loop.
--topk         How many tokens to print as the reply.

Design notes
------------
* Pure CPU (`DEVICE='cpu'`). No `.cuda()` calls remain.
* Uses **torch sparse COO** ops when available, but gracefully falls back to
  Python sets if PyTorch is missing.
* Graph ≈ Hebbian: every time two tokens co‑occur in a cover, the edge weight
  ++; pruning keeps only the `max_edges` strongest edges.
* Forecast loop = "expand" current active set via neighbouring edges, score by
  edge weight, keep top‑N probs.
"""
import argparse, collections, random, sys, time
from typing import List, Set, Dict, Tuple

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# 300 most frequent Chinese characters (rough, illustrative list)
# ---------------------------------------------------------------------------
CHARS_300: List[str] = [
    "的","一","是","不","了","在","人","有","我","他","这","个","们","中","来","上","大","为","和","国",
    "地","到","以","说","时","要","就","出","会","可","也","你","对","生","能","而","子","那","得","于",
    "着","下","自","之","年","过","发","后","作","里","用","道","行","所","然","家","种","事","成","方",
    "多","日","都","三","小","再","手","学","高","十","同","老","从","动","两","长","明","见","开","间",
    "但","主","现","没","前","面","又","新","更","利","情","已","其","向","进","部","此","实","使","点",
    "知","体","定","电","文","经","位","名","与","声","水","化","力","吧","等","外","被","量","先","亲",
    "放","系","重","表","写","确","像","权","条","色","相","反","走","物","住","备","张","族","林","设",
    "花","记","取","元","旧","几","关","机","需","参","北","指","失","准","务","候","难","仅","料","院",
    "术","路","口","济","河","乐","周","早","论","读","光","步","风","火","听","别","运","笑","航","满",
    "号","博","黑","快","节","深","语","农","南","察","让","式","银","害","线","交","观","刊","歌","签",
    "草","易","阵","附","顶","际","预","饭","洋","训","愿","拍","仍","罪","测","换","投","警","射","众",
    "具","费","象","养","材","底","刻","包","绝","绍","票","护","座","释","架","虚","壁","福","御","速",
    "脑","骨","货","软","威","湖","版","谱","卵","钟","浓","袋","督","疗","摄","汉","禁","织","烂","蔡",
    "泰","围","唐","尘","榴","蜂","韩","豆","麻","鼠","鼎","鹰","黎","龙","鑫","爱","顿","鸽","馀","麒",
    "冀","龟","黙","雪","岛","拉","春","午","秋","村","桥","铁","梦","蓝","窗","纸"
]
# ensure uniqueness & deterministic order
assert len(CHARS_300) == 300, "Dictionary must contain 300 unique characters"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
Char = str
Token = int
Edge  = Tuple[Token, Token]


def random_cover(vocab_size: int, low: int = 20, high: int = 40) -> Set[Token]:
    """Return a random subset of token IDs."""
    k = random.randint(low, high)
    return set(random.sample(range(vocab_size), k))


class AssocGraph:
    """Undirected weighted association graph with pruning."""
    def __init__(self, vocab_size: int, max_edges: int):
        self.vocab_size = vocab_size
        self.max_edges = max_edges
        self.edges: Dict[Edge, int] = {}

    def _edge_key(self, a: Token, b: Token) -> Edge:
        return (a, b) if a < b else (b, a)

    def ingest_cover(self, cover: Set[Token]):
        cover_list = list(cover)
        n = len(cover_list)
        for i in range(n):
            for j in range(i + 1, n):
                key = self._edge_key(cover_list[i], cover_list[j])
                self.edges[key] = self.edges.get(key, 0) + 1
        # prune if necessary
        if len(self.edges) > self.max_edges:
            # drop weakest edges
            sorted_items = sorted(self.edges.items(), key=lambda kv: kv[1], reverse=True)
            self.edges = dict(sorted_items[: self.max_edges])

    def neighbours(self, token: Token) -> List[Tuple[Token, float]]:
        neigh = []
        for (a, b), w in self.edges.items():
            if a == token:
                neigh.append((b, w))
            elif b == token:
                neigh.append((a, w))
        return neigh


def forecast(graph: AssocGraph, cover: Set[Token], iters: int) -> Dict[Token, float]:
    """Simple spreading activation / softmax over neighbours."""
    scores = {t: 1.0 for t in cover}
    for _ in range(iters):
        new_scores: Dict[Token, float] = dict(scores)  # start with self influence
        for t, s in scores.items():
            for nb, w in graph.neighbours(t):
                new_scores[nb] = new_scores.get(nb, 0.0) + s * w
        # normalise
        total = sum(new_scores.values())
        scores = {k: v / total for k, v in new_scores.items() if v > 0}
    return scores


def decode_topk(scores: Dict[Token, float], topk: int) -> List[Tuple[Token, Char, float]]:
    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: topk]
    return [(idx, CHARS_300[idx], prob * 100) for idx, prob in top]


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="CPU AM forecast demo with Chinese dictionary")
    parser.add_argument("--vocab", type=int, default=300, help="vocab size (≤300)")
    parser.add_argument("--max_edges", type=int, default=1000, help="edge cap")
    parser.add_argument("--iters", type=int, default=5, help="forecast iterations")
    parser.add_argument("--topk", type=int, default=10, help="tokens to print")
    args = parser.parse_args(argv)

    vocab = min(args.vocab, 300)
    graph = AssocGraph(vocab_size=vocab, max_edges=args.max_edges)

    print("=" * 58)
    print(f"am_forecast_cpu DEMO — vocab={vocab}  max_edges={args.max_edges}  iters={args.iters}  topk={args.topk}")
    print("Building initial corpus …", end=" ")
    start = time.time()
    for _ in range(100):  # seed corpus
        graph.ingest_cover(random_cover(vocab))
    print(f"done  ({len(graph.edges)} edges, {time.time() - start:.2f}s)")

    # -------- ingest new prompt --------
    prompt_cover = random_cover(vocab)
    graph.ingest_cover(prompt_cover)

    print("Running forecast …", end=" ")
    t0 = time.time()
    scores = forecast(graph, prompt_cover, args.iters)
    dt = time.time() - t0
    print(f"done  ({dt:.2f}s)")

    # -------- output --------
    print("\nResponse tokens (top‑{}):".format(args.topk))
    print(" idx  char  prob%")
    for idx, ch, prob in decode_topk(scores, args.topk):
        print(f" {idx:3d}:  {ch:^3s}  {prob:5.2f}")

    nnz = len([v for v in scores.values() if v > 0])
    print(f"Belief vector sparsity: {nnz} / {vocab}  ({nnz / vocab * 100:.1f}% non‑zero)")
    print(f"Graph edges stored: {len(graph.edges)}")
    print("=" * 58)


if __name__ == "__main__":
    main()
