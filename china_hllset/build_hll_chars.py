# file: build_hll_chars.py
import re, gzip, json, hashlib, pathlib, requests, itertools
from datasketch import HyperLogLog
from tqdm import tqdm
from lxml import etree

# ---------- 1. Данные ----------
CEDICT_URL = "https://www.mdbg.net/chinese/export/cedict/cedict_1_0_ts_utf-8_mdbg.txt.gz"
FREQ_URL   = "https://raw.githubusercontent.com/mozillazg/zhfreqlist/master/mandarin_char_1000.txt"

WORKDIR = pathlib.Path("data"); WORKDIR.mkdir(exist_ok=True)
cedict_gz  = WORKDIR / "cedict.txt.gz"
freq_file  = WORKDIR / "freq.txt"

def download(url: str, to: pathlib.Path):
    if not to.exists():
        print(f"Downloading {url} → {to}")
        r = requests.get(url, timeout=30)
        to.write_bytes(r.content)

download(CEDICT_URL, cedict_gz)
download(FREQ_URL,   freq_file)

# ---------- 2. Частотный список ----------
chars = [line.strip().split()[0] for line in freq_file.read_text().splitlines()][:1200]

# ---------- 3. Парсим CC-CEDICT ----------
def iter_cedict_entries(gz_path):
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"): continue
            trad, simp, rest = line.split(" ", 2)
            pinyin, eng = rest.split("] ", 1)
            eng = eng.strip("/").replace("/", "; ")
            yield simp, eng.lower()

cedict_map = {simp: desc for simp, desc in iter_cedict_entries(cedict_gz) if simp in chars}

# ---------- 4. HLL для каждого иероглифа ----------
def tokens(text):
    return re.findall(r"[a-z']+", text)

hll_map, hll_params = {}, dict(p=14)  # p=14 ~ 1.6 KB/эскиз, ~1% σ
for ch, definition in tqdm(cedict_map.items(), desc="HLL sets"):
    h = HyperLogLog(**hll_params)
    for tok in tokens(definition):
        h.add(tok.encode())
    hll_map[ch] = h

# ---------- 5. Строим граф похожести ----------
GRAPH_THRESHOLD = 0.10
edges = []
char_list = list(hll_map)
for a, b in tqdm(itertools.combinations(char_list, 2), total=len(char_list)*(len(char_list)-1)//2, desc="Similarity"):
    h1, h2 = hll_map[a], hll_map[b]
    inter = HyperLogLog.jaccard(h1, h2)
    if inter >= GRAPH_THRESHOLD:
        edges.append((a, b, round(inter, 3)))

# ---------- 6. Инвертированный индекс ----------
inv_index = {}
for ch, h in hll_map.items():
    for tok in tokens(cedict_map[ch]):
        hsh = hashlib.blake2b(tok.encode(), digest_size=8).hexdigest()
        inv_index.setdefault(hsh, []).append(ch)

# ---------- 7. Сериализация ----------
def serialize_hll(h: HyperLogLog):  # bytes → hex
    return h.bytes().hex()

out = {
    "meta": {"hll_p": hll_params["p"]},
    "characters": {
        ch: {
            "cedict": cedict_map[ch],
            "hll_hex": serialize_hll(hll_map[ch]),
        } for ch in hll_map
    },
    "edges": edges,
    "inv_index": inv_index,
}
json.dump(out, open("char_hll_db.json", "w"), ensure_ascii=False, indent=2)
print("✓ char_hll_db.json ready")
