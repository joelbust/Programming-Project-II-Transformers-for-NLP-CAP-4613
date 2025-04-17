import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
from tqdm import tqdm

# load bert tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
model.eval()

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# BERT embedding for words
def get_word_embedding(word):
    inputs = tokenizer(word, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.squeeze(0)  # shape: (num_tokens, hidden_dim)
    return embeddings.mean(dim=0)  # average if word splits into subwords

def load_analogies(filename):
    groups = defaultdict(list)
    current_group = None
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            # skips empty lines/comments, needing for start of file
            if not line or line.startswith("//"):
                continue
            #  : = new group
            if line.startswith(":"):
                current_group = line[2:].strip()
            elif current_group is not None:
                words = line.split()
                # checks if a valid analogy
                if len(words) == 4:
                    groups[current_group].append(words)
    return groups


# cosine scoring
def get_top_k_cosine(query, candidates, k):
    sims = [(cand_word, F.cosine_similarity(query, cand_emb, dim=0).item())
            for cand_word, cand_emb in candidates.items()]
    sims.sort(key=lambda x: x[1], reverse=True)
    return [word for word, _ in sims[:k]]

# L2 scoring
def get_top_k_l2(query, candidates, k):
    dists = [(cand_word, torch.norm(query - cand_emb).item())
             for cand_word, cand_emb in candidates.items()]
    dists.sort(key=lambda x: x[1])
    return [word for word, _ in dists[:k]]

# the main loop
def evaluate_group(name, analogies, ks=[1, 2, 5, 10, 20]):
    accuracy_cosine = {k: 0 for k in ks}
    accuracy_l2 = {k: 0 for k in ks}
    total = 0

    print(f"\nEvaluating group: {name}...")

    # caching the embeddings
    words = set()
    for a, b, c, d in analogies:
        words.update([a, b, c, d])
    word_embeds = {w: get_word_embedding(w) for w in tqdm(words)}

    for a, b, c, d in tqdm(analogies):
        total += 1
        # query vector = b - a + c
        q = word_embeds[b] - word_embeds[a] + word_embeds[c]

        # candiates = 2nd and 4th from other lines
        candidates = {w: word_embeds[w] for x, y, z, w in analogies if w != d for w in [y, w]}

        # cosine acc
        for k in ks:
            topk = get_top_k_cosine(q, candidates, k)
            if d in topk:
                accuracy_cosine[k] += 1

        # L2 acc
        for k in ks:
            topk = get_top_k_l2(q, candidates, k)
            if d in topk:
                accuracy_l2[k] += 1

    # normalize
    for k in ks:
        accuracy_cosine[k] /= total
        accuracy_l2[k] /= total

    return accuracy_cosine, accuracy_l2
