"""exp001: NAQ-Feat + support-to-query cross-attention residual.

Place this file beside the original NAQ files (arg.py, data.py, model.py,
meta.py, task_generator.py, utils.py) and its data/ directory.

For each query node, ALL support nodes in the episode serve as keys/values:
    q_enhanced = q_gnn + attn_lambda * MHA(q_gnn, S_gnn, S_gnn)
The original NAQ-Feat pseudo-episode generator and GCN encoder are unchanged.
"""

import os
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from data import load_data
from model import GNNEncoder
from task_generator import proto_task_generator, downstream_task_generator
from utils import seed_everything


class NAQCrossAttention(nn.Module):
    def __init__(self, args, config, data):
        super().__init__()
        self.args = args
        self.data = data
        self.encoder = GNNEncoder(config)
        self.attention = nn.MultiheadAttention(embed_dim=args.latent, num_heads=args.attn_heads, dropout=args.attn_dropout, batch_first=True)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)

    def attend(self, queries, supports):
        # queries: [num_queries, D], supports: [N*K, D]
        output, _ = self.attention(queries.unsqueeze(0), supports.unsqueeze(0), supports.unsqueeze(0), need_weights=False)
        return queries + self.args.attn_lambda * output.squeeze(0)

    @staticmethod
    def euclidean_dist(x, y):
        return ((x[:, None, :] - y[None, :, :]) ** 2).sum(dim=-1)

    def forward(self, id_spt, y_spt, id_query, y_query):
        h = self.encoder(self.data.features, adj=self.data.adj)
        h_support = h[id_spt]
        h_query = self.attend(h[id_query], h_support)

        prototypes = h_support.reshape(self.args.n_way, self.args.k_shot, -1).mean(dim=1)
        logits = -self.euclidean_dist(h_query, prototypes)

        # Match the original NAQ label remapping logic.
        if self.args.k_shot == 1:
            classes = y_spt.tolist()
        else:
            classes = y_spt.reshape(-1, self.args.k_shot).float().mean(dim=1).long().tolist()
        label_new = torch.tensor([classes.index(int(label)) for label in y_query.tolist()], dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, label_new)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        acc = (logits.argmax(dim=1) == label_new).float().mean().item()
        return loss.item(), acc

    @torch.no_grad()
    def evaluate_task(self, id_spt, y_spt, id_query, y_query):
        self.eval()
        h = self.encoder(self.data.features, adj=self.data.adj)
        h_support = h[id_spt]
        h_query = h[id_query]

        # Preserve the original NAQ downstream logistic-regression protocol.
        # Condition both train and test embeddings on the SAME support context
        # to keep representation spaces consistent.
        z_support = self.attend(h_support, h_support).cpu().numpy()
        z_query = self.attend(h_query, h_support).cpu().numpy()
        y_support = y_spt.cpu().numpy()
        y_query_np = y_query.cpu().numpy()
        clf = LogisticRegression(max_iter=1000, C=1.0 / self.args.l2_penalty if self.args.l2_penalty > 0 else 1e12)
        clf.fit(z_support, y_support)
        return clf.score(z_query, y_query_np)


def parse_args():
    parser = argparse.ArgumentParser(description="exp001: NAQ-Feat + Cross Attention")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="Amazon_clothing")
    parser.add_argument("--epoch", type=int, default=2000)
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--k_shot", type=int, default=1)
    parser.add_argument("--k_shot_test", type=int, default=1)
    parser.add_argument("--q_query", type=int, default=10)
    parser.add_argument("--q_query_test", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent", type=int, default=256)
    parser.add_argument("--l2_penalty", type=float, default=1e-4)
    parser.add_argument("--num_valid_tasks", type=int, default=50)
    parser.add_argument("--num_downstream_tasks", type=int, default=500)
    parser.add_argument("--query_generation", type=str, default="NaQ")
    parser.add_argument("--type", type=str, default="feature", choices=["feature", "diffusion"])
    parser.add_argument("--PPR_alpha", type=float, default=0.1)
    parser.add_argument("--attn_lambda", type=float, default=0.1)
    parser.add_argument("--attn_heads", type=int, default=4)
    parser.add_argument("--attn_dropout", type=float, default=0.0)
    parser.add_argument("--eval_interval", type=int, default=160)
    parser.add_argument("--seed", type=int, default=1212)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.latent % args.attn_heads != 0:
        raise ValueError("--latent must be divisible by --attn_heads")
    if args.k_shot != 1:
        raise ValueError("Unsupervised NAQ task generation currently supports --k_shot 1 only")
    if args.query_generation != "NaQ":
        raise ValueError("Only --query_generation NaQ is supported")

    torch.set_num_threads(4)
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    args.setting = "unsup"
    data = load_data(args)
    if device.type == "cuda":
        data.set_device(str(device))
    data.add_diffusion(args)
    config = [("GCN", (data.features.shape[1], 2 * args.latent), True), ("GCN", (2 * args.latent, args.latent), True)]
    model = NAQCrossAttention(args, config, data).to(device)
    train_pool = [proto_task_generator(args, data) for _ in range(args.epoch)]
    valid_pool, test_pool = downstream_task_generator(args, data, args.n_way, args.k_shot_test)

    os.makedirs(f"results/{args.dataset}/exp001", exist_ok=True)
    best_valid = float("-inf")
    best_result = None
    start = time.time()
    print(f"exp001 | dataset={args.dataset} | lambda={args.attn_lambda} | device={device}", flush=True)

    for epoch, (id_spt, y_spt, id_query, y_query) in tqdm(enumerate(train_pool, start=1), total=len(train_pool), desc="exp001"):
        model.train()
        loss, train_acc = model(id_spt, y_spt.to(device), id_query, y_query.to(device))
        if epoch % args.eval_interval == 0:
            valid_acc = np.mean([model.evaluate_task(s, ys, q, yq) for s, ys, q, yq in valid_pool])
            test_scores = np.array([model.evaluate_task(s, ys, q, yq) for s, ys, q, yq in test_pool])
            test_acc = test_scores.mean()
            ci95 = 1.96 * test_scores.std() / np.sqrt(len(test_scores))
            print(f"[Epoch {epoch:5d}] loss={loss:.4f} train={train_acc:.4f} " f"valid={valid_acc:.4f} test={test_acc:.4f} ± {ci95:.4f}", flush=True)
            if valid_acc > best_valid:
                best_valid = valid_acc
                best_result = (epoch, valid_acc, test_acc, ci95)
                torch.save({"epoch": epoch, "model": model.state_dict(), "args": vars(args)}, f"results/{args.dataset}/exp001/best.pt")

    if best_result is not None:
        ep, val, test, ci = best_result
        result = f"Best epoch: {ep} | Valid: {val:.4f} | " f"Test: {test:.4f} ± {ci:.4f} (95% CI) | " f"Elapsed: {time.time() - start:.1f}s"
        print(result, flush=True)
        with open(f"results/{args.dataset}/exp001/result.txt", "w") as f:
            f.write(result + "\n")


if __name__ == "__main__":
    main()
