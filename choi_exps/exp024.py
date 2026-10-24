"""exp024: evaluate original NAQ exp000 checkpoint using Euclidean ProtoNet (5-way 1-shot).

Run from the NAQ repository root so local data/model/task_generator modules resolve.
No training or checkpoint modification occurs.
"""

import argparse
import json
import os

import numpy as np
import torch

from data import load_data
from model import GNNEncoder
from task_generator import downstream_task_generator
from utils import seed_everything


def get_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint type: {type(checkpoint).__name__}")

    for name in ["network", "encoder", "model_state_dict", "state_dict", "model"]:
        state = checkpoint.get(name)
        if isinstance(state, dict) and any(torch.is_tensor(v) for v in state.values()):
            return state

    if any(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint

    raise ValueError(f"Could not find weights. Checkpoint keys: {list(checkpoint.keys())}")


def load_encoder(encoder, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = get_state_dict(checkpoint)
    expected = encoder.state_dict()
    for prefix in ["", "network.", "encoder.", "module.", "module.network.", "module.encoder."]:
        candidate = {k[len(prefix) :] if prefix and k.startswith(prefix) else k: v for k, v in state.items()}
        candidate = {k: v for k, v in candidate.items() if k in expected}
        if set(candidate) == set(expected) and all(candidate[k].shape == expected[k].shape for k in expected):
            encoder.load_state_dict(candidate, strict=True)
            return checkpoint

    raise RuntimeError(
        f"Cannot match checkpoint to GNNEncoder. Expected: "
        f"{[(k, tuple(v.shape)) for k, v in expected.items()]}; "
        f"Found: {[(k, tuple(v.shape)) for k, v in state.items() if torch.is_tensor(v)][:30]}"
    )


def evaluate(embeddings, tasks, n_way, k_shot):
    scores = []
    for supports, y_support, queries, y_query in tasks:
        s = torch.as_tensor(supports, device=embeddings.device, dtype=torch.long)
        q = torch.as_tensor(queries, device=embeddings.device, dtype=torch.long)
        support_labels = torch.as_tensor(y_support).reshape(-1).tolist()
        query_labels = torch.as_tensor(y_query).reshape(-1).tolist()

        # Preserve order returned by NAQ's downstream_task_generator.
        class_labels = support_labels[::k_shot]
        if len(class_labels) != n_way or len(set(class_labels)) != n_way:
            raise ValueError("Support nodes must be ordered in consecutive class groups.")
        targets = torch.tensor([class_labels.index(y) for y in query_labels], device=embeddings.device)

        prototypes = embeddings[s].reshape(n_way, k_shot, -1).mean(1)
        distances = torch.cdist(embeddings[q].unsqueeze(0), prototypes.unsqueeze(0)).squeeze(0)
        prediction = distances.argmin(-1)
        scores.append(float((prediction == targets).float().mean()))
    scores = np.asarray(scores, dtype=np.float64)
    return {
        "accuracy": float(scores.mean()),
        "ci95_half_width": float(1.96 * scores.std(ddof=0) / np.sqrt(scores.size)),
        "num_episodes": int(scores.size),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/Amazon_clothing/unsup_NaQ/261009_112926_548403/best.pt")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dataset", default="Amazon_clothing")
    parser.add_argument("--latent", type=int, default=256)
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--k_shot_test", type=int, default=1)
    parser.add_argument("--q_query_test", type=int, default=8)
    parser.add_argument("--num_valid_tasks", type=int, default=50)
    parser.add_argument("--num_downstream_tasks", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1212)
    parser.add_argument("--json_path", default="exp024.json")
    args = parser.parse_args()
    args.setting = "unsup"
    args.query_generation = "NaQ"
    args.type = "feature"

    torch.set_num_threads(4)
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)
    task_path = f"save/downstream_tasks/{args.dataset}/{args.n_way}way_{args.k_shot_test}shot.pkl"
    if not os.path.isfile(task_path):
        raise FileNotFoundError(f"Missing fixed episode file: {task_path}. " "Evaluation stopped to avoid generating episodes different from exp023.")

    data = load_data(args)
    if device.type == "cuda":
        data.set_device(str(device))
    data.add_diffusion(args)
    config = [("GCN", (data.features.shape[1], 2 * args.latent), True), ("GCN", (2 * args.latent, args.latent), True)]
    encoder = GNNEncoder(config)
    checkpoint = load_encoder(encoder, args.checkpoint)
    encoder = encoder.to(device).eval()
    valid_tasks, test_tasks = downstream_task_generator(args, data, args.n_way, args.k_shot_test)

    with torch.no_grad():
        embeddings = encoder(data.features, adj=data.adj)
        valid = evaluate(embeddings, valid_tasks, args.n_way, args.k_shot_test)
        test = evaluate(embeddings, test_tasks, args.n_way, args.k_shot_test)

    result = {
        "experiment": "exp024",
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "selection_note": "Checkpoint selected by original NAQ protocol; this script performs eval only",
        "evaluation": "Euclidean ProtoNet (nearest class mean)",
        "task_file": task_path,
        "config": vars(args),
        "valid": valid,
        "test": test,
    }
    os.makedirs(f"results/{args.dataset}/exp024", exist_ok=True)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    with open(f"results/{args.dataset}/exp024/result.txt", "w", encoding="utf-8") as file:
        file.write(f"Valid {valid['accuracy']:.4f} ± {valid['ci95_half_width']:.4f} | Test {test['accuracy']:.4f} ± {test['ci95_half_width']:.4f}\n")
    with open(args.json_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
