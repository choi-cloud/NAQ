"""hl_setup_check: HL03 §0-2 fixed-episode self-check and §0-3 short tests 2-4 (data load, episodes, checkpoints). No labels, no training."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hl_common as H  # noqa: E402
from exp026 import load_fixed_episodes  # noqa: E402

SEEDS = (1212, 1213, 1214, 1215, 1216)
CKPTS = (
    "results/hl/p0/Amazon_clothing/ours/tau_0.3/seed_1212/best_1shot.pt",
    "results/hl/p0/Amazon_clothing/naq/baseline/seed_1212/best_1shot.pt",
)


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("episodes", "all"):
        n = 0
        for seed in SEEDS:
            for shot in (1, 5):
                folder = Path("save/fixed_episodes/Amazon_clothing") / f"5way_{shot}shot_8query_seed{seed}"
                for split, count in (("valid", 50), ("test", 500)):
                    eps = load_fixed_episodes(folder / f"{split}_episodes.json", dataset="Amazon_clothing", n_way=5, k_shot=shot, q_query=8)
                    assert len(eps) == count, (folder, split, len(eps))
                    n += 1
        print(f"[0-2] fixed episodes sha256 self-check passed: {n} files")
    if stage in ("data", "all"):
        data, device = H.load_everything(H.base_args())
        print(f"[test2] device={device} features={tuple(data.features.shape)} Diffusion_matrix={tuple(data.Diffusion_matrix.shape)}")
    if stage in ("load", "all"):
        folder = Path("save/fixed_episodes/Amazon_clothing/5way_1shot_8query_seed1212")
        v = load_fixed_episodes(folder / "valid_episodes.json", dataset="Amazon_clothing", n_way=5, k_shot=1, q_query=8)
        t = load_fixed_episodes(folder / "test_episodes.json", dataset="Amazon_clothing", n_way=5, k_shot=1, q_query=8)
        print(f"[test3] seed1212 1-shot valid={len(v)} test={len(t)}")
    if stage in ("ckpt", "all"):
        for path in CKPTS:
            state = torch.load(path, map_location="cpu", weights_only=True)
            print(f"[test4] {path} epoch={state['epoch']} model_keys={len(state['model'])}")


if __name__ == "__main__":
    main()
