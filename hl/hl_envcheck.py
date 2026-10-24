"""hl_envcheck: import/version check for running choi_exps/exp049.py. No installs."""
import sys
out = []
try:
    import torch, torch_geometric, numpy, scipy, sklearn, tqdm
    import torch_geometric.transforms as T
    import inspect
    out.append(f"python {sys.version.split()[0]}")
    out.append(f"torch {torch.__version__} cuda_build {torch.version.cuda} cuda_available {torch.cuda.is_available()}")
    out.append(f"pyg {torch_geometric.__version__} numpy {numpy.__version__} scipy {scipy.__version__} sklearn {sklearn.__version__}")
    assert "weights_only" in inspect.signature(torch.load).parameters, "torch>=1.13 required (weights_only)"
    assert torch.cuda.is_available(), "CUDA not available"
    T.GDC  # used by utils.graph_diffusion
    sparse = torch.sparse_coo_tensor(torch.tensor([[0, 1], [1, 0]]), torch.tensor([1.0, 1.0]), (2, 2)).coalesce()
    torch.sparse.mm(sparse, torch.ones(2, 3))  # used by exp048/049 mining
    out.append("OK")
except Exception as e:
    out.append(f"FAIL {type(e).__name__}: {e}")
print("\n".join(out))
