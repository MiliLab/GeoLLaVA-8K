import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from longva.utils import rank0_print


# ------------------------------------------------------------
# 1. RegProxy Affinity Head (生成 Q)
# ------------------------------------------------------------
class RegProxyAffinityHead(nn.Module):
    """
    输入 : patch‑token feature grid  E  (B, H, W, D)
    输出 : Q  (B, H, W, 9)    —— 每 token 对 3×3 邻域 9 个种子 token 的 softmax 权重
    无新增可学习量以外的 映射 (只有 3×3 DWConv + 1×1 PWConv)
    """

    def __init__(self, in_dim: int = 1024, std_init=0.01):
        super().__init__()
        self.dw = nn.Conv2d(in_dim, in_dim, 3, 1, 1, groups=in_dim, bias=False)
        self.pw = nn.Conv2d(in_dim, 9, 1, bias=True)
        # self.cls_Wq = torch.nn.Linear(in_dim, in_dim, bias=False)
        # self.cls_Wk = torch.nn.Linear(in_dim, in_dim, bias=False)
        # 更小方差初始化
        # nn.init.normal_(self.dw.weight, 0, std_init)
        # nn.init.normal_(self.pw.weight, 0, std_init)
        nn.init.zeros_(self.pw.bias)

    def forward(self, tok2d: torch.Tensor):
        """
        tok2d : (B, H, W, D)
        return : Q (B, H, W, 9)  (softmax on last dim)
        """
        # B, H, W, D = tok2d.shape
        x = tok2d.permute(0, 3, 1, 2).contiguous()  # (B,D,H,W)
        x = self.dw(x)
        x = self.pw(x)  # (B,9,H,W)
        Q = F.softmax(x.permute(0, 2, 3, 1), dim=-1)  # (B,H,W,9)
        return Q

    def clustering(
        self,
        tok2d: torch.Tensor,
        q_proj: nn.Linear,
        v_proj: nn.Linear,
        max_iters: int = 6,
        K: int | None = None,
        H: int = 24,
        W: int = 24,
        MAX_TOKENS: int = 576,
    ):
        """
        直接返回 Q 的聚类结果
        """
        B, M, D = tok2d.shape
        assert M > 1, "输入至少要有 [CLS] + 一个 patch"
        N = M - 1
        assert N == H * W, "H*W 必须等于 M-1"
        # 1) 拆分 cls token & patch token
        cls_tok = tok2d[:, :1, :]  # (B,1,D)
        patch_feats = tok2d[:, 1:, :].contiguous()  # (B,N,D)

        # 2) 恢复成 2D 格式计算 Q→A
        tok2d = patch_feats.view(B, H, W, D)
        Q = self.forward(tok2d)  # (B,H,W,9)
        A = build_token_affinity(Q, H, W)  # (B,N,N)
        # 3) 并查集式聚类得到 roots
        roots = cluster_tokens_by_parent(A, max_iters=max_iters)
        # centers = compute_cluster_centers(E, roots)
        # 4) batch 化计算簇中心均值
        C_pruned, counts_pruned = compute_cluster_centers_batched(patch_feats, roots, K=K)
        # mask = counts_pruned > 0
        # Q_cls = self.cls_Wq(cls_tok)  # (B,1,D)
        # K_v = self.cls_Wk(C_pruned)  # (B,N,D)
        Q_cls = F.linear(cls_tok, q_proj.weight, q_proj.bias)  # (B,1,D)
        K_v = F.linear(C_pruned, v_proj.weight, v_proj.bias)  # (B,N,D)

        attn_scores = torch.bmm(Q_cls, K_v.transpose(1, 2)) / math.sqrt(D)  # (B,1,N)
        attn_scores = attn_scores.squeeze(1)  # (B,N)
        # print(type(C_pruned), type(mask))
        valid_scores = attn_scores.masked_fill(~(counts_pruned > 0), float("-inf"))
        MAX_TOKENS = min(MAX_TOKENS, C_pruned.size(1))
        # 6) top-k_sel，并按原空间顺序排序
        _, idx_topk = valid_scores.topk(MAX_TOKENS, dim=-1)  # (B,k_sel)
        idx_sorted, _ = idx_topk.sort(dim=-1)  # (B,k_sel)
        batch_idx = torch.arange(B, device=tok2d.device)[:, None]
        sel_patches = C_pruned[batch_idx, idx_sorted]  # (B,k_sel,D)
        # C_pruned = [C_pruned[b][mask[b]] for b in range(B)]
        # assert torch.allclose(*[torch.cat(cs) for cs in [centers, _centers]])
        # for b, c in enumerate(centers):
        #     print(f"Batch {b}: {c.shape[0]} clusters, centers tensor shape = {c.shape}")
        return sel_patches


# ------------------------------------------------------------
# 2. Q  →  token‑token affinity  A
# ------------------------------------------------------------
def build_token_affinity(Q: torch.Tensor, H: int = 24, W: int = 24):
    """
    Q : (B, H, W, 9)
    A : (B, N, N)  (dense)   N = H*W
    """
    B = Q.size(0)
    N = H * W
    device = Q.device
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    tok_idx = (y * W + x).flatten()

    rel = torch.tensor([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]], device=device)

    A = torch.zeros(B, N, N, device=device)

    for k in range(9):
        dy, dx = int(rel[k, 0]), int(rel[k, 1])
        # 取 token 中心像素的权重
        q_tok = Q[..., k]  # (B,H,W)
        ny = (y + dy).clamp(0, H - 1)
        nx = (x + dx).clamp(0, W - 1)
        neigh_idx = (ny * W + nx).flatten()

        A[:, tok_idx, neigh_idx] += q_tok.reshape(B, N)

    # 对称化并归一化到 [0,1]
    # A = 0.5 * (A + A.transpose(1, 2))
    # A = A / (A.amax(dim=-1, keepdim=True) + 1e-6)
    return A  # (B,N,N)

def cluster_tokens_by_parent(A, max_iters: int = 20):
    """
    快速基于“父指针”找根 —— 并查集 find 操作向量化版本。

    输入:
      A: Tensor of shape (B, N, N), A[b,i,j] = “i → j” 的概率（不需对称化）
    输出:
      roots: LongTensor of shape (B, N), roots[b,i] = i 所属的根节点索引
    """
    B, N, _ = A.shape
    # 1) 每行选父节点
    parent = A.argmax(dim=-1)  # (B, N), parent[b,i] = argmax_j A[b,i,j]
    # 2) 指针追踪（pointer jumping），直到收敛或到达 max_iters
    roots = parent
    for _ in range(max_iters):
        # p_next[b,i] = roots[b, roots[b,i]]
        p_next = roots.gather(1, roots)
        if torch.equal(p_next, roots):
            break
        roots = p_next
    return roots  # (B, N)


def compute_cluster_centers(E, roots):
    """
    根据 roots 分组并计算每簇的均值向量。

    输入:
      E    : Tensor of shape (B, N, D), 原始 token 特征
      roots: LongTensor of shape (B, N), 每个 token 的根索引
    输出:
      centers_list: list of B 个 Tensor，各自形状 (K_b, D),
                    K_b = roots[b].unique().numel()
    """
    B, N, D = E.shape
    centers_list = []
    for b in range(B):
        # 唯一根节点
        uniq = torch.unique(roots[b])
        # 簇均值
        centers_b = torch.stack([E[b, roots[b] == r].mean(0) for r in uniq], dim=0)
        centers_list.append(centers_b)  # (K_b, D)
    return centers_list


def compute_cluster_centers_batched(
    E: torch.Tensor,  # (B, N, D)
    roots: torch.Tensor,  # (B, N)
    K: int | None = None,  # 最多保留多少簇
):
    B, N, D = E.shape
    device, dtype = E.device, E.dtype

    # 1) 批量累加：每个根的特征和 & 计数
    centers_sum = torch.zeros(B, N, D, device=device, dtype=dtype)
    centers_sum.scatter_add_(1, roots.unsqueeze(-1).expand(-1, -1, D), E)
    counts = torch.zeros(B, N, device=device, dtype=dtype)
    counts.scatter_add_(1, roots, torch.ones_like(counts))
    # _counts = counts.detach()
    # _counts = _counts[_counts > 0]
    # rank0_print("counts", _counts.shape, "max", _counts.max().item(), "mean", _counts.mean().item())
    # 2) 按簇大小 top-K
    if K is not None and K <= N:
        _, topk_desc = counts.topk(K, dim=1)  # (B, K)
        # 3) 保持原 token 顺序再排序
        topk_ord, _ = topk_desc.sort(dim=1)  # (B, K)

        # 4) Gather 和
        centers_sum_pruned = centers_sum.gather(1, topk_ord.unsqueeze(-1).expand(-1, -1, D))  # (B, K, D)

        # 5) Gather 计数
        counts_pruned = counts.gather(1, topk_ord)  # (B, K)
    else:
        topk_ord = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        centers_sum_pruned = centers_sum
        counts_pruned = counts
    counts_clamped = counts_pruned.clamp_min(1.0).unsqueeze(-1)  # (B, K, 1)
    C_pruned = centers_sum_pruned / counts_clamped  # (B, K, D)
    # mask = counts_pruned > 0
    return C_pruned, counts_pruned


# ------------------------------------------------------------
# 4. Quick Self‑check
# ------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, W, D = 2, 4, 4, 768  # toy ViT‑L/14
    patch_tok = torch.randn(B, H, W, D, requires_grad=True)

    # Q
    head = RegProxyAffinityHead(in_dim=D)
    toks = torch.randn(B, H * W + 1, D, requires_grad=True)
    _centers = head.clustering(toks, max_iters=6, K=None, H=H, W=W, MAX_TOKENS=4)
    print("_centers.shape", _centers.shape)
    for b, c in enumerate(_centers):
        print(f"Batch {b}: {c.shape[0]} clusters, centers tensor shape = {c.shape}")
    # Q = head(patch_tok.detach())  # (B,H,W,9)

    # # A
    # A = build_token_affinity(Q, H, W)  # (B,576,576)

    # # # 池化
    # # E = patch_tok.reshape(B, H * W, D)  # (B,576,D)
    # # pooled, S = nonparam_token_pool(E, A, k_fix=4)

    # # print("Q      :", Q.shape)  # (2,24,24,9)
    # # print("A      :", A.shape)  # (2,576,576)
    # # print("pooled :", pooled.shape)  # (2,32,768)

    # # # 梯度检查
    # # out = pooled.mean()
    # # out.backward()
    # # print("grad OK:", patch_tok.grad.abs().sum() > 0)
    # # # True  → 梯度已回传至原 patch‑token

    # # pool with PageRank
    # E = patch_tok.reshape(B, H * W, D)

    # roots = cluster_tokens_by_parent(A, max_iters=6)
    # centers = compute_cluster_centers(E, roots)
    # _centers, mask = compute_cluster_centers_batched(E, roots, K=8)
    # assert torch.allclose(
    #     compute_cluster_centers_batched(E, roots, K=16)[0], compute_cluster_centers_batched(E, roots, K=None)[0]
    # )
    # mask = mask.squeeze()
    # _centers = [_centers[b][mask[b]] for b in range(B)]
    # assert torch.allclose(*[torch.cat(cs) for cs in [centers, _centers]])
    # for b, c in enumerate(centers):
    #     print(f"Batch {b}: {c.shape[0]} clusters, centers tensor shape = {c.shape}")
