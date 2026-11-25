
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class PCAProjector:

    def __init__(self, proj_dim: int = 128):
        self.proj_dim = proj_dim
        self.fitted = False
        self.mean_s = None
        self.W_s = None
        self.mean_v = None
        self.W_v = None
        self.mean_t = None
        self.W_t = None
        self.mean_y = None
        self.W_y = None

    def fit(self, S_all: torch.Tensor, V_all: torch.Tensor, T_all: torch.Tensor, Y_all: torch.Tensor):
        def _fit_one(X, name):
            # X: [N, D]
            N, D = X.shape
            mu = X.mean(dim=0)
            X_c = X - mu

            # SVD: X_c = U @ S @ V^T
            U, S, Vt = torch.linalg.svd(X_c, full_matrices=False)

            k = min(self.proj_dim, D, S.size(0))
            Vt_k = Vt[:k]  # [k, D]
            S_k = S[:k]    # [k]

            S_k_clamped = S_k.clamp(min=1e-5)
            W = Vt_k / S_k_clamped.unsqueeze(1)  # [k, D]

            var_ratio = (S_k**2).sum() / (S**2).sum()
            logger.info(f"PCA {name}: {k} components, explained variance = {var_ratio:.4f}")

            return mu, W

        self.mean_s, self.W_s = _fit_one(S_all, "S[2H]")
        self.mean_v, self.W_v = _fit_one(V_all, "V[H]")
        self.mean_t, self.W_t = _fit_one(T_all, "T[H]")
        self.mean_y, self.W_y = _fit_one(Y_all, "Y[H]")
        self.fitted = True
        logger.info(f"PCAProjector fitted: proj_dim={self.proj_dim}")

    def to(self, device):
        self.mean_s = self.mean_s.to(device)
        self.W_s = self.W_s.to(device)
        self.mean_v = self.mean_v.to(device)
        self.W_v = self.W_v.to(device)
        self.mean_t = self.mean_t.to(device)
        self.W_t = self.W_t.to(device)
        self.mean_y = self.mean_y.to(device)
        self.W_y = self.W_y.to(device)
        return self

    def __call__(self, s: torch.Tensor, v: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        assert self.fitted, "Call fit() before projection"
        s_p = (s.float() - self.mean_s) @ self.W_s.t()
        v_p = (v.float() - self.mean_v) @ self.W_v.t()
        t_p = (t.float() - self.mean_t) @ self.W_t.t()
        y_p = (y.float() - self.mean_y) @ self.W_y.t()
        return s_p, v_p, t_p, y_p

    def save(self, path: str):
        torch.save({
            'proj_dim': self.proj_dim,
            'mean_s': self.mean_s,
            'W_s': self.W_s,
            'mean_v': self.mean_v,
            'W_v': self.W_v,
            'mean_t': self.mean_t,
            'W_t': self.W_t,
            'mean_y': self.mean_y,
            'W_y': self.W_y,
        }, path)
        logger.info(f"PCAProjector saved to: {path}")

    @classmethod
    def load(cls, path: str):
        ckpt = torch.load(path, map_location='cpu')
        projector = cls(proj_dim=ckpt['proj_dim'])
        projector.mean_s = ckpt['mean_s']
        projector.W_s = ckpt['W_s']
        projector.mean_v = ckpt['mean_v']
        projector.W_v = ckpt['W_v']
        projector.mean_t = ckpt['mean_t']
        projector.W_t = ckpt['W_t']
        projector.mean_y = ckpt['mean_y']
        projector.W_y = ckpt['W_y']
        projector.fitted = True
        logger.info(f"PCAProjector loaded: proj_dim={projector.proj_dim}")
        return projector


class PIDProjector(nn.Module):

    def __init__(self, hidden_size: int, proj_dim: int = 256):
        super().__init__()
        self.proj_s = nn.Linear(hidden_size * 2, proj_dim, bias=False)
        self.proj_v = nn.Linear(hidden_size, proj_dim, bias=False)
        self.proj_t = nn.Linear(hidden_size, proj_dim, bias=False)
        self.proj_y = nn.Linear(hidden_size, proj_dim, bias=False)
        self.proj_dim = proj_dim

    def forward(self, s, v, t, y):
        return self.proj_s(s), self.proj_v(v), self.proj_t(t), self.proj_y(y)


def cov_xy(Y: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    Yc = Y - Y.mean(dim=0, keepdim=True)
    Xc = X - X.mean(dim=0, keepdim=True)
    return Yc.t() @ Xc / (Y.size(0) - 1)


class GaussianPIDScorer:

    def __init__(self, proj_dim: int = 256, reg_scale: float = 1e-4):
        self.proj_dim = proj_dim
        self.reg_scale = reg_scale
        self.count = 0
        self._buf_S = []
        self._buf_V = []
        self._buf_T = []
        self._buf_Y = []
        self.finalized = False

    def update(self, S: torch.Tensor, V: torch.Tensor, T: torch.Tensor, Y: torch.Tensor):
        self._buf_S.append(S.detach().cpu())
        self._buf_V.append(V.detach().cpu())
        self._buf_T.append(T.detach().cpu())
        self._buf_Y.append(Y.detach().cpu())
        self.count += S.size(0)

    def finalize(self):
        assert self.count > 1, f"At least 2 samples are required, got {self.count}"

        S_all = torch.cat(self._buf_S, dim=0)
        V_all = torch.cat(self._buf_V, dim=0)
        T_all = torch.cat(self._buf_T, dim=0)
        Y_all = torch.cat(self._buf_Y, dim=0)
        del self._buf_S, self._buf_V, self._buf_T, self._buf_Y

        X_map = {
            'S': S_all, 'V': V_all, 'T': T_all, 'Y': Y_all,
            'SV': torch.cat([S_all, V_all], dim=-1),
            'ST': torch.cat([S_all, T_all], dim=-1),
            'SVT': torch.cat([S_all, V_all, T_all], dim=-1),
        }

        self.cov = {}
        for k in ['S', 'V', 'T', 'Y', 'SV', 'ST', 'SVT']:
            C = cov_xy(X_map[k], X_map[k])
            eps = self.reg_scale * C.trace() / C.size(0)
            C += eps * torch.eye(C.size(0))
            self.cov[k] = C

        self.cross_cov = {}
        for cross_key, x_key in [('Y_S', 'S'), ('Y_SV', 'SV'), ('Y_ST', 'ST'), ('Y_SVT', 'SVT')]:
            self.cross_cov[cross_key] = cov_xy(X_map['Y'], X_map[x_key])

        self.mean = {k: X_map[k].mean(dim=0) for k in X_map}

        self._pmi_cache = {}
        for cross_key, x_key in [('Y_S', 'S'), ('Y_SV', 'SV'), ('Y_ST', 'ST'), ('Y_SVT', 'SVT')]:
            Sigma_YX = self.cross_cov[cross_key]
            Sigma_X = self.cov[x_key]
            W = torch.linalg.solve(Sigma_X, Sigma_YX.t()).t()
            Sigma_Y_given_X = self.cov['Y'] - Sigma_YX @ torch.linalg.solve(Sigma_X, Sigma_YX.t())
            eps = self.reg_scale * Sigma_Y_given_X.trace().abs() / Sigma_Y_given_X.size(0)
            Sigma_Y_given_X += eps * torch.eye(Sigma_Y_given_X.size(0))
            P = torch.linalg.inv(Sigma_Y_given_X)
            self._pmi_cache[cross_key] = {'W': W, 'P': P, 'x_key': x_key}

        global_raw = (
            self._compute_mi('Y', 'SVT', 'Y_SVT') - self._compute_mi('Y', 'ST', 'Y_ST')
            - self._compute_mi('Y', 'SV', 'Y_SV') + self._compute_mi('Y', 'S', 'Y_S')
        )
        self._global_synergy_raw = global_raw
        self._global_synergy = max(0.0, global_raw)

        self.finalized = True
        logger.info(
            f"GaussianPIDScorer finalized: {self.count} samples, proj_dim={self.proj_dim}, "
            f"global_synergy={self._global_synergy:.4f} (raw={self._global_synergy_raw:.4f})"
        )

    def to(self, device):
        self.mean = {k: v.to(device) for k, v in self.mean.items()}
        for cache in self._pmi_cache.values():
            cache['W'] = cache['W'].to(device)
            cache['P'] = cache['P'].to(device)
        return self

    def score(self, S: torch.Tensor, V: torch.Tensor, T: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        assert self.finalized, "Call finalize() before scoring"
        S, V, T, Y = S.detach().float(), V.detach().float(), T.detach().float(), Y.detach().float()

        X_map = {
            'Y_SVT': torch.cat([S, V, T], dim=-1),
            'Y_SV': torch.cat([S, V], dim=-1),
            'Y_ST': torch.cat([S, T], dim=-1),
            'Y_S': S,
        }

        r_y = Y - self.mean['Y'].unsqueeze(0)

        def quad_cond(cross_key):
            cache = self._pmi_cache[cross_key]
            W, P, x_key = cache['W'], cache['P'], cache['x_key']
            dx = X_map[cross_key] - self.mean[x_key].unsqueeze(0)
            r_cond = r_y - dx @ W.t()
            return (r_cond @ P * r_cond).sum(dim=-1)

        q_full = quad_cond('Y_SVT')
        q_no_v = quad_cond('Y_ST')
        q_no_t = quad_cond('Y_SV')
        q_only_s = quad_cond('Y_S')

        scores = 0.5 * (-q_full + q_no_v + q_no_t - q_only_s) + self._global_synergy
        return scores.clamp(min=0.0)

    def _compute_mi(self, y_key: str, x_key: str, cross_key: str) -> float:
        """I_G(Y; X) = 0.5 * log(det(Σ_Y) / det(Σ_{Y|X}))"""
        Sigma_Y = self.cov[y_key]
        Sigma_X = self.cov[x_key]
        Sigma_YX = self.cross_cov[cross_key]

        # Σ_{Y|X} = Σ_Y - Σ_YX @ Σ_X^{-1} @ Σ_XY
        Sigma_XY = Sigma_YX.t()
        solved = torch.linalg.solve(Sigma_X, Sigma_XY)  # Σ_X^{-1} @ Σ_XY
        Sigma_Y_given_X = Sigma_Y - Sigma_YX @ solved

        eps = self.reg_scale * Sigma_Y_given_X.trace() / Sigma_Y_given_X.size(0)
        Sigma_Y_given_X += eps.abs() * torch.eye(Sigma_Y_given_X.size(0))

        log_det_Y = self._log_det_cholesky(Sigma_Y)
        log_det_Y_given_X = self._log_det_cholesky(Sigma_Y_given_X)

        mi = 0.5 * (log_det_Y - log_det_Y_given_X)
        return max(0.0, mi.item())

    @staticmethod
    def _log_det_cholesky(M: torch.Tensor) -> torch.Tensor:
        """log det via Cholesky: log det(M) = 2 * sum(log(diag(L)))"""
        L = torch.linalg.cholesky(M)
        return 2.0 * L.diagonal().log().sum()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        scorer = cls(proj_dim=ckpt['proj_dim'], reg_scale=ckpt['reg_scale'])
        scorer.count = ckpt['count']
        scorer.mean = ckpt['mean']
        scorer.cov = ckpt['cov']
        scorer.cross_cov = ckpt['cross_cov']
        scorer._pmi_cache = ckpt['_pmi_cache']
        scorer._global_synergy = ckpt['_global_synergy']
        scorer._global_synergy_raw = ckpt.get('_global_synergy_raw', ckpt['_global_synergy'])
        scorer.finalized = True
        logger.info(f"GaussianPIDScorer loaded: {scorer.count} samples, proj_dim={scorer.proj_dim}, global_synergy={scorer._global_synergy:.4f}")
        return scorer


class PIDScorerWrapper:

    def __init__(self, projector: PCAProjector, scorer: GaussianPIDScorer, device: torch.device):
        self.projector = projector
        self.scorer = scorer
        self.device = device

    @classmethod
    def from_checkpoint(cls, projector_path: str, scorer_path: str, device=None, **kwargs):
        device = device or torch.device('cpu')
        projector = PCAProjector.load(projector_path)
        scorer = GaussianPIDScorer.from_checkpoint(scorer_path)
        projector.to(device)
        scorer.to(device)
        return cls(projector=projector, scorer=scorer, device=device)

    @staticmethod
    def _masked_mean_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1).to(x.dtype)
        return (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

    @torch.no_grad()
    def compute_reward(self, dis_out: Dict[str, torch.Tensor], question_hidden: torch.Tensor, question_mask: torch.Tensor) -> torch.Tensor:
        z_si = dis_out['v_shared'].mean(dim=1)
        z_st = dis_out['t_shared'].mean(dim=1)
        z_vp = self._masked_mean_pool(dis_out['v_priv'], dis_out['v_mask'])
        z_tp = self._masked_mean_pool(dis_out['t_priv'], dis_out['t_mask'])

        S = torch.cat([z_si, z_st], dim=-1)
        V, T = z_vp, z_tp
        Y = self._masked_mean_pool(question_hidden, question_mask)

        S_p, V_p, T_p, Y_p = self.projector(S, V, T, Y)
        scores = self.scorer.score(S_p, V_p, T_p, Y_p)
        return scores.to(self.device)


class vMFPIDScorer:

    def __init__(self, reg_scale: float = 1e-4):
        self.reg_scale = reg_scale
        self.count = 0
        self._buf_S = []
        self._buf_V = []
        self._buf_T = []
        self._buf_Y = []
        self.finalized = False

    def update(self, S: torch.Tensor, V: torch.Tensor, T: torch.Tensor, Y: torch.Tensor):
        self._buf_S.append(S.detach().cpu())
        self._buf_V.append(V.detach().cpu())
        self._buf_T.append(T.detach().cpu())
        self._buf_Y.append(Y.detach().cpu())
        self.count += S.size(0)

    def finalize(self):
        assert self.count > 1, f"At least 2 samples are required, got {self.count}"

        S_all = torch.cat(self._buf_S, dim=0)
        V_all = torch.cat(self._buf_V, dim=0)
        T_all = torch.cat(self._buf_T, dim=0)
        Y_all = torch.cat(self._buf_Y, dim=0)
        del self._buf_S, self._buf_V, self._buf_T, self._buf_Y

        Y_norm = F.normalize(Y_all, dim=-1)

        X_map = {
            'S': S_all,
            'SV': torch.cat([S_all, V_all], dim=-1),
            'ST': torch.cat([S_all, T_all], dim=-1),
            'SVT': torch.cat([S_all, V_all, T_all], dim=-1),
        }

        self._regressors = {}
        for key, X in X_map.items():
            mean_x = X.mean(dim=0)
            Xc = X - mean_x
            XtX = Xc.t() @ Xc
            reg = self.reg_scale * XtX.trace() / XtX.size(0)
            XtX += reg * torch.eye(XtX.size(0))
            W = torch.linalg.solve(XtX, Xc.t() @ Y_norm)  # [d_x, d_y]
            self._regressors[key] = {'W': W, 'mean_x': mean_x}

        self.finalized = True
        logger.info(f"vMFPIDScorer finalized: {self.count} samples, reg_scale={self.reg_scale}")

    def to(self, device):
        for cache in self._regressors.values():
            cache['W'] = cache['W'].to(device)
            cache['mean_x'] = cache['mean_x'].to(device)
        return self

    def score(self, S: torch.Tensor, V: torch.Tensor, T: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        assert self.finalized, "Call finalize() before scoring"
        S, V, T, Y = S.detach().float(), V.detach().float(), T.detach().float(), Y.detach().float()
        y = F.normalize(Y, dim=-1)

        X_map = {
            'S': S,
            'SV': torch.cat([S, V], dim=-1),
            'ST': torch.cat([S, T], dim=-1),
            'SVT': torch.cat([S, V, T], dim=-1),
        }

        dots = {}
        for key, X in X_map.items():
            reg = self._regressors[key]
            xc = X - reg['mean_x'].unsqueeze(0)
            mu = F.normalize(xc @ reg['W'], dim=-1)
            dots[key] = (mu * y).sum(dim=-1)

        scores = dots['SVT'] - dots['SV'] - dots['ST'] + dots['S']
        return scores.clamp(min=0.0)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        scorer = cls(reg_scale=ckpt['reg_scale'])
        scorer.count = ckpt['count']
        scorer._regressors = ckpt['_regressors']
        scorer.finalized = True
        logger.info(f"vMFPIDScorer loaded: {scorer.count} samples")
        return scorer


def _masked_mean_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.unsqueeze(-1).to(x.dtype)
    return (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)


class vMFScorerWrapper:

    def __init__(self, scorer: vMFPIDScorer, device: torch.device):
        self.scorer = scorer
        self.device = device

    @classmethod
    def from_checkpoint(cls, scorer_path: str, device=None, **kwargs):
        device = device or torch.device('cpu')
        scorer = vMFPIDScorer.from_checkpoint(scorer_path)
        scorer.to(device)
        return cls(scorer=scorer, device=device)

    @torch.no_grad()
    def compute_reward(self, dis_out: Dict[str, torch.Tensor], question_hidden: torch.Tensor, question_mask: torch.Tensor) -> torch.Tensor:
        z_si = dis_out['v_shared'].mean(dim=1)
        z_st = dis_out['t_shared'].mean(dim=1)
        z_vp = _masked_mean_pool(dis_out['v_priv'], dis_out['v_mask'])
        z_tp = _masked_mean_pool(dis_out['t_priv'], dis_out['t_mask'])
        S = torch.cat([z_si, z_st], dim=-1)
        Y = _masked_mean_pool(question_hidden, question_mask)
        scores = self.scorer.score(S, z_vp, z_tp, Y)
        return scores.to(self.device)


class PIDRewardModel(nn.Module):

    def __init__(
        self,
        input_dim: int = 17920,
        hidden_dims: list = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [2048, 1024, 512]

        layers = []
        in_d = input_dim
        for h_d in hidden_dims:
            layers.extend([
                nn.Linear(in_d, h_d),
                nn.LayerNorm(h_d),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_d = h_d
        layers.append(nn.Linear(in_d, 1))

        self.mlp = nn.Sequential(*layers)

        logger.info(
            f"PIDRewardModel: input_dim={input_dim}, "
            f"hidden_dims={hidden_dims}, dropout={dropout}"
        )

    def forward(self, s: torch.Tensor, v: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, v, t, y], dim=-1)
        return self.mlp(x).squeeze(-1)


class PIDRewardWrapper:

    def __init__(
        self,
        reward_model: PIDRewardModel,
        device: torch.device = None,
    ):
        self.reward_model = reward_model
        self.reward_model.eval()
        self.device = device or next(reward_model.parameters()).device

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: torch.device = None):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        config = ckpt['config']

        reward_model = PIDRewardModel(
            input_dim=config['input_dim'],
            hidden_dims=config['hidden_dims'],
            dropout=config.get('dropout', 0.1),
        )
        reward_model.load_state_dict(ckpt['reward_model'])

        if device is not None:
            reward_model = reward_model.to(device)

        return cls(reward_model=reward_model, device=device)

    @staticmethod
    def _masked_mean_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1).to(x.dtype)
        return (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

    @torch.no_grad()
    def compute_reward(
        self,
        dis_out: Dict[str, torch.Tensor],
        question_hidden: torch.Tensor,
        question_mask: torch.Tensor,
    ) -> torch.Tensor:
        z_si = dis_out['v_shared'].mean(dim=1)   # [B, H]
        z_st = dis_out['t_shared'].mean(dim=1)   # [B, H]
        z_vp = self._masked_mean_pool(dis_out['v_priv'], dis_out['v_mask'])  # [B, H]
        z_tp = self._masked_mean_pool(dis_out['t_priv'], dis_out['t_mask'])  # [B, H]

        S = torch.cat([z_si, z_st], dim=-1)  # [B, 2H]
        V = z_vp                              # [B, H]
        T = z_tp                              # [B, H]
        Y = self._masked_mean_pool(question_hidden, question_mask)  # [B, H]

        return self.reward_model(S, V, T, Y)
