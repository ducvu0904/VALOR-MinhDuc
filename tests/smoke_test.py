"""Smoke test for VALOR implementation."""
import sys
sys.path.insert(0, '.')

import torch
import numpy as np

def test_imports():
    print('Testing imports...')
    from utils.ziln_utils import ziln_expected_value, ziln_uplift
    from models.losses import ZILNLoss, FocalZILNLoss, ValueWeightedRankingLoss, VALORLoss
    from models.gating import TreatmentGatedInteraction
    from models.srm_heads import SRMHead
    from models.baselines import TARNet, DragonNet, CFR, UniTE, EUEN, TLearner, SLearner
    from models.valor_net import VALOR
    from models.ziln_gbdt import ZILNGBDTForest
    print('All imports OK')

def test_losses():
    print('\n--- Unit testing losses ---')
    from models.losses import ZILNLoss, FocalZILNLoss, ValueWeightedRankingLoss, VALORLoss
    
    B = 64
    pi_logit = torch.randn(B)
    mu = torch.randn(B)
    sigma = torch.ones(B) * 0.5
    y = torch.clamp(torch.randn(B).abs() * 10, min=0.0)
    mask = torch.rand(B) < 0.8
    y[mask] = 0.0

    ziln = ZILNLoss()
    l = ziln(pi_logit, mu, sigma, y)
    print(f'ZILNLoss: {l.item():.4f}')

    focal = FocalZILNLoss(gamma=2.0, alpha=0.25)
    l = focal(pi_logit, mu, sigma, y)
    print(f'FocalZILNLoss: {l.item():.4f}')

    tau_hat = torch.randn(B, requires_grad=True)
    z = torch.randn(B).abs()
    rank_loss = ValueWeightedRankingLoss()
    l = rank_loss(tau_hat, z)
    print(f'ValueWeightedRankingLoss: {l.item():.4f}')
    l.backward()
    print('Ranking loss gradient flow OK')

    print('Loss tests PASSED')

def test_ziln_math():
    print('\n--- Testing ZILN math ---')
    from utils.ziln_utils import ziln_expected_value
    pi = torch.tensor([0.5])
    mu = torch.tensor([1.0])
    sig = torch.tensor([0.5])
    ev = ziln_expected_value(pi, mu, sig)
    expected = 0.5 * np.exp(1.0 + 0.5**2/2)
    print(f'ziln_expected_value: {ev.item():.6f} (expected: {expected:.6f})')
    assert abs(ev.item() - expected) < 1e-4, "ZILN math mismatch!"
    print('ZILN math PASSED')

def test_models():
    print('\n--- Testing model forward passes ---')
    from models.baselines import TARNet, DragonNet, CFR, UniTE, EUEN, TLearner, SLearner
    from models.valor_net import VALOR
    
    cate_dims = [4, 4, 4, 6]
    num_count = 200
    B = 32
    x_cat = torch.randint(0, 3, (B, 4))
    x_num = torch.randn(B, num_count)
    t = (torch.rand(B) > 0.5).float()
    
    # TARNet (scalar)
    m = TARNet(cate_dims, num_count, use_ziln=False)
    y0, y1 = m(x_cat, x_num)
    assert y0.shape == (B,), f"TARNet y0 shape {y0.shape}"
    uplift = m.predict_uplift(x_cat, x_num)
    assert uplift.shape == (B,), f"TARNet uplift shape {uplift.shape}"
    print(f'TARNet (scalar): y0={y0.shape}, uplift={uplift.shape} OK')
    
    # TARNet (ZILN)
    m = TARNet(cate_dims, num_count, use_ziln=True)
    y0, y1 = m(x_cat, x_num)
    pi0, mu0, sig0 = y0
    assert pi0.shape == (B,), f"TARNet ZILN pi0 shape {pi0.shape}"
    uplift = m.predict_uplift(x_cat, x_num)
    assert uplift.shape == (B,), f"TARNet ZILN uplift shape {uplift.shape}"
    print(f'TARNet (ZILN): pi={pi0.shape}, uplift={uplift.shape} OK')
    
    # DragonNet
    m = DragonNet(cate_dims, num_count, use_ziln=True)
    y0, y1, prop = m(x_cat, x_num)
    print(f'DragonNet: prop={prop.shape} OK')
    
    # CFR
    m = CFR(cate_dims, num_count, use_ziln=True, mode='wass')
    y0, y1, ipm = m(x_cat, x_num, t)
    print(f'CFR-WASS: ipm={ipm.item():.4f} OK')
    
    m = CFR(cate_dims, num_count, use_ziln=True, mode='mmd')
    y0, y1, ipm = m(x_cat, x_num, t)
    print(f'CFR-MMD: ipm={ipm.item():.4f} OK')
    
    # UniTE
    m = UniTE(cate_dims, num_count, use_ziln=True)
    mu_prog, tau = m(x_cat, x_num)
    uplift = m.predict_uplift(x_cat, x_num)
    print(f'UniTE: uplift={uplift.shape} OK')
    
    # EUEN
    m = EUEN(cate_dims, num_count, use_ziln=True)
    ctrl, upl = m(x_cat, x_num)
    uplift = m.predict_uplift(x_cat, x_num)
    print(f'EUEN: uplift={uplift.shape} OK')
    
    # T-Learner
    m = TLearner(cate_dims, num_count, use_ziln=True)
    y0, y1 = m(x_cat, x_num)
    uplift = m.predict_uplift(x_cat, x_num)
    print(f'TLearner: uplift={uplift.shape} OK')
    
    # S-Learner
    m = SLearner(cate_dims, num_count)
    out = m(x_cat, x_num, t)
    uplift = m.predict_uplift(x_cat, x_num)
    print(f'SLearner: uplift={uplift.shape} OK')
    
    # VALOR wrapper
    backbone = TARNet(cate_dims, num_count, use_ziln=True)
    valor = VALOR(backbone, use_gating=True)
    y0p, y1p, extras = valor(x_cat, x_num, t)
    pi0, mu0, sig0 = y0p
    assert pi0.shape == (B,)
    uplift = valor.predict_uplift(x_cat, x_num)
    assert uplift.shape == (B,)
    print(f'VALOR(TARNet): uplift={uplift.shape} OK')
    
    # VALOR with CFR backbone
    backbone = CFR(cate_dims, num_count, use_ziln=True, mode='wass')
    valor = VALOR(backbone, use_gating=True)
    y0p, y1p, extras = valor(x_cat, x_num, t)
    assert 'ipm_loss' in extras
    print(f'VALOR(CFR-WASS): ipm_loss={extras["ipm_loss"].item():.4f} OK')
    
    print('All model tests PASSED')

def test_gbdt():
    print('\n--- Testing ZILN-GBDT ---')
    from models.ziln_gbdt import ZILNGBDTForest
    
    N = 500
    d = 10
    X = np.random.randn(N, d)
    t = np.random.binomial(1, 0.5, N).astype(float)
    y = np.maximum(0, np.random.randn(N) * 5)
    y[np.random.rand(N) < 0.8] = 0.0
    
    forest = ZILNGBDTForest(n_estimators=3, max_depth=3, random_state=42)
    forest.fit(X, t, y)
    preds = forest.predict_uplift(X)
    assert preds.shape == (N,), f"GBDT preds shape {preds.shape}"
    print(f'ZILN-GBDT: preds range [{preds.min():.4f}, {preds.max():.4f}]')
    print('GBDT test PASSED')

if __name__ == '__main__':
    test_imports()
    test_losses()
    test_ziln_math()
    test_models()
    test_gbdt()
    print('\n' + '='*50)
    print('  ALL SMOKE TESTS PASSED')
    print('='*50)
