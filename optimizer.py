"""
GeomGAP Optimizer

A custom PyTorch‑based optimizer that operates on geometric prime polynomial logic.

Mathematical Foundation:
- η_k = a · r^k + b  (GAP formula)
- Gradient scaling with hyperbolic curvature effect
- Geometric damping (amortizer) mechanism
"""

import torch
from torch.optim.optimizer import Optimizer
from typing import Any, Dict, List, Optional, Callable
import math


class GeomGAPOptimizer(Optimizer):
    """
    Geometric Prime Polynomial Based Optimizer

    Prevents gradient explosion, dynamically updates the learning rate
    with a geometric function.

    Args:
        params: Parameters to optimize
        a: Initial coefficient (default: 0.0005)
        b: Bias / base rate (default: 1e-5)
        r: Geometric multiplier (default: 1.0005)
        curvature_factor: Hyperbolic curvature coefficient (default: 0.1)
        grad_threshold: Gradient damping threshold (default: 10.0)
        max_grad_norm: Maximum gradient norm (default: 1.0)
        eps: Epsilon for numerical stability (default: 1e-8)
        weight_decay: Weight decay (default: 0)
        betas: Coefficients for first and second moment (default: (0.9, 0.999))
        use_first_moment: Enable first moment (momentum) like Adam (default: True)
        gradient_centralization: Center gradients by subtracting mean (default: True)
        element_wise_clip: Element-wise gradient clipping threshold (default: 1.0)

    Example:
        >>> model = torch.nn.Linear(10, 2)
        >>> optimizer = GeomGAPOptimizer(
        ...     model.parameters(),
        ...     a=0.0005,
        ...     b=1e-5,
        ...     r=1.0005,
        ...     curvature_factor=0.1
        ... )
    """
    
    def __init__(
        self,
        params,
        a: float = 0.0005,
        b: float = 1e-5,
        r: float = 1.0005,
        curvature_factor: float = 0.1,
        grad_threshold: float = 10.0,
        max_grad_norm: float = 1.0,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        betas: tuple = (0.9, 0.999),
        use_first_moment: bool = True,
        gradient_centralization: bool = True,
        element_wise_clip: float = 1.0
    ):
        if a < 0:
            raise ValueError(f"a (başlangıç katsayısı) negatif olamaz: {a}")
        if b < 0:
            raise ValueError(f"b (bias) negatif olamaz: {b}")
        if r <= 0:
            raise ValueError(f"r (geometrik çarpan) pozitif olmalı: {r}")
        if not 1.0 <= r <= 1.1:
            import warnings
            warnings.warn(f"r={r} önerilen aralık dışında (1.0-1.1). "
                         "Gradyan patlaması riski yüksek olabilir.")
        
        defaults = dict(
            a=a,
            b=b,
            r=r,
            curvature_factor=curvature_factor,
            grad_threshold=grad_threshold,
            max_grad_norm=max_grad_norm,
            eps=eps,
            weight_decay=weight_decay,
            betas=betas,
            use_first_moment=use_first_moment,
            gradient_centralization=gradient_centralization,
            element_wise_clip=element_wise_clip
        )
        super(GeomGAPOptimizer, self).__init__(params, defaults)
        
        # Global step counter (her parametre grubu için ayrı k değeri)
        self.global_step = 0
    
    def _gap_learning_rate(self, k: int, a: float, b: float, r: float) -> float:
        """
        GAP Formula: η_k = a · r^(k/1000) + b
        
        Exponent normalization: k/1000.0 ensures smoother geometric growth
        over thousands of steps.

        Args:
            k: Step count
            a: Initial coefficient
            b: Bias
            r: Geometric multiplier

        Returns:
            Learning coefficient (capped at max_lr = a * 10)
        """
        lr = a * (r ** (k / 1000.0)) + b
        max_lr = a * 10
        return min(lr, max_lr)
    
    def _hyperbolic_curvature(
        self,
        grad: torch.Tensor,
        curvature_factor: float,
        step: int
    ) -> torch.Tensor:
        """
        Apply hyperbolic curvature effect.

        Instead of standard Euclidean distance, scale gradients
        to create a hyperbolic curvature.

        Mathematical formula:
        g_scaled = g / (1 + c · ||g||^2)^(1/2)

        Args:
            grad: Gradient tensor
            curvature_factor: Curvature coefficient (c)
            step: Current step

        Returns:
            Scaled gradient
        """
        grad_norm = torch.norm(grad)
        
        # Kavis dinamik olarak adımla birlikte azalır
        # 0.01 * step ile kavis etkisi zamanla daha belirgin azalır
        adaptive_curvature = curvature_factor / (1 + 0.01 * step)
        
        # Hiperbolik ölçeklendirme
        denominator = torch.sqrt(1.0 + adaptive_curvature * grad_norm ** 2)
        scaled_grad = grad / denominator.clamp(min=1e-8)
        
        return scaled_grad
    
    def _geometric_damping(
        self,
        grad_norm: float,
        threshold: float,
        step: int,
        r: float
    ) -> float:
        """
        Geometrik amortisör etkisi.
        
        Gradyan normu eşiği aştığında, lineer değil geometrik
        olarak sönümleme uygula.
        
        Args:
            grad_norm: Gradyan normu
            threshold: Sönümleme eşiği
            step: Mevcut adım
            r: Geometrik çarpan
            
        Returns:
            Sönümleme katsayısı (0-1 arası)
        """
        if grad_norm <= threshold:
            return 1.0
        
        # Fazla büyüklük (excess)
        excess = (grad_norm - threshold) / threshold
        
        # Geometrik sönümleme: α artar, damping azalır
        # α = 0.05 · r^(step/1000)
        alpha = 0.05 * (r ** (step / 1000.0))
        
        # Exponential decay: e^(-α · excess)
        damping = math.exp(-alpha * excess)
        
        # Minimum %1 güncelleme garantisi
        return max(damping, 0.01)
    
    def _safe_geometric_clamp(
        self,
        param: torch.Tensor,
        update: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Güvenli geometrik sınırlama.
        
        Ağırlıkların NaN veya sonsuz olmasını engelleyen
        güvenlik mekanizması.
        
        Args:
            param: Mevcut parametre değeri
            update: Uygulanacak güncelleme
            eps: Sayısal stabilite için epsilon
            
        Returns:
            Güvenli güncelleme
        """
        # NaN/Inf kontrolü
        if torch.isnan(update).any() or torch.isinf(update).any():
            # Tehlikeli güncellemeyi sıfırla
            safe_update = torch.where(
                torch.isnan(update) | torch.isinf(update),
                torch.zeros_like(update),
                update
            )
        else:
            safe_update = update
        
        # Geometrik sınırlama: Parametre büyüklüğüne göre adaptif
        param_norm = torch.norm(param)
        update_norm = torch.norm(safe_update)
        
        if update_norm > 0:
            # Güncelleme, parametre normunun %50'sini geçemez
            max_ratio = 0.5
            ratio = update_norm / (param_norm + eps)
            
            if ratio > max_ratio:
                # Geometrik olarak küçült
                scale_factor = max_ratio / ratio
                safe_update = safe_update * scale_factor
        
        # Son kontrol: Numerik sınırlar
        safe_update = torch.clamp(safe_update, -1e6, 1e6)
        
        return safe_update
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """
        Optimizer adımı.
        
        Her parametre grubu için:
        1. Adım sayısını güncelle (k)
        2. GAP formülüyle öğrenme katsayısı hesapla
        3. Hiperbolik kavis uygula
        4. Geometrik sönümleme kontrolü
        5. Güvenli güncelleme uygula
        
        Args:
            closure: Kayıp fonksiyonunu yeniden değerlendiren fonksiyon
            
        Returns:
            Kayıp değeri (closure varsa)
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Global adım sayısını artır
        self.global_step += 1
        
        for group in self.param_groups:
            a = group['a']
            b = group['b']
            r = group['r']
            curvature_factor = group['curvature_factor']
            grad_threshold = group['grad_threshold']
            max_grad_norm = group['max_grad_norm']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            # Her parametre için state'i başlat/güncelle
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                betas = group.get('betas', (0.9, 0.999))
                beta1, beta2 = betas
                use_first_moment = group.get('use_first_moment', True)
                gradient_centralization = group.get('gradient_centralization', True)
                element_wise_clip = group.get('element_wise_clip', 1.0)
                
                # State sözlüğü başlatma
                if len(self.state[p]) == 0:
                    self.state[p]['step'] = 0
                    # First moment (momentum) - Adam gibi
                    if use_first_moment:
                        self.state[p]['exp_avg'] = torch.zeros_like(p)
                    # Second moment (RMSProp gibi)
                    self.state[p]['exp_avg_sq'] = torch.zeros_like(p)
                
                state = self.state[p]
                state['step'] += 1
                step = state['step']
                
                # === 1. GAP Öğrenme Katsayısı ===
                lr = self._gap_learning_rate(step, a, b, r)
                
                # === 2. Ağırlık Çürütme (Weight Decay) ===
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                
                # === 3. Gradyan Merkezileştirme (Gradient Centralization) ===
                if gradient_centralization and len(grad.shape) > 1:
                    # Gradyanları mean=0 yap - Batch Norm etkisi
                    grad = grad - grad.mean(dim=tuple(range(1, len(grad.shape))), keepdim=True)
                
                # === 4. Element-wise Gradyan Kırpma ===
                if element_wise_clip > 0:
                    grad = torch.clamp(grad, -element_wise_clip, element_wise_clip)
                
                # === 5. Norm Gradyan Kırpma (Gradient Clipping) ===
                grad_norm = torch.norm(grad)
                if grad_norm > max_grad_norm:
                    grad = grad * (max_grad_norm / grad_norm)
                
                # === 6. Hiperbolik Kavis Etkisi ===
                scaled_grad = self._hyperbolic_curvature(
                    grad, curvature_factor, step
                )
                
                # === 7. Geometrik Sönümleme ===
                current_grad_norm = torch.norm(scaled_grad).item()
                damping = self._geometric_damping(
                    current_grad_norm,
                    grad_threshold,
                    step,
                    r
                )
                
                # === 8. First Moment Tahmini (Adam gibi Momentum) ===
                if use_first_moment:
                    exp_avg = state['exp_avg']
                    exp_avg.mul_(beta1).add_(scaled_grad, alpha=1 - beta1)
                    # Bias correction
                    bias_correction1 = 1 - beta1 ** step
                    corrected_exp_avg = exp_avg / bias_correction1
                else:
                    corrected_exp_avg = scaled_grad
                
                # === 9. Second Moment Tahmini (RMSProp benzeri) ===
                exp_avg_sq = state['exp_avg_sq']
                exp_avg_sq.mul_(beta2).addcmul_(scaled_grad, scaled_grad, value=1 - beta2)
                # Bias correction
                bias_correction2 = 1 - beta2 ** step
                corrected_exp_avg_sq = exp_avg_sq / bias_correction2
                
                # === 10. Güvenli Güncelleme ===
                denom = corrected_exp_avg_sq.sqrt().add_(eps)
                step_size = lr * damping
                
                # Adam tarzı update: m / (sqrt(v) + eps)
                update = corrected_exp_avg / denom
                update = update.mul_(step_size)
                
                # Güvenlik mekanizması
                safe_update = self._safe_geometric_clamp(p, update, eps)
                
                # === 11. Parametre Güncelleme ===
                p.add_(safe_update, alpha=-1)
        
        return loss
    
    def get_learning_rate(self, param_group_idx: int = 0) -> float:
        """
        Mevcut öğrenme katsayısını al.
        
        Args:
            param_group_idx: Parametre grubu indeksi
            
        Returns:
            GAP formülüyle hesaplanan mevcut LR
        """
        if param_group_idx >= len(self.param_groups):
            raise IndexError(f"Geçersiz parametre grubu indeksi: {param_group_idx}")
        
        group = self.param_groups[param_group_idx]
        
        # İlk parametrenin adımını kullan
        for p in group['params']:
            if p in self.state and 'step' in self.state[p]:
                step = self.state[p]['step']
                return self._gap_learning_rate(
                    step,
                    group['a'],
                    group['b'],
                    group['r']
                )
        
        return group['a'] + group['b']  # Başlangıç değeri
    
    def __repr__(self) -> str:
        """
        Optimizer'ın string temsili.
        """
        return (
            f"GeomGAPOptimizer("
            f"a={self.defaults['a']}, "
            f"b={self.defaults['b']}, "
            f"r={self.defaults['r']}, "
            f"curvature={self.defaults['curvature_factor']}, "
            f"betas={self.defaults.get('betas', (0.9, 0.999))}, "
            f"first_moment={self.defaults.get('use_first_moment', True)}, "
            f"grad_cent={self.defaults.get('gradient_centralization', True)}, "
            f"steps={self.global_step}"
            f")"
        )


class GeomGAPSGD(Optimizer):
    """
    GAP mantığıyla çalışan SGD varyantı.
    
    Daha basit bir implementasyon, momentum kullanır.
    """
    
    def __init__(
        self,
        params,
        a: float = 0.01,
        b: float = 1e-5,
        r: float = 1.005,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        grad_threshold: float = 10.0
    ):
        defaults = dict(
            a=a, b=b, r=r,
            momentum=momentum,
            weight_decay=weight_decay,
            grad_threshold=grad_threshold
        )
        super(GeomGAPSGD, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                if len(self.state[p]) == 0:
                    self.state[p]['step'] = 0
                    self.state[p]['momentum_buffer'] = torch.zeros_like(p)
                
                state = self.state[p]
                state['step'] += 1
                step = state['step']
                
                # GAP LR
                lr = group['a'] * (group['r'] ** step) + group['b']
                
                # Weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                
                # Geometrik sönümleme
                grad_norm = torch.norm(grad).item()
                if grad_norm > group['grad_threshold']:
                    excess = (grad_norm - group['grad_threshold']) / group['grad_threshold']
                    damping = math.exp(-0.05 * excess)
                    lr *= max(damping, 0.01)
                
                # Momentum
                buf = state['momentum_buffer']
                buf.mul_(group['momentum']).add_(grad)
                
                # Update
                p.add_(buf, alpha=-lr)
        
        return loss
