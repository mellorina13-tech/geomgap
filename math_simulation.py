"""
Mathematical Engine: Geometric Prime Polynomial (GAP) Simulation

This module simulates the behavior of the GAP formula for different r values:
η_k = a · r^k + b

Goal: Detect gradient explosion points and determine the optimal r parameter.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict


class GAPSimulator:
    """
    Geometric Prime Polynomial simulator.

    Args:
        a: Initial coefficient (scale)
        b: Bias / base rate (offset)
        r: Geometric multiplier (growth factor)
        max_steps: Maximum simulation steps
        gradient_norm_threshold: Gradient explosion threshold
    """
    
    def __init__(
        self,
        a: float = 0.001,
        b: float = 0.0001,
        r: float = 1.01,
        max_steps: int = 10000,
        gradient_norm_threshold: float = 1e6
    ):
        self.a = a
        self.b = b
        self.r = r
        self.max_steps = max_steps
        self.gradient_norm_threshold = gradient_norm_threshold
        
    def learning_rate(self, k: int) -> float:
        """
        GAP formülü: η_k = a · r^k + b
        
        Args:
            k: Adım sayısı (step count)
            
        Returns:
            O anki öğrenme katsayısı
        """
        return self.a * (self.r ** k) + self.b
    
    def geometric_damping(self, grad_norm: float, step: int) -> float:
        """
        Amortisör etkisi: Gradyan normu eşiği aştığında
        geometrik sönümleme uygula.
        
        Args:
            grad_norm: Gradyan normu
            step: Mevcut adım
            
        Returns:
            Sönümleme katsayısı (0-1 arası)
        """
        if grad_norm < self.gradient_norm_threshold:
            return 1.0
        
        # Geometrik sönümleme: exp(-α · (||g|| - threshold) / threshold)
        excess = (grad_norm - self.gradient_norm_threshold) / self.gradient_norm_threshold
        alpha = 0.1 * (self.r ** (step / 1000))  # Adım sayısı arttıkça daha agresif
        damping = np.exp(-alpha * excess)
        
        return max(damping, 0.01)  # Minimum %1 güncelleme
    
    def simulate_gradient_flow(
        self,
        initial_grad_norm: float = 1.0,
        noise_std: float = 0.1
    ) -> Dict[str, np.ndarray]:
        """
        Gradyan akışını simüle et.
        
        Returns:
            Simülasyon sonuçları sözlüğü
        """
        steps = np.arange(self.max_steps)
        learning_rates = np.array([self.learning_rate(k) for k in steps])
        
        # Gradyan normlarını simüle et (basit bir model)
        grad_norms = [initial_grad_norm]
        damped_lr = []
        
        for k in range(self.max_steps):
            lr = learning_rates[k]
            damping = self.geometric_damping(grad_norms[-1], k)
            effective_lr = lr * damping
            damped_lr.append(effective_lr)
            
            # Gradyan normunun büyümesi (basit model)
            # ||g_{k+1}|| ≈ ||g_k|| · (1 + η_k · curvature)
            curvature = 0.01 + np.random.normal(0, noise_std)
            new_norm = grad_norms[-1] * (1 + effective_lr * curvature)
            grad_norms.append(new_norm)
        
        return {
            'steps': steps,
            'learning_rates': learning_rates,
            'damped_lr': np.array(damped_lr),
            'grad_norms': np.array(grad_norms[:-1]),
            'explosion_step': next(
                (i for i, g in enumerate(grad_norms) if g > self.gradient_norm_threshold),
                None
            )
        }
    
    @staticmethod
    def find_critical_r_values(
        r_range: Tuple[float, float] = (1.001, 1.1),
        num_points: int = 50,
        a: float = 0.001,
        b: float = 0.0001,
        max_steps: int = 5000
    ) -> List[Dict]:
        """
        Farklı r değerleri için patlama noktalarını bul.
        
        Returns:
            Her r değeri için patlama adımı bilgisi
        """
        r_values = np.linspace(r_range[0], r_range[1], num_points)
        results = []
        
        for r in r_values:
            sim = GAPSimulator(a=a, b=b, r=r, max_steps=max_steps)
            result = sim.simulate_gradient_flow()
            
            results.append({
                'r': r,
                'explosion_step': result['explosion_step'],
                'final_lr': result['learning_rates'][-1],
                'stable': result['explosion_step'] is None or result['explosion_step'] > max_steps * 0.9
            })
        
        return results


def plot_gap_analysis(results: List[Dict], save_path: str = None):
    """
    GAP analiz sonuçlarını görselleştir.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    r_values = [r['r'] for r in results]
    explosion_steps = [
        r['explosion_step'] if r['explosion_step'] is not None 
        else 5000 
        for r in results
    ]
    final_lrs = [r['final_lr'] for r in results]
    stable = [r['stable'] for r in results]
    
    # 1. r vs Patlama Adımı
    ax1 = axes[0, 0]
    colors = ['green' if s else 'red' for s in stable]
    ax1.scatter(r_values, explosion_steps, c=colors, alpha=0.6)
    ax1.set_xlabel('Geometrik Çarpan (r)')
    ax1.set_ylabel('Patlama Adımı')
    ax1.set_title('r Değeri vs Gradyan Patlama Noktası')
    ax1.axhline(y=4500, color='orange', linestyle='--', label='Güvenli Bölge')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. r vs Final LR
    ax2 = axes[0, 1]
    ax2.semilogy(r_values, final_lrs, 'b-', linewidth=2)
    ax2.set_xlabel('Geometrik Çarpan (r)')
    ax2.set_ylabel('Final Öğrenme Katsayısı (log)')
    ax2.set_title('r Değeri vs Öğrenme Hızı Büyümesi')
    ax2.grid(True, alpha=0.3)
    
    # 3. Örnek simülasyon (r=1.01)
    ax3 = axes[1, 0]
    sim = GAPSimulator(r=1.01, max_steps=3000)
    result = sim.simulate_gradient_flow()
    ax3.semilogy(result['steps'], result['learning_rates'], 'b-', label='η_k = a·r^k + b')
    ax3.semilogy(result['steps'], result['damped_lr'], 'r--', label='Amortisörlü')
    ax3.set_xlabel('Adım (k)')
    ax3.set_ylabel('Öğrenme Katsayısı (log)')
    ax3.set_title('GAP Formülü Davranışı (r=1.01)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Gradyan normu
    ax4 = axes[1, 1]
    ax4.semilogy(result['steps'], result['grad_norms'], 'g-', linewidth=2)
    ax4.axhline(y=sim.gradient_norm_threshold, color='r', linestyle='--', label='Eşik')
    ax4.set_xlabel('Adım (k)')
    ax4.set_ylabel('Gradyan Normu (log)')
    ax4.set_title('Gradyan Normu Büyümesi')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


if __name__ == '__main__':
    # Simülasyon çalıştır
    print("GAP Matematiksel Simülasyonu Başlatılıyor...")
    print("=" * 50)
    
    # Kritik r değerlerini bul
    results = GAPSimulator.find_critical_r_values(
        r_range=(1.001, 1.1),
        num_points=100,
        max_steps=5000
    )
    
    # Sonuçları yazdır
    stable_configs = [r for r in results if r['stable']]
    unstable_configs = [r for r in results if not r['stable']]
    
    print(f"\nStabil yapılandırmalar: {len(stable_configs)}")
    print(f"Stabil olmayan yapılandırmalar: {len(unstable_configs)}")
    
    if stable_configs:
        max_stable_r = max(r['r'] for r in stable_configs)
        print(f"Maksimum stabil r değeri: {max_stable_r:.4f}")
    
    # Görselleştir
    fig = plot_gap_analysis(results, save_path='geomgap/simulation_results.png')
    print("\nGrafik kaydedildi: geomgap/simulation_results.png")
    # plt.show()
