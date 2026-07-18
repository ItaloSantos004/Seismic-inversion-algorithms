import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import time

dados = np.load('dados_marmousi_P_W.npz')
P_cube = dados['P_cube']
W_cube = dados['W_cube']
Vp_real = dados['Vp_real']

Nx_sub, nx_win, nt = P_cube.shape
ni = 2800                  
nt2 = nt                  # Será 1600 automaticamente
dx = 50.0
w_freq = 2 * np.pi * 50
xi = (2 * np.pi / (nx_win * dx)) * np.arange(-(nx_win//2), (nx_win//2) + 1, dtype=np.float32)

print(f"-> Iniciando inversão de {Nx_sub} colunas...")

# 2. Transformada de Fourier Espacial (Regresso ao domínio dos ângulos)
P_canal = np.zeros_like(P_cube, dtype=np.float32)
W_canal = np.zeros_like(W_cube, dtype=np.float32)

for col in range(Nx_sub):
    for j in range(nt):
        P_canal[col, :, j] = np.real(np.fft.fftshift(np.fft.fft(P_cube[col, :, j])))
        W_canal[col, :, j] = np.real(np.fft.fftshift(np.fft.fft(W_cube[col, :, j])))

# 3. Motor Numba: Descida Recursiva Layer-Peeling
@njit(parallel=True, fastmath=True)
def inversao_layer_peeling(P_canal, W_canal, xi, w_freq, ni, nt2, Nx_sub, nx_win):
    Vel_rec = np.zeros((ni, Nx_sub), dtype=np.float32)
    xi_sq = xi**2
    
    for col in prange(Nx_sub):
        Zrec = np.zeros((nx_win, ni), dtype=np.float32)
        
        for m in range(nx_win):
            Pinv = np.zeros((ni, nt2), dtype=np.float32)
            Winv = np.zeros((ni, nt2), dtype=np.float32)
            Zinv = np.zeros(ni, dtype=np.float32)
            
            # Condição de Superfície
            for j in range(0, nt2, 2):
                if j + 2 < nt2:
                    Pinv[0, j] = P_canal[col, m, j + 2]
                    Winv[0, j] = W_canal[col, m, j + 2]
                    
            if Winv[0, 0] != 0:
                Zinv[0] = Pinv[0, 0] / Winv[0, 0]
            else:
                Zinv[0] = 1.0
                
            # Continuação Descendente (A Recursão)
            for i in range(1, ni):
                for j in range(i, nt2 - i, 2):
                    a = Winv[i-1, j-1] + Winv[i-1, j+1]
                    b = Winv[i-1, j-1] - Winv[i-1, j+1]
                    c = Pinv[i-1, j-1] + Pinv[i-1, j+1]
                    d = Pinv[i-1, j-1] - Pinv[i-1, j+1]
                    
                    if Zinv[i-1] != 0:
                        Winv[i, j] = 0.5 * (a + d / Zinv[i-1])
                        Pinv[i, j] = 0.5 * (Zinv[i-1] * b + c)
                        
                if Winv[i, i] != 0:
                    Zinv[i] = Pinv[i, i] / Winv[i, i]
                else:
                    Zinv[i] = Zinv[i-1]
                    
            for i in range(ni):
                Zrec[m, i] = Zinv[i]
                
        # 4. Regressão Linear para isolar Velocidade (Polyfit manual Numba)
        for i in range(ni):
            y = np.zeros(nx_win, dtype=np.float32)
            for m in range(nx_win):
                if Zrec[m, i] != 0:
                    y[m] = 1.0 / (Zrec[m, i]**2)
                    
            n_pts = nx_win
            sum_x = np.sum(xi_sq)
            sum_y = np.sum(y)
            sum_x2 = np.sum(xi_sq**2)
            sum_xy = np.sum(xi_sq * y)
            denom = (n_pts * sum_x2 - sum_x**2)
            
            if denom != 0:
                p1 = (n_pts * sum_xy - sum_x * sum_y) / denom
                p2 = (sum_y * sum_x2 - sum_x * sum_xy) / denom
                
                # Recuperação da velocidade assumindo Densidade
                if p1 != 0 and (-1.0 / (p1 * w_freq**2)) > 0:
                    prec = np.sqrt(-1.0 / (p1 * w_freq**2))
                    if p2 != 0 and (1.0 / (p2 * prec**2)) > 0:
                        Vel_rec[i, col] = np.sqrt(1.0 / (p2 * prec**2))
                    else:
                        Vel_rec[i, col] = Vel_rec[i-1, col] if i > 0 else 1500.0
                else:
                    Vel_rec[i, col] = Vel_rec[i-1, col] if i > 0 else 1500.0
            else:
                Vel_rec[i, col] = Vel_rec[i-1, col] if i > 0 else 1500.0

    return Vel_rec

inicio = time.time()
Velocidade_Reconstruida = inversao_layer_peeling(P_canal, W_canal, xi, w_freq, ni, nt2, Nx_sub, nx_win)
print(f"-> Inversão 2D concluída com sucesso em {time.time() - inicio:.2f} segundos!")

# 4. Comparativo Visual: Real vs Reconstruído
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

im1 = axes[0].imshow(Vp_real, aspect='auto', cmap='jet', vmin=1500, vmax=4500)
axes[0].set_title('Modelo Marmousi Real - Dividido', fontweight='bold')
axes[0].set_ylabel('Profundidade')
fig.colorbar(im1, ax=axes[0], label='Velocidade (m/s)')

im2 = axes[1].imshow(Velocidade_Reconstruida, aspect='auto', cmap='jet', vmin=1500, vmax=4500)
axes[1].set_title('Marmousi Reconstruído', fontweight='bold')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Profundidade')
fig.colorbar(im2, ax=axes[1], label='Velocidade (m/s)')

plt.tight_layout()
plt.savefig('marmousi_comparativo_inversao.png', dpi=300)
print("-> Imagem comparativa salva como 'marmousi_comparativo_inversao.png'")