import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import time
from scipy.interpolate import interp1d
import scipy.io as sio # Biblioteca necessária para ler arquivos .mat

# 1. Carregamento dos Dados de Entrada (Subamostrados para Inversão)
dados = np.load('dados_marmousi_P_W.npz')
P_cube = dados['P_cube']
W_cube = dados['W_cube']
Vp_real = dados['Vp_real'] # Matriz real no passo de 50m (usada no plot simples)

Nx_sub, nx_win, nt = P_cube.shape
ni = 2800                  # Profundidade em amostras
nt2 = nt                   # Será 1600 automaticamente
dx_inv = 50.0              # Passo da inversão
w_freq = 2 * np.pi * 50
xi = (2 * np.pi / (nx_win * dx_inv)) * np.arange(-(nx_win//2), (nx_win//2) + 1, dtype=np.float32)

print(f"-> Iniciando inversão Layer-Peeling de {Nx_sub} colunas...")

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
                
            # Continuação Descendente
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
                
        # 4. Regressão Linear para isolar Velocidade
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
print(f"-> Inversão concluída com sucesso em {time.time() - inicio:.2f} segundos!")

# ==============================================================================
# 5. MÓDULO DE INTERPOLAÇÃO E ANÁLISE DE ERRO (Lendo do .mat)
# ==============================================================================
print("-> Carregando matriz original do Marmousi (marmousi_matrizes.mat)...")

try:
    # Lê o arquivo .mat exportado pelo seu script Octave
    mat_data = sio.loadmat('marmousi_matrizes.mat')
    
    # Extrai a variável 'Vp' e garante que é um array NumPy float32
    v_true_high_res = np.array(mat_data['Vp'], dtype=np.float32)
    
    # -> ADICIONE ESTA LINHA PARA RESOLVER O ERRO <-
    v_true_high_res = v_true_high_res[:ni, :]
    
except FileNotFoundError:
    print("CRÍTICO: Arquivo 'marmousi_matrizes.mat' não encontrado no diretório.")
    print("Certifique-se de rodar o script Octave antes para gerar o arquivo.")
    exit()

dx_true = 1.25
nx_true = v_true_high_res.shape[1]

print("-> Iniciando interpolação horizontal (dx = 50m para dx = 1.25m)...")

# Vetores de posição espacial
x_inv  = np.arange(Nx_sub) * dx_inv
x_true = np.arange(nx_true) * dx_true

# Pré-alocando a matriz interpolada
v_inv_interp = np.zeros((ni, nx_true), dtype=np.float32)

# Interpolação linha por linha
for i in range(ni):
    # kind='cubic' simula o comportamento de splines para manter suavidade
    # fill_value='extrapolate' garante que as bordas não gerem erros (NaN)
    interpolador = interp1d(x_inv, Velocidade_Reconstruida[i, :], kind='cubic', fill_value='extrapolate')
    v_inv_interp[i, :] = interpolador(x_true)

print("-> Calculando matriz de erro relativo...")
# Cálculo das Matrizes de Erro
erro_absoluto = np.abs(v_inv_interp - v_true_high_res)
erro_relativo = (erro_absoluto / v_true_high_res) * 100

# ==============================================================================
# 6. VISUALIZAÇÃO: ORIGINAL vs INTERPOLADO vs ERRO RELATIVO
# ==============================================================================
print("-> Gerando gráficos de comparação...")
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Convertendo limites de metros para QUILÔMETROS
x_min_km = x_true[0] / 1000
x_max_km = x_true[-1] / 1000
z_max_km = (ni * dx_true) / 1000 # 2800 * 1.25 / 1000 = 3.5 km

# A ordem do extent é: [esquerda, direita, inferior, superior]
# Colocamos o z_max_km em 'inferior' e 0 em 'superior' para o eixo Y crescer para baixo
extent_km = [x_min_km, x_max_km, z_max_km, 0]

# Plot 1: Modelo Original Alta Resolução
im1 = axes[0].imshow(v_true_high_res, aspect='auto', cmap='jet', vmin=1500, vmax=4500, extent=extent_km)
axes[0].set_title('Modelo Original Marmousi', fontweight='bold')
axes[0].set_ylabel('Profundidade (km)')
fig.colorbar(im1, ax=axes[0], label='Velocidade (m/s)')

# Plot 2: Modelo Invertido Interpolado
im2 = axes[1].imshow(v_inv_interp, aspect='auto', cmap='jet', vmin=1500, vmax=4500, extent=extent_km)
axes[1].set_title('Inversão com Interpolação', fontweight='bold')
axes[1].set_ylabel('Profundidade (km)')
fig.colorbar(im2, ax=axes[1], label='Velocidade (m/s)')

# Plot 3: Erro Relativo
im3 = axes[2].imshow(erro_relativo, aspect='auto', cmap='hot', vmin=0, vmax=10, extent=extent_km)
axes[2].set_title('Erro (%)', fontweight='bold')
axes[2].set_xlabel('Distância X (km)')
axes[2].set_ylabel('Profundidade (km)')
fig.colorbar(im3, ax=axes[2], label='Erro (%)')

plt.tight_layout()
plt.savefig('analise_erro_marmousi_km.png', dpi=300)
print("-> Imagem da análise salva com sucesso!")