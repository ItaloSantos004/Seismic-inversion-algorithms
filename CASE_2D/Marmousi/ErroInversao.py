import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from scipy.interpolate import interp1d
import scipy.io as sio #pra ler .mat

#extraindo os dados para inversão
dados = np.load('dados_marmousi_P_W.npz')
P_real = dados['P_real']
W_real = dados['W_real']
Vel_idx = dados['Vp_real']

#parametros
Nx_idx, n_sens, nt = P_real.shape
ni = 2800                  
nt2 = nt 
dx = 50.0
w0 = 2 * np.pi * 50
xi = (2 * np.pi / (n_sens * dx)) * np.arange(-(n_sens//2), (n_sens//2) + 1, dtype=np.float32)

print("Inicio - fazendo transformada de fourier")

#fazendo trasnformada para voltar pro dominio dos angulos
P_canal = np.zeros_like(P_real, dtype=np.float32)
W_canal = np.zeros_like(W_real, dtype=np.float32)

for col in range(Nx_idx):
    for j in range(nt):
        P_canal[col, :, j] = np.real(np.fft.fftshift(np.fft.fft(P_real[col, :, j])))
        W_canal[col, :, j] = np.real(np.fft.fftshift(np.fft.fft(W_real[col, :, j])))

print("Inversão")
#iniciando loop inversão
@njit(parallel=True, fastmath=True)
def inversao(P_canal, W_canal, xi, w0, ni, nt2, Nx_idx, n_sens):
    Vel_rec = np.zeros((ni, Nx_idx), dtype=np.float32)
    xi2 = xi**2
    
    for col in prange(Nx_idx):
        Zrec = np.zeros((n_sens, ni), dtype=np.float32)
        
        for m in range(n_sens):
            Pinv = np.zeros((ni, nt2), dtype=np.float32)
            Winv = np.zeros((ni, nt2), dtype=np.float32)
            Zinv = np.zeros(ni, dtype=np.float32)
            
            #superficie
            for j in range(0, nt2, 2):
                if j + 2 < nt2:
                    Pinv[0, j] = P_canal[col, m, j + 2]
                    Winv[0, j] = W_canal[col, m, j + 2]
                    
            if Winv[0, 0] != 0:
                Zinv[0] = Pinv[0, 0] / Winv[0, 0]
            else:
                Zinv[0] = 1.0
                
            #recursao
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
                
        #separando a velocidade
        for i in range(ni):
            y = np.zeros(n_sens, dtype=np.float32)
            for m in range(n_sens):
                if Zrec[m, i] != 0:
                    y[m] = 1.0 / (Zrec[m, i]**2)
                    
            n_pts = n_sens
            sum_x = np.sum(xi2)
            sum_y = np.sum(y)
            sum_x2 = np.sum(xi2**2)
            sum_xy = np.sum(xi2 * y)
            denom = (n_pts * sum_x2 - sum_x**2)
            
            if denom != 0:
                p1 = (n_pts * sum_xy - sum_x * sum_y) / denom
                p2 = (sum_y * sum_x2 - sum_x * sum_xy) / denom
                
                #recuperando velocidade com densidade constante
                if p1 != 0 and (-1.0 / (p1 * w0**2)) > 0:
                    prec = np.sqrt(-1.0 / (p1 * w0**2))
                    if p2 != 0 and (1.0 / (p2 * prec**2)) > 0:
                        Vel_rec[i, col] = np.sqrt(1.0 / (p2 * prec**2))
                    else:
                        Vel_rec[i, col] = Vel_rec[i-1, col] if i > 0 else 1500.0
                else:
                    Vel_rec[i, col] = Vel_rec[i-1, col] if i > 0 else 1500.0
            else:
                Vel_rec[i, col] = Vel_rec[i-1, col] if i > 0 else 1500.0

    return Vel_rec


vel_inversao = inversao(P_canal, W_canal, xi, w0, ni, nt2, Nx_idx, n_sens)
print("Inversão Concluida")

#fazendo interpolação e vendo o erro
print("carregando dados originais")

try:
    dados_orig = sio.loadmat('marmousi_matrizes.mat')
    
    v_true = np.array(dados_orig['Vp'], dtype=np.float32) #extraindo a velocidade
    
    v_true = v_true[:ni, :]
    
except FileNotFoundError:
    print("Erro")
    exit()

dx_true = 1.25
nx_true = v_true.shape[1]

print("Interpolação")

#posição
x_inv  = np.arange(Nx_idx) * dx
x_true = np.arange(nx_true) * dx_true

#inicializando matriz interpolada
v_intp = np.zeros((ni, nx_true), dtype=np.float32)

#fazendo interpolação 
for i in range(ni):
    interpolador = interp1d(x_inv, vel_inversao[i, :], kind='cubic', fill_value='extrapolate')
    v_intp[i, :] = interpolador(x_true)

print("calculando a matriz de erro")
erro_absoluto = np.abs(v_intp - v_true)
erro_relativo = (erro_absoluto / v_true) * 100

#visualização
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

#km
x_min_km = x_true[0] / 1000
x_max_km = x_true[-1] / 1000
z_max_km = (ni * dx_true) / 1000

extent_km = [x_min_km, x_max_km, z_max_km, 0] #apenas invertendo

#modelo original
im1 = axes[0].imshow(v_true, aspect='auto', cmap='jet', vmin=1500, vmax=4500, extent=extent_km)
axes[0].set_title('Modelo Original Marmousi', fontweight='bold')
axes[0].set_ylabel('Profundidade (km)')
fig.colorbar(im1, ax=axes[0], label='Velocidade (m/s)')

#modelo interpolado
im2 = axes[1].imshow(v_intp, aspect='auto', cmap='jet', vmin=1500, vmax=4500, extent=extent_km)
axes[1].set_title('Inversão com Interpolação', fontweight='bold')
axes[1].set_ylabel('Profundidade (km)')
fig.colorbar(im2, ax=axes[1], label='Velocidade (m/s)')

#erro relativo
im3 = axes[2].imshow(erro_relativo, aspect='auto', cmap='hot', vmin=0, vmax=10, extent=extent_km)
axes[2].set_title('Erro (%)', fontweight='bold')
axes[2].set_xlabel('Distância X (km)')
axes[2].set_ylabel('Profundidade (km)')
fig.colorbar(im3, ax=axes[2], label='Erro (%)')

plt.tight_layout()
plt.savefig('erro_marmousi.png', dpi=300)
print("Imagem Salva")