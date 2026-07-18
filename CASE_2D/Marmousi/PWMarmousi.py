import numpy as np
import scipy.io as sio
from numba import njit, prange
import time

dados = sio.loadmat('marmousi_matrizes.mat')
Vp_full = np.ascontiguousarray(dados['Vp']).astype(np.float32)

dx = 50.0                
w_freq = 2 * np.pi * 50  
nx_win = 31           
ni = 2800          
nt = 5600                

# Subamostrando as colunas do Marmousi para casar com o dx seguro
passo_subamostragem = int(dx // 1.25)
Vp_sub = Vp_full[:ni, ::passo_subamostragem]
Nz_sub, Nx_sub = Vp_sub.shape

print(f"-> Marmousi subamostrado para {Nx_sub} colunas e {Nz_sub} camadas.")

# Matrizes temporárias para guardar os dados complexos no domínio dos ângulos
P_cube_complex = np.zeros((Nx_sub, nx_win, nt), dtype=np.complex64)
W_cube_complex = np.zeros((Nx_sub, nx_win, nt), dtype=np.complex64)

# Matrizes finais reais para os sensores
P_cube = np.zeros((Nx_sub, nx_win, nt), dtype=np.float32)
W_cube = np.zeros((Nx_sub, nx_win, nt), dtype=np.float32)

# Vetor de números de onda espaciais
xi = (2 * np.pi / (nx_win * dx)) * np.arange(-(nx_win//2), (nx_win//2) + 1, dtype=np.float32)
fonte = np.zeros(nt, dtype=np.float32)
fonte[2] = 1.0 # Pulso na superfície

@njit(parallel=True, fastmath=True)
def gerar_dados_diretos_numba(Vp_sub, P_cube_comp, W_cube_comp, xi, fonte, w_freq, ni, nt, Nx_sub, nx_win):
    for col in prange(Nx_sub):
        if col % 10 == 0:
            print("Modelagem Direta - Processando coluna:", col, "de", Nx_sub)
            
        creal = Vp_sub[:, col]
        preal = np.ones(ni, dtype=np.float32) * 1000.0
        
        Pteo = np.zeros((nx_win, nt), dtype=np.complex64)
        Wteo = np.zeros((nx_win, nt), dtype=np.complex64)
        
        for m in range(nx_win):
            Zm = np.zeros(ni, dtype=np.complex64)
            for i in range(ni):
                termo = (w_freq / creal[i])**2 - xi[m]**2
                if termo > 0:
                    kz = np.sqrt(termo)
                    Zm[i] = preal[i] * w_freq / kz
                else:
                    Zm[i] = Zm[i-1] if i > 0 else (preal[i] * creal[i])
            
            P = np.zeros((ni, nt), dtype=np.complex64)
            W = np.zeros((ni, nt), dtype=np.complex64)
            
            for j in range(1, nt):
                if (0 + j) % 2 == 0:
                    P[0, j] = fonte[j]
                    U = Zm[0] * W[1, j-1] - P[1, j-1]
                    W[0, j] = (P[0, j] + U) / Zm[0]
                
                for i in range(1, ni - 1):
                    if (i + j) % 2 == 0:
                        D = Zm[i-1] * W[i-1, j-1] + P[i-1, j-1]
                        U = Zm[i] * W[i+1, j-1] - P[i+1, j-1]
                        W[i, j] = (D + U) / (Zm[i-1] + Zm[i])
                        P[i, j] = (Zm[i] * D - Zm[i-1] * U) / (Zm[i-1] + Zm[i])
                
                if (ni - 1 + j) % 2 == 0:
                    D = Zm[ni-2] * W[ni-2, j-1] + P[ni-2, j-1]
                    W[ni-1, j] = D / (2 * Zm[ni-2])
                    P[ni-1, j] = D / 2.0
            
            Pteo[m, :] = P[0, :]
            Wteo[m, :] = W[0, :]
            
        for j in range(nt):
            P_cube_comp[col, :, j] = Pteo[:, j]
            W_cube_comp[col, :, j] = Wteo[:, j]

inicio = time.time()
print("-> Iniciando motor paralelo (Numba)...")
gerar_dados_diretos_numba(Vp_sub, P_cube_complex, W_cube_complex, xi, fonte, w_freq, ni, nt, Nx_sub, nx_win)

print("-> Aplicando Transformada Inversa (IFFT) via NumPy...")
for col in range(Nx_sub):
    for j in range(nt):
        P_cube[col, :, j] = np.real(np.fft.ifft(np.fft.ifftshift(P_cube_complex[col, :, j])))
        W_cube[col, :, j] = np.real(np.fft.ifft(np.fft.ifftshift(W_cube_complex[col, :, j])))

np.savez('dados_marmousi_P_W.npz', P_cube=P_cube, W_cube=W_cube, Vp_real=Vp_sub)

print(f"-> Dados P e W gerados com sucesso em {time.time() - inicio:.2f} segundos!")
print("-> Ficheiro guardado como 'dados_marmousi_P_W.npz'")