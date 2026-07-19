import numpy as np
import scipy.io as sio
from numba import njit, prange

#matrizes reais marmousi
dados = sio.loadmat('marmousi_matrizes.mat')
Vel = np.ascontiguousarray(dados['Vp']).astype(np.float32)

#parametros
dx = 50.0  #espaçamento de 50m               
w0 = 2 * np.pi * 50  
n_sens = 31 #numero de sensores na janela          
ni = 2800 #percorre todos os 3,5 km          
nt = 5600                

#pegando os dados amostrados
idx = int(dx // 1.25) #criando um passo maior de 50m
Vel_idx = Vel[:ni, ::idx]
Nz_idx, Nx_idx = Vel_idx.shape

print(f"Amostragem dos dados para {Nx_idx} colunas e {Nz_idx} camadas.")

#matrizes no domínio dos ângulos
P_ang = np.zeros((Nx_idx, n_sens, nt), dtype=np.complex64)
W_ang = np.zeros((Nx_idx, n_sens, nt), dtype=np.complex64)

#matrizes finais
P_real = np.zeros((Nx_idx, n_sens, nt), dtype=np.float32)
W_real = np.zeros((Nx_idx, n_sens, nt), dtype=np.float32)

#vetor do numero de onda
xi = (2 * np.pi / (n_sens * dx)) * np.arange(-(n_sens//2), (n_sens//2) + 1, dtype=np.float32)
fonte = np.zeros(nt, dtype=np.float32)
fonte[2] = 1.0 #fonte

print("Modelagem Direta")
#loop direto para gerar os dados
@njit(parallel=True, fastmath=True)
def geradorPW(Vel_idx, Paux, Waux, xi, fonte, w0, ni, nt, Nx_idx, n_sens):
    for col in prange(Nx_idx):
            
        creal = Vel_idx[:, col]
        preal = np.ones(ni, dtype=np.float32) * 1000.0 #densidade constante
        
        Pteo = np.zeros((n_sens, nt), dtype=np.complex64)
        Wteo = np.zeros((n_sens, nt), dtype=np.complex64)
        
        for m in range(n_sens):
            Zm = np.zeros(ni, dtype=np.complex64)
            for i in range(ni):
                termo = (w0 / creal[i])**2 - xi[m]**2
                if termo > 0:
                    kz = np.sqrt(termo)
                    Zm[i] = preal[i] * w0 / kz
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
            Paux[col, :, j] = Pteo[:, j]
            Waux[col, :, j] = Wteo[:, j]

print("gerando os dados PW")
geradorPW(Vel_idx, P_ang, W_ang, xi, fonte, w0, ni, nt, Nx_idx, n_sens)

print("aplicando transformada de Fourier")
for col in range(Nx_idx):
    for j in range(nt):
        P_real[col, :, j] = np.real(np.fft.ifft(np.fft.ifftshift(P_ang[col, :, j])))
        W_real[col, :, j] = np.real(np.fft.ifft(np.fft.ifftshift(W_ang[col, :, j])))

np.savez('dados_marmousi_P_W.npz', P_real=P_real, W_real=W_real, Vp_real=Vel_idx)

print("Dados salvos")