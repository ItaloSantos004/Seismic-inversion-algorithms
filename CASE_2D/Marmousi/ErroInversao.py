import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.io as sio

print("Carregando matriz de inversão")
vinv = np.load('matriz_inversão.npy') 

#parametros
ni, Nx_idx = vinv.shape
dx = 50.0

print("Carregando Marmousi Original")
dados = sio.loadmat('marmousi_matrizes.mat')
vm = np.array(dados['Vp'], dtype=np.float32) #extraindo a velocidade
vm = vm[:ni, :]

dxm = 1.25
nxm = vm.shape[1]

print("Interpolação")

#posição
xinv  = np.arange(Nx_idx) * dx
xm = np.arange(nxm) * dxm

vint = np.zeros((ni, nxm), dtype=np.float32)

for i in range(ni): #fazendo interpolação a cada linha
    int = interp1d(xinv, vinv[i, :], kind='cubic', fill_value='extrapolate')
    vint[i, :] = int(xm)

print("Calculando a matriz de erro")
abs_erro = np.abs(vint - vm)
rel_erro = (abs_erro / vm) * 100

#visualização
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

xkm1 = xm[0] / 1000
xkm2 = xm[-1] / 1000
zkm = (ni * dxm) / 1000

eixo = [xkm1, xkm2, zkm, 0] #invertendo

#original
im1 = axes[0].imshow(vm, aspect='auto', cmap='jet', vmin=1500, vmax=4500, extent=eixo)
axes[0].set_title('Modelo Original Marmousi', fontweight='bold')
axes[0].set_ylabel('Profundidade (km)')
fig.colorbar(im1, ax=axes[0], label='Velocidade (m/s)')

#interpolação
im2 = axes[1].imshow(vint, aspect='auto', cmap='jet', vmin=1500, vmax=4500, extent=eixo)
axes[1].set_title('Inversão com Interpolação', fontweight='bold')
axes[1].set_ylabel('Profundidade (km)')
fig.colorbar(im2, ax=axes[1], label='Velocidade (m/s)')

#erro
im3 = axes[2].imshow(rel_erro, aspect='auto', cmap='hot', vmin=0, vmax=10, extent=eixo)
axes[2].set_title('Erro (%)', fontweight='bold')
axes[2].set_xlabel('X (km)')
axes[2].set_ylabel('Profundidade (km)')
fig.colorbar(im3, ax=axes[2], label='Erro (%)')

plt.tight_layout()
plt.savefig('erro_marmousi.png', dpi=300)
print("Imagem salva")