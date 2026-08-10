import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
import scipy.io as sio

print("Carregando matrizes de inversão (Velocidade e Densidade)...")
dados_inv = np.load('matrizes_inversao_completa.npz')
vinv = dados_inv['Vel_rec']
rhoinv = dados_inv['Rho_rec']

#parametros
ni, Nx_idx = vinv.shape
dx = 50.0

print("Carregando dados originais")
dados = sio.loadmat('marmousi_matrizes.mat')
vm = np.array(dados['Vp'], dtype=np.float32)[:ni, :]
rhom = np.array(dados['Rho'], dtype=np.float32)[:ni, :]

#suavizando o modelo
print("Suavizando dados originais")
sig = 3.0
vm_suav = gaussian_filter(vm, sigma=sig).astype(np.float32)
rhom_suav = gaussian_filter(rhom, sigma=sig).astype(np.float32)

dxm = 1.25
nxm = vm.shape[1]

print("Interpolação")

xinv  = np.arange(Nx_idx) * dx #posiçoes
xm = np.arange(nxm) * dxm

#matrizes interpoladas
vint = np.zeros((ni, nxm), dtype=np.float32) 
rhoint = np.zeros((ni, nxm), dtype=np.float32)

#interpola cada linha
for i in range(ni):
    #velocidade
    int_v = interp1d(xinv, vinv[i, :], kind='cubic', fill_value='extrapolate')
    vint[i, :] = int_v(xm)
    
    #densidade
    int_rho = interp1d(xinv, rhoinv[i, :], kind='cubic', fill_value='extrapolate')
    rhoint[i, :] = int_rho(xm)

print("Matrizes de erro")
#erro velocidade
abs_erro_v = np.abs(vint - vm_suav)
rel_erro_v = (abs_erro_v / vm_suav) * 100

#erro densidade
abs_erro_rho = np.abs(rhoint - rhom_suav)
rel_erro_rho = (abs_erro_rho / rhom_suav) * 100

print(f"Erro máximo de Velocidade: {np.max(rel_erro_v):.2f}%")
print(f"Erro máximo de Densidade: {np.max(rel_erro_rho):.2f}%")

#eixos em km
xkm1 = xm[0] / 1000
xkm2 = xm[-1] / 1000
zkm = (ni * dxm) / 1000
eixo = [xkm1, xkm2, zkm, 0] 

#visualização velocidade
print("Imagem - Velocidade")
fig_v, axes_v = plt.subplots(3, 1, figsize=(14, 12))

min_vel, max_vel = np.min(vm_suav), np.max(vm_suav)

#suavidade
im1 = axes_v[0].imshow(vm_suav, aspect='auto', cmap='jet', vmin=min_vel, vmax=max_vel, extent=eixo)
axes_v[0].set_title('Marmousi Suavizado - Velocidade', fontweight='bold')
axes_v[0].set_ylabel('Profundidade (km)')
fig_v.colorbar(im1, ax=axes_v[0], label='Velocidade (m/s)')

#interpolado
im2 = axes_v[1].imshow(vint, aspect='auto', cmap='jet', vmin=min_vel, vmax=max_vel, extent=eixo)
axes_v[1].set_title('Inversão com Interpolação - Velocidade', fontweight='bold')
axes_v[1].set_ylabel('Profundidade (km)')
fig_v.colorbar(im2, ax=axes_v[1], label='Velocidade (m/s)')

#erro
im3 = axes_v[2].imshow(rel_erro_v, aspect='auto', cmap='turbo', vmin=0, vmax=100, extent=eixo)
axes_v[2].set_title('Erro (%)', fontweight='bold')
axes_v[2].set_xlabel('X (km)')
axes_v[2].set_ylabel('Profundidade (km)')
fig_v.colorbar(im3, ax=axes_v[2], label='Erro (%)')

fig_v.tight_layout()
fig_v.savefig('erro_marmousi_velocidade.png', dpi=300)
plt.close(fig_v)

#visualização densidade
print("Imagem - Densidade")
fig_r, axes_r = plt.subplots(3, 1, figsize=(14, 12))

min_rho, max_rho = np.min(rhom_suav), np.max(rhom_suav)

#suavizado
im4 = axes_r[0].imshow(rhom_suav, aspect='auto', cmap='viridis', vmin=min_rho, vmax=max_rho, extent=eixo)
axes_r[0].set_title('Marmousi Suavizado - Densidade', fontweight='bold')
axes_r[0].set_ylabel('Profundidade (km)')
fig_r.colorbar(im4, ax=axes_r[0], label='Densidade (kg/m³)')

#interpolado
im5 = axes_r[1].imshow(rhoint, aspect='auto', cmap='viridis', vmin=min_rho, vmax=max_rho, extent=eixo)
axes_r[1].set_title('Inversão com Interpolação - Densidade', fontweight='bold')
axes_r[1].set_ylabel('Profundidade (km)')
fig_r.colorbar(im5, ax=axes_r[1], label='Densidade (kg/m³)')

#erro
im6 = axes_r[2].imshow(rel_erro_rho, aspect='auto', cmap='turbo', vmin=0, vmax=100, extent=eixo)
axes_r[2].set_title('Erro (%)', fontweight='bold')
axes_r[2].set_xlabel('X (km)')
axes_r[2].set_ylabel('Profundidade (km)')
fig_r.colorbar(im6, ax=axes_r[2], label='Erro (%)')

fig_r.tight_layout()
fig_r.savefig('erro_marmousi_densidade.png', dpi=300)
plt.close(fig_r)

print("Pronto")