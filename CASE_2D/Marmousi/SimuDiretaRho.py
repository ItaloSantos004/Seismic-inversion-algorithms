import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from numba import njit, prange
from scipy.ndimage import gaussian_filter

#carregando dados originais
print("Carregando dados do Marmousi")
dados = sio.loadmat('marmousi_matrizes.mat')

Vp_marm = np.ascontiguousarray(dados['Vp']).astype(np.float32)
Rho_marm = np.ascontiguousarray(dados['Rho']).astype(np.float32) # Nova matriz de densidade
dx = np.float32(np.squeeze(dados['dx']))
dz = np.float32(np.squeeze(dados['dz']))

Nz_marm, Nx_marm = Vp_marm.shape

#suavizando modelo
print("Suavizando modelo")
sig = 3.0
Vp_marm = gaussian_filter(Vp_marm, sigma=sig).astype(np.float32)
Rho_marm = gaussian_filter(Rho_marm, sigma=sig).astype(np.float32)

#gerando e salvando a imagem do modelo suavizado
print("imagem dos modelos suavizados")
max_x_km = (Nx_marm - 1) * float(dx) / 1000.0
max_z_km = (Nz_marm - 1) * float(dz) / 1000.0

figm, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

#velocidade
im1 = ax1.imshow(Vp_marm, cmap='jet', extent=[0, max_x_km, max_z_km, 0], aspect='auto')
cb1 = figm.colorbar(im1, ax=ax1)
cb1.set_label('Velocidade (m/s)', weight='bold')
ax1.set_title('Marmousi - Velocidade Suavizada', fontsize=14)
ax1.set_xlabel('X (km)')
ax1.set_ylabel('Profundidade (km)')

#densidade
im2 = ax2.imshow(Rho_marm, cmap='viridis', extent=[0, max_x_km, max_z_km, 0], aspect='auto')
cb2 = figm.colorbar(im2, ax=ax2)
cb2.set_label('Densidade (kg/m^3)', weight='bold')
ax2.set_title('Marmousi - Densidade Suavizada', fontsize=14)
ax2.set_xlabel('X (km)')
ax2.set_ylabel('Profundidade (km)')

figm.tight_layout()
figm.savefig('Modelo_Suavizado_Marmousi.png', dpi=300)
plt.close(figm)
print("Imagem salva")


L_pml = 300 #espessura do PML

Nz = Nz_marm + L_pml #tamanho da grade total
Nx = Nx_marm + (2 * L_pml)

#criando as matrizes expandidas e simulação no centro
Vp = np.zeros((Nz, Nx), dtype=np.float32)
Rho = np.zeros((Nz, Nx), dtype=np.float32) 

Vp[0:Nz_marm, L_pml:L_pml + Nx_marm] = Vp_marm
Rho[0:Nz_marm, L_pml:L_pml + Nx_marm] = Rho_marm

#preenche a grade do PML com o valor das bordas
for i in range(Nz_marm):
    Vp[i, 0:L_pml] = Vp_marm[i, 0] #esquerda
    Rho[i, 0:L_pml] = Rho_marm[i, 0]
    Vp[i, L_pml + Nx_marm:Nx] = Vp_marm[i, -1] #direita
    Rho[i, L_pml + Nx_marm:Nx] = Rho_marm[i, -1]

for i in range(Nz_marm, Nz): #fundo
    Vp[i, :] = Vp[Nz_marm - 1, :]
    Rho[i, :] = Rho[Nz_marm - 1, :]

#parametros
dt = np.float32(0.0001)
tempo = 40000

Vp_sq = Vp**2 

#perfil de amortecimento
R = 0.0001    
V_max = np.max(Vp)

n_pml = 3.0
d_max = -((n_pml + 1.0) * V_max) / (2.0 * L_pml * dx) * np.log(R)

#calcula o d_max
def get_damping(pos_in_pml, L):
    if pos_in_pml < 0: return 0.0
    if pos_in_pml > L: return d_max
    return d_max * (pos_in_pml / L)**n_pml

sx = np.zeros(Nx, dtype=np.float32) #para U
sz = np.zeros(Nz, dtype=np.float32)
sx_half = np.zeros(Nx, dtype=np.float32) #para Q
sz_half = np.zeros(Nz, dtype=np.float32)

for j in range(Nx):
    if j < L_pml:  #paredes esquerda
        sx[j] = get_damping(L_pml - j, L_pml)
        sx_half[j] = get_damping(L_pml - (j + 0.5), L_pml)
    elif j >= Nx - L_pml: #parede direita
        dist_j = j - (Nx - L_pml)
        sx[j] = get_damping(dist_j + 1, L_pml)
        sx_half[j] = get_damping(dist_j + 0.5, L_pml)

for i in range(Nz): #fundo
    if i >= Nz - L_pml: 
        dist_i = i - (Nz - L_pml)
        sz[i] = get_damping(dist_i + 1, L_pml)
        sz_half[i] = get_damping(dist_i + 0.5, L_pml)

#Variaveis
U  = np.zeros((Nz, Nx, 3), dtype=np.float32)
Qx = np.zeros((Nz, Nx, 2), dtype=np.float32)
Qz = np.zeros((Nz, Nx, 2), dtype=np.float32)

#fonte
t0 = np.float32(0.1)
s = np.float32(0.02)
z0 = 4            
x0 = 6000

#indices
p1, p2, p3 = 0, 1, 2
q1, q2 = 0, 1

#função pra calcular a propagação
@njit(parallel=True, fastmath=True)
def propagacao(U, Qx, Qz, sx, sx_half, sz, sz_half, Vp_sq, Rho, dx, dz, dt, p1, p2, p3, q1, q2, Nz, Nx):
    
    #Atualizando Qx
    for i in prange(1, Nz - 1):
        for j in range(0, Nx - 1):
            du_dx = (U[i, j+1, p2] - U[i, j, p2]) / dx
            sx_val = sx_half[j]
            sz_val = sz[i]
            
            A_minus = 1.0 - (sx_val * dt / 2.0)
            A_plus  = 1.0 + (sx_val * dt / 2.0)
            Qx[i, j, q2] = (A_minus * Qx[i, j, q1] - dt * (sx_val - sz_val) * du_dx) / A_plus

    #Atualizando Qz
    for i in prange(0, Nz - 1):
        for j in range(1, Nx - 1):
            du_dz = (U[i+1, j, p2] - U[i, j, p2]) / dz
            sx_val = sx[j]
            sz_val = sz_half[i]
            
            A_minus = 1.0 - (sz_val * dt / 2.0)
            A_plus  = 1.0 + (sz_val * dt / 2.0)
            Qz[i, j, q2] = (A_minus * Qz[i, j, q1] - dt * (sz_val - sx_val) * du_dz) / A_plus

    #Atualizando U
    for i in prange(1, Nz - 1):
        for j in range(1, Nx - 1):
            
            dqx_dx = (Qx[i, j, q2] - Qx[i, j-1, q2]) / dx
            dqz_dz = (Qz[i, j, q2] - Qz[i-1, j, q2]) / dz

            #Derivada em x
            rho_inv_e = 2.0 / (Rho[i, j+1] + Rho[i, j])
            rho_inv_w = 2.0 / (Rho[i, j] + Rho[i, j-1])
            d2u_dx2_rho = Rho[i, j] * (rho_inv_e * (U[i, j+1, p2] - U[i, j, p2]) - rho_inv_w * (U[i, j, p2] - U[i, j-1, p2])) / (dx**2)

            #Derivada em z
            rho_inv_s = 2.0 / (Rho[i+1, j] + Rho[i, j])
            rho_inv_n = 2.0 / (Rho[i, j] + Rho[i-1, j])
            d2u_dz2_rho = Rho[i, j] * (rho_inv_s * (U[i+1, j, p2] - U[i, j, p2]) - rho_inv_n * (U[i, j, p2] - U[i-1, j, p2])) / (dz**2)

            operador_espacial = d2u_dx2_rho + d2u_dz2_rho + dqx_dx + dqz_dz
            
            alpha = sx[j] + sz[i]
            beta  = sx[j] * sz[i]

            A_u_minus = 1.0 - (alpha * dt / 2.0)
            A_u_plus  = 1.0 + (alpha * dt / 2.0)

            termo_fonte = Vp_sq[i, j] * operador_espacial - beta * U[i, j, p2]

            U[i, j, p3] = (2.0 * U[i, j, p2] - A_u_minus * U[i, j, p1] + (dt**2) * termo_fonte) / A_u_plus

#Loop temporal e o .gif
arq_gif = 'simulacao_marmousi_vel_rho.gif'
writer = imageio.get_writer(arq_gif, mode='I', duration=0.15, loop=0)

fig, ax = plt.subplots(figsize=(10, 5))

print("Iniciando propagação e gravando .gif")

for n in range(1, tempo + 1):
    t_atual = np.float32(n * dt)
    
    propagacao(U, Qx, Qz, sx, sx_half, sz, sz_half, Vp_sq, Rho, dx, dz, dt, p1, p2, p3, q1, q2, Nz, Nx)
    
    #Fonte
    f = np.exp(-((t_atual - t0) / s)**2, dtype=np.float32)
    U[z0, x0, p3] += f
    
    #Condição de dirichlet na superficie
    U[0, :, p3] = 0.0
    
    if n % 250 == 0:
        U_visivel = U[0:Nz_marm, L_pml:L_pml + Nx_marm, p3] #plota só a grade do marmousi
        
        if n == 250:
            im = ax.imshow(U_visivel, cmap='jet', vmin=-0.02, vmax=0.05, 
                           extent=[0, max_x_km, max_z_km, 0], aspect='auto')
            
            title_text = ax.set_title(f'Marmousi- Propagação (t = {t_atual:.4f} s)', fontweight='bold')
            ax.set_xlabel('X (km)')
            ax.set_ylabel('Profundidade (km)')
            fig.tight_layout()
        else:
            im.set_data(U_visivel)
            title_text.set_text(f'Marmousi - Propagação (t = {t_atual:.4f} s)')
        
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        imagem_matriz = rgba.copy() 
        
        writer.append_data(imagem_matriz)
        print(f"Passo {n} / {tempo}")

    p1, p2, p3 = p2, p3, p1 #atualiza os indices
    q1, q2 = q2, q1

writer.close()
plt.close()

print("Finalizado")