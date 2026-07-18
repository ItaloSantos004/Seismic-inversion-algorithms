import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from numba import njit, prange
import time

# =========================================================================
# 1. IMPORTAÇÃO E EXPANSÃO DO DOMÍNIO (MARMOUSI2 COM PADDING EXTERNO)
# =========================================================================
print("Carregando matrizes completas do Marmousi...")
dados = sio.loadmat('marmousi_matrizes.mat')

Vp_marm = np.ascontiguousarray(dados['Vp']).astype(np.float32)
dx = np.float32(np.squeeze(dados['dx']))
dz = np.float32(np.squeeze(dados['dz']))

Nz_marm, Nx_marm = Vp_marm.shape

# Definindo a espessura do PML externo (250 pixels)
L_pml = 300

# Novo tamanho total da grade (Marmousi + Bordas PML)
# Topo (z=0) mantém-se como superfície livre (sem padding em cima)
Nz = Nz_marm + L_pml
Nx = Nx_marm + (2 * L_pml)

print(f"-> Tamanho original do Marmousi: {Nx_marm} x {Nz_marm} pixels.")
print(f"-> Tamanho expandido com PML externo: {Nx} x {Nz} pixels.")

# Criando a nova matriz Vp expandida
Vp = np.zeros((Nz, Nx), dtype=np.float32)

# Insere o Marmousi no centro exato da nova matriz
Vp[0:Nz_marm, L_pml:L_pml + Nx_marm] = Vp_marm

# Preenche as extensões laterais e inferiores com os valores de borda correspondentes
for i in range(Nz_marm):
    Vp[i, 0:L_pml] = Vp_marm[i, 0]                     # Borda esquerda
    Vp[i, L_pml + Nx_marm:Nx] = Vp_marm[i, -1]        # Borda direita

for i in range(Nz_marm, Nz):
    Vp[i, :] = Vp[Nz_marm - 1, :]                      # Borda inferior

# Coordenadas em km para o gráfico completo
x_km = np.arange(Nx) * float(dx) / 1000.0
z_km = np.arange(Nz) * float(dz) / 1000.0
max_x_km = x_km[-1]
max_z_km = z_km[-1]

# =========================================================================
# 2. PARÂMETROS DA SIMULAÇÃO (Física)
# =========================================================================
dt = np.float32(0.0001)
tempo = 30000

# Pré-calculando o quadrado da velocidade para o Numba
Vp_sq = Vp**2 

# =========================================================================
# 3. CONFIGURAÇÃO DO STAGGERED ADE-PML NA GRADE EXPANDIDA
# =========================================================================
R = 0.0001    
V_max = np.max(Vp)

n_pml = 3.0
d_max = -((n_pml + 1.0) * V_max) / (2.0 * L_pml * dx) * np.log(R)

def get_damping(pos_in_pml, L):
    if pos_in_pml < 0: return 0.0
    if pos_in_pml > L: return d_max
    return d_max * (pos_in_pml / L)**n_pml

sx = np.zeros(Nx, dtype=np.float32)
sz = np.zeros(Nz, dtype=np.float32)
sx_half = np.zeros(Nx, dtype=np.float32)
sz_half = np.zeros(Nz, dtype=np.float32)

for j in range(Nx):
    if j < L_pml: 
        sx[j] = get_damping(L_pml - j, L_pml)
        sx_half[j] = get_damping(L_pml - (j + 0.5), L_pml)
    elif j >= Nx - L_pml: 
        dist_j = j - (Nx - L_pml)
        sx[j] = get_damping(dist_j + 1, L_pml)
        sx_half[j] = get_damping(dist_j + 0.5, L_pml)

for i in range(Nz):
    if i >= Nz - L_pml: 
        dist_i = i - (Nz - L_pml)
        sz[i] = get_damping(dist_i + 1, L_pml)
        sz_half[i] = get_damping(dist_i + 0.5, L_pml)

# =========================================================================
# 4. ALOCAÇÃO DE VARIÁVEIS E FONTE PRÓXIMA À BORDA
# =========================================================================
U  = np.zeros((Nz, Nx, 3), dtype=np.float32)
Qx = np.zeros((Nz, Nx, 2), dtype=np.float32)
Qz = np.zeros((Nz, Nx, 2), dtype=np.float32)

t0 = np.float32(0.1)
s = np.float32(0.02)

# Fonte posicionada perto do canto inferior direito do domínio Marmousi para teste de borda
z0 = 4            
x0 = 6000

p1, p2, p3 = 0, 1, 2
q1, q2 = 0, 1

# =========================================================================
# 5. O MOTOR DE ALTA PERFORMANCE (Numba JIT)
# =========================================================================
@njit(parallel=True, fastmath=True)
def calcular_propagacao_marmousi(U, Qx, Qz, sx, sx_half, sz, sz_half, Vp_sq, dx, dz, dt, p1, p2, p3, q1, q2, Nz, Nx):
    
    # 1. Memória Horizontal (Qx)
    for i in prange(1, Nz - 1):
        for j in range(0, Nx - 1):
            du_dx = (U[i, j+1, p2] - U[i, j, p2]) / dx
            sx_val = sx_half[j]
            sz_val = sz[i]
            
            A_minus = 1.0 - (sx_val * dt / 2.0)
            A_plus  = 1.0 + (sx_val * dt / 2.0)
            Qx[i, j, q2] = (A_minus * Qx[i, j, q1] - dt * (sx_val - sz_val) * du_dx) / A_plus

    # 2. Memória Vertical (Qz)
    for i in prange(0, Nz - 1):
        for j in range(1, Nx - 1):
            du_dz = (U[i+1, j, p2] - U[i, j, p2]) / dz
            sx_val = sx[j]
            sz_val = sz_half[i]
            
            A_minus = 1.0 - (sz_val * dt / 2.0)
            A_plus  = 1.0 + (sz_val * dt / 2.0)
            Qz[i, j, q2] = (A_minus * Qz[i, j, q1] - dt * (sz_val - sx_val) * du_dz) / A_plus

    # 3. Atualização da Onda Acústica Principal (U)
    for i in prange(1, Nz - 1):
        for j in range(1, Nx - 1):
            
            dqx_dx = (Qx[i, j, q2] - Qx[i, j-1, q2]) / dx
            dqz_dz = (Qz[i, j, q2] - Qz[i-1, j, q2]) / dz

            d2u_dx2 = (U[i, j+1, p2] - 2.0*U[i, j, p2] + U[i, j-1, p2]) / (dx**2)
            d2u_dz2 = (U[i+1, j, p2] - 2.0*U[i, j, p2] + U[i-1, j, p2]) / (dz**2)

            laplaciano = d2u_dx2 + d2u_dz2 + dqx_dx + dqz_dz
            
            alpha = sx[j] + sz[i]
            beta  = sx[j] * sz[i]

            A_u_minus = 1.0 - (alpha * dt / 2.0)
            A_u_plus  = 1.0 + (alpha * dt / 2.0)

            termo_fonte = Vp_sq[i, j] * laplaciano - beta * U[i, j, p2]

            U[i, j, p3] = (2.0 * U[i, j, p2] - A_u_minus * U[i, j, p1] + (dt**2) * termo_fonte) / A_u_plus

# =========================================================================
# 6. CONFIGURAÇÃO DO GIF E LOOP TEMPORAL
# =========================================================================
arq_gif = 'simulacao_marmousi_padded_pml.gif'
writer = imageio.get_writer(arq_gif, mode='I', duration=0.15, loop=0)

fig, ax = plt.subplots(figsize=(10, 5))

print("===================================================")
print("INICIANDO PROPAGAÇÃO OTIMIZADA COM NUMBA JIT...")
print("===================================================")

inicio_timer = time.time()

# Calculando a posição exata da fronteira do Marmousi em km para as linhas tracejadas
marm_x_start = L_pml * float(dx) / 1000.0
marm_x_end   = (L_pml + Nx_marm) * float(dx) / 1000.0
marm_z_end   = Nz_marm * float(dz) / 1000.0

for n in range(1, tempo + 1):
    t_atual = np.float32(n * dt)
    
    calcular_propagacao_marmousi(U, Qx, Qz, sx, sx_half, sz, sz_half, Vp_sq, dx, dz, dt, p1, p2, p3, q1, q2, Nz, Nx)
    
    # Injeção da Fonte
    f = np.exp(-((t_atual - t0) / s)**2, dtype=np.float32)
    U[z0, x0, p3] += f
    
    # Superfície Livre 
    U[0, :, p3] = 0.0
    
    if n % 250 == 0:
        if n == 250:
            # Plota o domínio expandido inteiro (com as bordas de PML visíveis)
            im = ax.imshow(U[:, :, p3], cmap='jet', vmin=-0.02, vmax=0.05, 
                           extent=[0, max_x_km, max_z_km, 0], aspect='auto')
            
            # Linhas pontilhadas vermelhas mostrando exatamente onde o Marmousi real começa/termina
            ax.axvline(marm_x_start, color='r', linestyle='--', alpha=0.6, linewidth=1.5)
            ax.axvline(marm_x_end, color='r', linestyle='--', alpha=0.6, linewidth=1.5)
            ax.axhline(marm_z_end, color='r', linestyle='--', alpha=0.6, linewidth=1.5)
            
            title_text = ax.set_title(f'Marmousi com PML Externo (t = {t_atual:.4f} s)', fontweight='bold')
            ax.set_xlabel('Distância Total (km)')
            ax.set_ylabel('Profundidade Total (km)')
            fig.tight_layout()
        else:
            im.set_data(U[:, :, p3])
            title_text.set_text(f'Marmousi com PML Externo (t = {t_atual:.4f} s)')
        
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        imagem_matriz = rgba.copy() 
        
        writer.append_data(imagem_matriz)
        print(f"-> Andamento: Passo {n} / {tempo} concluído...")

    p1, p2, p3 = p2, p3, p1
    q1, q2 = q2, q1

writer.close()
plt.close()

tempo_execucao = time.time() - inicio_timer
print(f"\nSimulação finalizada com sucesso em {tempo_execucao:.2f} segundos!")
print(f"Ficheiro guardado como: {arq_gif}")