import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from numba import njit, prange

#grade para teste
Nz, Nx = 250, 500
dx = np.float32(10.0)
dz = np.float32(10.0)

Vp = np.full((Nz, Nx), 2500.0, dtype=np.float32) #mesma velocidade para teste

print("Iniciando e criando a função")

#parametros
dt = np.float32(0.001)
tempo = 1200

#perfil de amortecimento
L_pml = 100
R = 0.0001
V_max = np.max(Vp)
n_pml = 3.0

d_max = -((n_pml + 1.0) * V_max) / (2.0 * L_pml * dx) * np.log(R)

#calcula o d_max
def get_damping(pos_in_pml, L):
    if pos_in_pml < 0: return 0.0
    if pos_in_pml > L: return d_max
    return d_max * (pos_in_pml / L)**n_pml

#vetores U
sx = np.zeros(Nx, dtype=np.float32)
sz = np.zeros(Nz, dtype=np.float32)

#vetores Q_i 
sx_half = np.zeros(Nx, dtype=np.float32)
sz_half = np.zeros(Nz, dtype=np.float32)

for j in range(Nx):
    if j < L_pml: #parede esquerda
        sx[j] = get_damping(L_pml - j, L_pml)
        sx_half[j] = get_damping(L_pml - (j + 0.5), L_pml)
    elif j >= Nx - L_pml: #parede direita
        sx[j] = get_damping(j - (Nx - L_pml - 1), L_pml)
        sx_half[j] = get_damping((j + 0.5) - (Nx - L_pml - 1), L_pml)

for i in range(Nz):
    if i >= Nz - L_pml: #fundo
        sz[i] = get_damping(i - (Nz - L_pml - 1), L_pml)
        sz_half[i] = get_damping((i + 0.5) - (Nz - L_pml - 1), L_pml)

#variaveis
U = np.zeros((Nz, Nx, 3), dtype=np.float32)
Qx = np.zeros((Nz, Nx, 2), dtype=np.float32)
Qz = np.zeros((Nz, Nx, 2), dtype=np.float32)

#dados da fonte
t0 = np.float32(0.05)
s  = np.float32(0.015)
z0 = 10            
x0 = int(Nx / 2)  

#indices
p1, p2, p3 = 0, 1, 2 
q1, q2 = 0, 1        

#função para calcular a propagação
@njit(parallel=True, fastmath=True)
def calcular_staggered_pml(U, Qx, Qz, sx, sx_half, sz, sz_half, Vp, dx, dz, dt, p1, p2, p3, q1, q2, Nz, Nx):
    
    #Atualizamos Qx
    for i in prange(1, Nz - 1):
        for j in range(0, Nx - 1):
            du_dx = (U[i, j+1, p2] - U[i, j, p2]) / dx
            sx_val = sx_half[j]
            sz_val = sz[i]
            
            A_minus = 1.0 - (sx_val * dt / 2.0)
            A_plus  = 1.0 + (sx_val * dt / 2.0)
            Qx[i, j, q2] = (A_minus * Qx[i, j, q1] - dt * (sx_val - sz_val) * du_dx) / A_plus

    #Atualizamos Qz
    for i in prange(0, Nz - 1):
        for j in range(1, Nx - 1):
            du_dz = (U[i+1, j, p2] - U[i, j, p2]) / dz
            sx_val = sx[j]
            sz_val = sz_half[i]
            
            A_minus = 1.0 - (sz_val * dt / 2.0)
            A_plus  = 1.0 + (sz_val * dt / 2.0)
            Qz[i, j, q2] = (A_minus * Qz[i, j, q1] - dt * (sz_val - sx_val) * du_dz) / A_plus

    #Atualizamos a onda principal U
    for i in prange(1, Nz - 1):
        for j in range(1, Nx - 1):
            
            #derivadas de Q_i
            dqx_dx = (Qx[i, j, q2] - Qx[i, j-1, q2]) / dx
            dqz_dz = (Qz[i, j, q2] - Qz[i-1, j, q2]) / dz

            d2u_dx2 = (U[i, j+1, p2] - 2.0*U[i, j, p2] + U[i, j-1, p2]) / (dx**2)
            d2u_dz2 = (U[i+1, j, p2] - 2.0*U[i, j, p2] + U[i-1, j, p2]) / (dz**2)

            laplaciano = d2u_dx2 + d2u_dz2 + dqx_dx + dqz_dz
            
            alpha = sx[j] + sz[i]
            beta  = sx[j] * sz[i]

            A_u_minus = 1.0 - (alpha * dt / 2.0)
            A_u_plus  = 1.0 + (alpha * dt / 2.0)

            termo_fonte = (Vp[i, j]**2) * laplaciano - beta * U[i, j, p2]

            U[i, j, p3] = (2.0 * U[i, j, p2] - A_u_minus * U[i, j, p1] + (dt**2) * termo_fonte) / A_u_plus

#loop temporando e salvando o .gif
arq_gif = 'teste_pml_ajustes.gif'
writer = imageio.get_writer(arq_gif, mode='I', duration=0.1, loop=0)

fig, ax = plt.subplots(figsize=(6, 6))

print("Simulação e gravando o .gif")


for n in range(1, tempo + 1):
    t_atual = np.float32(n * dt)
    
    calcular_staggered_pml(U, Qx, Qz, sx, sx_half, sz, sz_half, Vp, dx, dz, dt, p1, p2, p3, q1, q2, Nz, Nx)
    
    #injeção da fonte gaussiana
    f = np.exp(-((t_atual - t0) / s)**2, dtype=np.float32)
    U[z0, x0, p3] += f
    
    #condição de dirichlet na superficie
    U[0, :, p3] = 0.0
    
    if n % 20 == 0:
        if n == 20:
            im = ax.imshow(U[:, :, p3], cmap='jet', vmin=-0.02, vmax=0.08)
            ax.axvline(L_pml, color='r', linestyle='--', alpha=0.5)
            ax.axvline(Nx - L_pml, color='r', linestyle='--', alpha=0.5)
            ax.axhline(Nz - L_pml, color='r', linestyle='--', alpha=0.5)
            title_text = ax.set_title(f'Teste PML (Tempo = {t_atual:.3f} s)', fontweight='bold')
            fig.tight_layout()
        else:
            im.set_data(U[:, :, p3])
            title_text.set_text(f'Teste PML (Tempo = {t_atual:.3f} s)')
        
        fig.canvas.draw()
        imagem_matriz = np.asarray(fig.canvas.buffer_rgba()).copy()
        writer.append_data(imagem_matriz)
        print(f"Passo {n} / {tempo}")

    p1, p2, p3 = p2, p3, p1
    q1, q2 = q2, q1

writer.close()
plt.close()
print("Finalizado")