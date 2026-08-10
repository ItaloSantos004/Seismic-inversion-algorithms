import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from multiprocessing import Pool, cpu_count

#inverte cada coluna
def inversao(args):
    #extraindo os dados necessarios
    id, Pc, Wc, xi, w0, ni, nt2, n_sens = args
    
    print(f"Calculando coluna {id+1}") 
    
    xi2 = xi**2
    lim = [(900, 6000), (1400, 4000)]
    
    vel_id = np.zeros(ni, dtype=np.float32)
    rho_id = np.zeros(ni, dtype=np.float32)
    Zrec = np.zeros((n_sens, ni), dtype=np.float32)
    
    for m in range(n_sens):
        Pinv = np.zeros((ni, nt2), dtype=np.float32)
        Winv = np.zeros((ni, nt2), dtype=np.float32)
        Zinv = np.zeros(ni, dtype=np.float32)
        
        #superficie
        for j in range(0, nt2, 2):
            if j + 2 < nt2:
                Pinv[0, j] = Pc[m, j + 2]
                Winv[0, j] = Wc[m, j + 2]
                
        if Winv[0, 0] != 0:
            Zinv[0] = Pinv[0, 0] / Winv[0, 0]
        else:
            Zinv[0] = 1.0
            
        #recursao layer-peeling
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
            
    #Otimizador para separar velocidade e densidade
    chute_rho = 1000.0 #chute inicial
    chute_c = 1500.0

    for i in range(ni):
        validos = np.abs(Zrec[:, i]) > 1e-5
        
        if np.sum(validos) < 2:
            prec = chute_rho
            crec = chute_c
        else:
            Z_medido = Zrec[validos, i]
            xi3 = xi2[validos]
            
            def min(p):
                rho, c = p
                kz = np.maximum((w0 / c)**2 - xi3, 1e-10)
                Z_teo = (rho * w0) / np.sqrt(kz)
                return np.sum(np.abs(Z_medido - Z_teo)**2)
            
            try:
                res = minimize(min, [chute_rho, chute_c], method='SLSQP', bounds=lim)
                if not res.success:
                    raise RuntimeError("SLSQP não convergiu")
                prec = res.x[0]
                crec = res.x[1]
                
            except:
                def min2(p):
                    rho, c = p
                    pen = 0.0 # penalidade
                    if rho < lim[0][0]: pen += 1e10
                    if rho > lim[0][1]: pen += 1e10
                    if c < lim[1][0]: pen += 1e10
                    if c > lim[1][1]: pen += 1e10
                    
                    kz = np.maximum((w0 / c)**2 - xi3, 1e-10)
                    Z_teo = (rho * w0) / np.sqrt(kz)
                    return np.sum(np.abs(Z_medido - Z_teo)**2) + pen
                
                op = {'maxiter': 1000, 'maxfev': 1000}
                res2 = minimize(min2, [chute_rho, chute_c], method='Nelder-Mead', options=op)
                prec = res2.x[0]
                crec = res2.x[1]
        
        chute_rho = prec
        chute_c = crec
        vel_id[i] = crec
        rho_id[i] = prec 

    return id, vel_id, rho_id 

if __name__ == '__main__':
    #extraindo os dados para inversão
    dados = np.load('dados_marmousi_P_W_completos.npz')
    P_real = dados['P_real']
    W_real = dados['W_real']
    Vel_idx = dados['Vp_real']
    Rho_idx = dados['Rho_real']

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

    for id in range(Nx_idx):
        for j in range(nt):
            P_canal[id, :, j] = np.real(np.fft.fftshift(np.fft.fft(P_real[id, :, j])))
            W_canal[id, :, j] = np.real(np.fft.fftshift(np.fft.fft(W_real[id, :, j])))

    print("Iniciando Inversão das colunas")
    
    #organizando os argumentos para a função
    arg = []
    for id in range(Nx_idx):
        arg.append((id, P_canal[id, :, :], W_canal[id, :, :], xi, w0, ni, nt2, n_sens))
     
    num_nucleos = cpu_count()
    print(f"Usando {num_nucleos} núcleos")
    
    Vel_rec = np.zeros((ni, Nx_idx), dtype=np.float32)
    Rho_rec = np.zeros((ni, Nx_idx), dtype=np.float32)
    
    with Pool(processes=num_nucleos) as pool: #roda em paralelo
        resultados = pool.map(inversao, arg)
        
    #organiza de volta 
    for id, vel_id, rho_id in resultados:
        Vel_rec[:, id] = vel_id
        Rho_rec[:, id] = rho_id
        
    print("Inversão Concluída")

    #Salvando as matrizes de inversão para comparação futura
    np.savez('matrizes_inversao_completa.npz', Vel_rec=Vel_rec, Rho_rec=Rho_rec)
    print("Matrizes Salvas")

    #visualização velocidade
    fig_vel, axes_vel = plt.subplots(2, 1, figsize=(12, 8))

    min_vel, max_vel = np.min(Vel_idx), np.max(Vel_idx)

    im1 = axes_vel[0].imshow(Vel_idx, aspect='auto', cmap='jet', vmin=min_vel, vmax=max_vel)
    axes_vel[0].set_title('Marmousi Real - Velocidade', fontweight='bold')
    axes_vel[0].set_ylabel('Profundidade')
    fig_vel.colorbar(im1, ax=axes_vel[0], label='Velocidade (m/s)')

    im2 = axes_vel[1].imshow(Vel_rec, aspect='auto', cmap='jet', vmin=min_vel, vmax=max_vel)
    axes_vel[1].set_title('Marmousi Reconstruído - Velocidade', fontweight='bold')
    axes_vel[1].set_xlabel('Colunas (X)')
    axes_vel[1].set_ylabel('Profundidade')
    fig_vel.colorbar(im2, ax=axes_vel[1], label='Velocidade (m/s)')

    fig_vel.tight_layout()
    fig_vel.savefig('marmousi_inversao_velocidade.png', dpi=300)
    print("Imagem de Velocidade Salva")
    plt.close(fig_vel)

    #visualização densidade
    fig_rho, axes_rho = plt.subplots(2, 1, figsize=(12, 8))

    min_rho, max_rho = np.min(Rho_idx), np.max(Rho_idx)

    im3 = axes_rho[0].imshow(Rho_idx, aspect='auto', cmap='viridis', vmin=min_rho, vmax=max_rho)
    axes_rho[0].set_title('Marmousi Real - Densidade', fontweight='bold')
    axes_rho[0].set_ylabel('Profundidade')
    fig_rho.colorbar(im3, ax=axes_rho[0], label='Densidade (kg/m³)')

    im4 = axes_rho[1].imshow(Rho_rec, aspect='auto', cmap='viridis', vmin=min_rho, vmax=max_rho)
    axes_rho[1].set_title('Marmousi Reconstruído - Densidade', fontweight='bold')
    axes_rho[1].set_xlabel('Colunas (X)')
    axes_rho[1].set_ylabel('Profundidade')
    fig_rho.colorbar(im4, ax=axes_rho[1], label='Densidade (kg/m³)')

    fig_rho.tight_layout()
    fig_rho.savefig('marmousi_inversao_densidade.png', dpi=300)
    print("Imagem de Densidade Salva")
    plt.close(fig_rho)