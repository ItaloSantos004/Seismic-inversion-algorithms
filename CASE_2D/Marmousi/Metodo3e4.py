import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from multiprocessing import Pool, cpu_count
import scipy.io as sio

# Função para cada coluna adaptada para receber o alpha
def inversao(args):
    id, Pc, Wc, xi, w0, ni, nt2, n_sens, alpha = args

    xi2 = xi**2
    lim = [(900, 6000), (1400, 4000)]
    rho_min, rho_max = lim[0]
    c_min, c_max = lim[1]
    
    Z_min = rho_min * c_min
    Z_max = rho_max * c_max
    
    vel_id = np.zeros(ni, dtype=np.float32)
    rho_id = np.zeros(ni, dtype=np.float32)
    Z0_id = np.zeros(ni, dtype=np.float32)
    Zrec = np.zeros((n_sens, ni), dtype=np.float32)
    
    for m in range(n_sens):
        Pinv = np.zeros((ni, nt2), dtype=np.float32)
        Winv = np.zeros((ni, nt2), dtype=np.float32)
        Zinv = np.zeros(ni, dtype=np.float32)
        
        # Superfície
        for j in range(0, nt2, 2):
            if j + 2 < nt2:
                Pinv[0, j] = Pc[m, j + 2]
                Winv[0, j] = Wc[m, j + 2]
                
        if Winv[0, 0] != 0 and np.isfinite(Winv[0, 0]):
            Zinv[0] = Pinv[0, 0] / Winv[0, 0]
        else:
            Zinv[0] = Z_min
            
        Zinv[0] = np.clip(Zinv[0], Z_min, Z_max)
            
        # Recursão Layer-Peeling
        for i in range(1, ni):
            Zi = Zinv[i - 1]
            if Zi == 0 or not np.isfinite(Zi):
                Zi = Z_min

            for j in range(i, nt2 - i, 2):
                a = Winv[i-1, j-1] + Winv[i-1, j+1]
                b = Winv[i-1, j-1] - Winv[i-1, j+1]
                c_calc = Pinv[i-1, j-1] + Pinv[i-1, j+1]
                d = Pinv[i-1, j-1] - Pinv[i-1, j+1]
                
                if np.isfinite(Zi) and Zi != 0:
                    Winv[i, j] = 0.5 * (a + d / Zi)
                    Pinv[i, j] = 0.5 * (Zi * b + c_calc)
                    
            if Winv[i, i] != 0 and np.isfinite(Winv[i, i]):
                Znovo = Pinv[i, i] / Winv[i, i]
                Zinv[i] = np.clip(Znovo, Z_min, Z_max) if np.isfinite(Znovo) else Zinv[i-1]
            else:
                Zinv[i] = Zinv[i-1]
                
        for i in range(ni):
            Zrec[m, i] = Zinv[i]
            
    # Otimizador Mínimos Quadrados Reparametrizado
    chute_rho = (rho_min + rho_max) / 2.0 
    chute_c = (c_min + c_max) / 2.0

    for i in range(ni):
        validos = np.abs(Zrec[:, i]) > 1e-5
        
        if np.sum(validos) < 2:
            prec = chute_rho
            crec = chute_c
            Z0_rec = prec * crec
        else:
            Z_medido = Zrec[validos, i]
            xi3 = xi2[validos]
            
            escala_Z = np.median(np.abs(Z_medido))
            if not np.isfinite(escala_Z) or escala_Z < 1e-12:
                escala_Z = 1.0

            Z0_min = rho_min * c_min
            Z0_max = rho_max * c_max
            chute_Z0 = chute_rho * chute_c

            # Pesos Gaussianos dinâmicos baseados no alpha do loop
            xi_valid = xi[validos]
            xi_max = np.max(np.abs(xi))
            
            #pesos_gauss = 0.10 + 0.90 * (1.0 - np.exp(- (xi_valid / (alpha * xi_max))**2))
            pesos_gauss = np.exp(- (xi_valid / (alpha * xi_max))**2)

            def residuos_reparam(p):
                Z0, c_opt = p
                rho_virtual = Z0 / c_opt
                kz2 = np.maximum((w0 / c_opt)**2 - xi3, 1e-10)
                kz = np.sqrt(kz2)
                Z_teo = (rho_virtual * w0) / kz
                if not np.all(np.isfinite(Z_teo)):
                    Z_teo = np.nan_to_num(Z_teo, nan=0.0, posinf=Z_max, neginf=-Z_max)
                
                erro_puro = (Z_medido - Z_teo) / escala_Z
                return erro_puro * pesos_gauss 
            
            try:
                res = least_squares(
                    residuos_reparam,
                    x0=[np.clip(chute_Z0, Z0_min, Z0_max), np.clip(chute_c, c_min, c_max)],
                    method='trf',
                    bounds=([Z0_min, c_min], [Z0_max, c_max]),
                    max_nfev=300
                )
                
                Z0_rec = res.x[0]
                crec = res.x[1]
                prec = Z0_rec / crec
                
                if not np.isfinite(prec) or not np.isfinite(crec):
                    raise RuntimeError("Solução inválida")

            except Exception:
                prec = chute_rho
                crec = chute_c
                Z0_rec = prec * crec

        prec = np.clip(prec, rho_min, rho_max)
        crec = np.clip(crec, c_min, c_max)
        Z0_rec = np.clip(Z0_rec, Z_min, Z_max)
        
        chute_rho = prec
        chute_c = crec
        
        vel_id[i] = crec
        rho_id[i] = prec 
        Z0_id[i] = Z0_rec

    return id, vel_id, rho_id, Z0_id 


if __name__ == '__main__':
    print("Carregando dados PW")
    dados = np.load('dados_marmousi_P_W_completos.npz')
    P_real = dados['P_real']
    W_real = dados['W_real']
    
    Nx_idx, n_sens, nt = P_real.shape
    ni = 2800                  
    nt2 = nt 
    dx = 50.0
    w0 = 2 * np.pi * 50
    xi = (2 * np.pi / (n_sens * dx)) * np.arange(-(n_sens//2), (n_sens//2) + 1, dtype=np.float32)

    SNR_dB = 20
    N_tiros = 5 # Conforme a tabela que você mostrou
    fator_ruido = 10.0 ** (-SNR_dB / 20.0)
    num_nucleos = cpu_count()

    print("Carregando dados originais para cálculo de erro")
    dados_marmousi = sio.loadmat('marmousi_matrizes.mat')
    vm = np.array(dados_marmousi['Vp'], dtype=np.float32)[:ni, :]
    rhom = np.array(dados_marmousi['Rho'], dtype=np.float32)[:ni, :]

    sig = 3.0
    vm_suav = gaussian_filter(vm, sigma=sig).astype(np.float32)
    rhom_suav = gaussian_filter(rhom, sigma=sig).astype(np.float32)
    Zm_suav = rhom_suav * vm_suav

    dxm = 1.25
    nxm = vm.shape[1]
    xm = np.arange(nxm) * dxm
    xinv = np.arange(Nx_idx) * dx 

    # ========================================================
    # LOOP DE VARREDURA DOS ALPHAS
    # ========================================================
    lista_alphas = [1.5, 2.0]
    
    for alpha in lista_alphas:
        print("\n" + "="*50)
        print(f"Iniciando inversão com alpha = {alpha}")
        print("="*50)
        
        # Semente resetada a cada alpha para garantir que todos os alphas 
        # testem exatamentes os mesmos dados ruidosos
        np.random.seed(42)
        
        # Matrizes para acumular os resultados das inversões de todos os tiros
        Vel_rec_acum = np.zeros((ni, Nx_idx), dtype=np.float32)
        Rho_rec_acum = np.zeros((ni, Nx_idx), dtype=np.float32)
        Z0_rec_acum = np.zeros((ni, Nx_idx), dtype=np.float32) 

        # --- LOOP DOS TIROS (Ruído gerado individualmente por tiro) ---
        for tiro in range(1, N_tiros + 1):
            print(f"  -> Injetando ruído e processando tiro {tiro}/{N_tiros}...")
            
            P_noisy = np.zeros_like(P_real, dtype=np.float32)
            W_noisy = np.zeros_like(W_real, dtype=np.float32)
            
            # Adicionando ruído cru ao tiro atual
            for col in range(Nx_idx):
                rms_P = np.sqrt(np.mean(P_real[col]**2))
                rms_W = np.sqrt(np.mean(W_real[col]**2))
                
                P_noisy[col] = P_real[col] + (fator_ruido * rms_P) * np.random.randn(*P_real[col].shape)
                W_noisy[col] = W_real[col] + (fator_ruido * rms_W) * np.random.randn(*W_real[col].shape)

            # Aplicando a FFT no tiro ruidoso atual
            P_canal = np.zeros_like(P_noisy, dtype=np.float32)
            W_canal = np.zeros_like(W_noisy, dtype=np.float32)
            
            for id in range(Nx_idx):
                for j in range(nt):
                    P_canal[id, :, j] = np.real(np.fft.fftshift(np.fft.fft(P_noisy[id, :, j])))
                    W_canal[id, :, j] = np.real(np.fft.fftshift(np.fft.fft(W_noisy[id, :, j])))

            # Preparando argumentos para o Pool deste tiro
            arg = []
            for id in range(Nx_idx):
                arg.append((id, P_canal[id, :, :], W_canal[id, :, :], xi, w0, ni, nt2, n_sens, alpha))
            
            # Rodando a inversão em paralelo para este tiro
            with Pool(processes=num_nucleos) as pool:
                resultados = pool.map(inversao, arg)
            
            # Acumulando os resultados do tiro
            for id, vel_id, rho_id, Z0_id in resultados:
                Vel_rec_acum[:, id] += vel_id
                Rho_rec_acum[:, id] += rho_id
                Z0_rec_acum[:, id] += Z0_id

        # --- FIM DO LOOP DE TIROS ---
        
        # Tirando a Média dos Resultados PÓS-Inversão
        print(f"Calculando médias dos resultados e interpolando (alpha={alpha})...")
        Vel_rec = Vel_rec_acum / N_tiros
        Rho_rec = Rho_rec_acum / N_tiros
        Z0_rec_media = Z0_rec_acum / N_tiros

        # Interpolação para malha original
        vint = np.zeros((ni, nxm), dtype=np.float32) 
        rhoint = np.zeros((ni, nxm), dtype=np.float32)
        Zint = np.zeros((ni, nxm), dtype=np.float32) 

        for i in range(ni):
            int_v = interp1d(xinv, Vel_rec[i, :], kind='cubic', fill_value='extrapolate')
            vint[i, :] = int_v(xm)
            
            int_rho = interp1d(xinv, Rho_rec[i, :], kind='cubic', fill_value='extrapolate')
            rhoint[i, :] = int_rho(xm)
            
            int_Z = interp1d(xinv, Z0_rec_media[i, :], kind='cubic', fill_value='extrapolate')
            Zint[i, :] = int_Z(xm)

        # Cálculo de Erros
        abs_erro_v = np.abs(vint - vm_suav)
        rel_erro_v = (abs_erro_v / np.maximum(vm_suav, 1e-10)) * 100

        abs_erro_rho = np.abs(rhoint - rhom_suav)
        rel_erro_rho = (abs_erro_rho / np.maximum(rhom_suav, 1e-10)) * 100

        abs_erro_Z = np.abs(Zint - Zm_suav)
        rel_erro_Z = (abs_erro_Z / np.maximum(Zm_suav, 1e-10)) * 100

        mean_erro_v, median_erro_v, max_erro_v = np.mean(rel_erro_v), np.median(rel_erro_v), np.max(rel_erro_v)
        mean_erro_rho, median_erro_rho, max_erro_rho = np.mean(rel_erro_rho), np.median(rel_erro_rho), np.max(rel_erro_rho)
        mean_erro_Z, median_erro_Z, max_erro_Z = np.mean(rel_erro_Z), np.median(rel_erro_Z), np.max(rel_erro_Z)
        
        nome_arquivo_txt = f'estatisticas2_{N_tiros}_alpha_{alpha}_SNR{SNR_dB}.txt'
        with open(nome_arquivo_txt, 'w', encoding='utf-8') as f:
            f.write(f"Resultados da Inversão - alpha = {alpha} | SNR: {SNR_dB} dB | Tiros: {N_tiros}\n")
            f.write("-" * 80 + "\n")
            f.write(f"VELOCIDADE -> Erro Médio: {mean_erro_v:.2f}% | Mediana: {median_erro_v:.2f}% | Máximo: {max_erro_v:.2f}%\n")
            f.write(f"DENSIDADE  -> Erro Médio: {mean_erro_rho:.2f}% | Mediana: {median_erro_rho:.2f}% | Máximo: {max_erro_rho:.2f}%\n")
            f.write(f"IMPEDÂNCIA -> Erro Médio: {mean_erro_Z:.2f}%   | Mediana: {median_erro_Z:.2f}% | Máximo: {max_erro_Z:.2f}%\n")
        
        xkm1, xkm2, zkm = xm[0] / 1000, xm[-1] / 1000, (ni * dxm) / 1000
        eixo = [xkm1, xkm2, zkm, 0] 

        # ========================================================
        # SALVAR GRÁFICOS
        # ========================================================
        print(f"Gerando gráficos para alpha={alpha}...\n")
        
        # ==================== VELOCIDADE ====================
        fig_v, axes_v = plt.subplots(3, 1, figsize=(14, 12))
        min_vel, max_vel = np.min(vm_suav), np.max(vm_suav)

        im1 = axes_v[0].imshow(vm_suav, aspect='auto', cmap='jet', vmin=min_vel, vmax=max_vel, extent=eixo)
        axes_v[0].set_title('Marmousi Suavizado - Velocidade', fontweight='bold')
        axes_v[0].set_ylabel('Profundidade (km)')
        fig_v.colorbar(im1, ax=axes_v[0], label='Velocidade (m/s)')

        im2 = axes_v[1].imshow(vint, aspect='auto', cmap='jet', vmin=min_vel, vmax=max_vel, extent=eixo)
        axes_v[1].set_title(f'Inversão Interpolada - Velocidade (alpha={alpha} | SNR {SNR_dB}dB)', fontweight='bold')
        axes_v[1].set_ylabel('Profundidade (km)')
        fig_v.colorbar(im2, ax=axes_v[1], label='Velocidade (m/s)')

        im3 = axes_v[2].imshow(rel_erro_v, aspect='auto', cmap='turbo', vmin=0, vmax=100, extent=eixo)
        axes_v[2].set_title('Erro (%)', fontweight='bold')
        axes_v[2].set_xlabel('X (km)')
        axes_v[2].set_ylabel('Profundidade (km)')
        fig_v.colorbar(im3, ax=axes_v[2], label='Erro (%)')

        fig_v.tight_layout()
        fig_v.savefig(f'erro2_velocidade_{N_tiros}_alpha_{alpha}_SNR{SNR_dB}.png', dpi=300)
        plt.close(fig_v)

        # ==================== DENSIDADE ====================
        fig_r, axes_r = plt.subplots(3, 1, figsize=(14, 12))
        min_rho, max_rho = np.min(rhom_suav), np.max(rhom_suav)

        im4 = axes_r[0].imshow(rhom_suav, aspect='auto', cmap='viridis', vmin=min_rho, vmax=max_rho, extent=eixo)
        axes_r[0].set_title('Marmousi Suavizado - Densidade', fontweight='bold')
        axes_r[0].set_ylabel('Profundidade (km)')
        fig_r.colorbar(im4, ax=axes_r[0], label='Densidade (kg/m³)')

        im5 = axes_r[1].imshow(rhoint, aspect='auto', cmap='viridis', vmin=min_rho, vmax=max_rho, extent=eixo)
        axes_r[1].set_title(f'Inversão Interpolada - Densidade (alpha={alpha} | SNR {SNR_dB}dB)', fontweight='bold')
        axes_r[1].set_ylabel('Profundidade (km)')
        fig_r.colorbar(im5, ax=axes_r[1], label='Densidade (kg/m³)')

        im6 = axes_r[2].imshow(rel_erro_rho, aspect='auto', cmap='turbo', vmin=0, vmax=100, extent=eixo)
        axes_r[2].set_title('Erro (%)', fontweight='bold')
        axes_r[2].set_xlabel('X (km)')
        axes_r[2].set_ylabel('Profundidade (km)')
        fig_r.colorbar(im6, ax=axes_r[2], label='Erro (%)')

        fig_r.tight_layout()
        fig_r.savefig(f'erro2_densidade_{N_tiros}_alpha_{alpha}_SNR{SNR_dB}.png', dpi=300)
        plt.close(fig_r)

        # ==================== IMPEDÂNCIA ====================
        fig_z, axes_z = plt.subplots(3, 1, figsize=(14, 12))
        min_Z, max_Z = np.min(Zm_suav), np.max(Zm_suav)

        im7 = axes_z[0].imshow(Zm_suav, aspect='auto', cmap='magma', vmin=min_Z, vmax=max_Z, extent=eixo)
        axes_z[0].set_title('Marmousi Suavizado - Impedância', fontweight='bold')
        axes_z[0].set_ylabel('Profundidade (km)')
        fig_z.colorbar(im7, ax=axes_z[0], label='Impedância (kg/(m²s))')

        im8 = axes_z[1].imshow(Zint, aspect='auto', cmap='magma', vmin=min_Z, vmax=max_Z, extent=eixo)
        axes_z[1].set_title(f'Inversão Interpolada - Impedância (alpha={alpha} | SNR {SNR_dB}dB)', fontweight='bold')
        axes_z[1].set_ylabel('Profundidade (km)')
        fig_z.colorbar(im8, ax=axes_z[1], label='Impedância (kg/(m²s))')

        im9 = axes_z[2].imshow(rel_erro_Z, aspect='auto', cmap='turbo', vmin=0, vmax=100, extent=eixo)
        axes_z[2].set_title('Erro (%)', fontweight='bold')
        axes_z[2].set_xlabel('X (km)')
        axes_z[2].set_ylabel('Profundidade (km)')
        fig_z.colorbar(im9, ax=axes_z[2], label='Erro (%)')

        fig_z.tight_layout()
        fig_z.savefig(f'erro2_impedancia_{N_tiros}_alpha_{alpha}_SNR{SNR_dB}.png', dpi=300)
        plt.close(fig_z)

    print("Varredura de alphas totalmente concluída!")