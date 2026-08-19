import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from multiprocessing import Pool, cpu_count
import scipy.io as sio

# Função para cada coluna
def inversao(args):
    id, Pc, Wc, xi, w0, ni, nt2, n_sens = args

    print(f"Calculando coluna {id+1}")
    
    xi2 = xi**2
    lim = [(900, 6000), (1400, 4000)]
    rho_min, rho_max = lim[0]
    c_min, c_max = lim[1]
    Z_min = rho_min * c_min
    Z_max = rho_max * c_max
    
    vel_id = np.zeros(ni, dtype=np.float32)
    rho_id = np.zeros(ni, dtype=np.float32)
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
                c = Pinv[i-1, j-1] + Pinv[i-1, j+1]
                d = Pinv[i-1, j-1] - Pinv[i-1, j+1]
                
                if np.isfinite(Zi) and Zi != 0:
                    Winv[i, j] = 0.5 * (a + d / Zi)
                    Pinv[i, j] = 0.5 * (Zi * b + c)
                    
            if Winv[i, i] != 0 and np.isfinite(Winv[i, i]):
                Znovo = Pinv[i, i] / Winv[i, i]
                Zinv[i] = np.clip(Znovo, Z_min, Z_max) if np.isfinite(Znovo) else Zinv[i-1]
            else:
                Zinv[i] = Zinv[i-1]
                
        for i in range(ni):
            Zrec[m, i] = Zinv[i]
            
    # Otimizador Mínimos Quadrados (TRF) para separar velocidade e densidade
    chute_rho = (rho_min + rho_max) / 2.0 
    chute_c = (c_min + c_max) / 2.0

    for i in range(ni):
        validos = np.abs(Zrec[:, i]) > 1e-5
        
        if np.sum(validos) < 2:
            prec = chute_rho
            crec = chute_c
        else:
            Z_medido = Zrec[validos, i]
            xi3 = xi2[validos]
            
            # Normalização da escala para estabilidade numérica com ruído
            escala_Z = np.median(np.abs(Z_medido))
            if not np.isfinite(escala_Z) or escala_Z < 1e-12:
                escala_Z = 1.0

            def residuos(p):
                rho, c = p
                kz2 = np.maximum((w0 / c)**2 - xi3, 1e-10)
                kz = np.sqrt(kz2)
                Z_teo = (rho * w0) / kz
                if not np.all(np.isfinite(Z_teo)):
                    Z_teo = np.nan_to_num(Z_teo, nan=0.0, posinf=Z_max, neginf=-Z_max)
                return (Z_medido - Z_teo) / escala_Z
            
            try:
                res = least_squares(
                    residuos,
                    x0=[np.clip(chute_rho, rho_min, rho_max), np.clip(chute_c, c_min, c_max)],
                    method='trf',
                    bounds=([rho_min, c_min], [rho_max, c_max]),
                    max_nfev=300
                )
                prec = res.x[0]
                crec = res.x[1]
                
                if not np.isfinite(prec) or not np.isfinite(crec):
                    raise RuntimeError("Solução inválida")

            except Exception:
                prec = chute_rho
                crec = chute_c

        prec = np.clip(prec, rho_min, rho_max)
        crec = np.clip(crec, c_min, c_max)
        
        chute_rho = prec
        chute_c = crec
        vel_id[i] = crec
        rho_id[i] = prec 

    return id, vel_id, rho_id 

if __name__ == '__main__':
    # Carregando os dados sinteticos
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

    # Injetando ruído

    SNR_dB = 20
    N_tiros = 1
    
    fator_ruido = 10.0 ** (-SNR_dB / 20.0)

    Vel_acumulada = np.zeros((ni, Nx_idx), dtype=np.float32)
    Rho_acumulada = np.zeros((ni, Nx_idx), dtype=np.float32)

    num_nucleos = cpu_count()
    print(f"Usando {num_nucleos} núcleos. Iniciando loop de tiros com SNR = {SNR_dB} dB")
    
    with Pool(processes=num_nucleos) as pool:
        for tiro in range(1, N_tiros + 1):
            print(f"Loop: {tiro}/{N_tiros}")
            
            P_noisy = np.zeros_like(P_real, dtype=np.float32)
            W_noisy = np.zeros_like(W_real, dtype=np.float32)
            
            # Injeção de ruído em cada coluna
            for col in range(Nx_idx):
                rms_P = np.sqrt(np.mean(P_real[col]**2))
                rms_W = np.sqrt(np.mean(W_real[col]**2))
                
                ruido_P = np.random.randn(*P_real[col].shape).astype(np.float32)
                ruido_W = np.random.randn(*W_real[col].shape).astype(np.float32)
                
                P_noisy[col] = P_real[col] + (fator_ruido * rms_P) * ruido_P
                W_noisy[col] = W_real[col] + (fator_ruido * rms_W) * ruido_W

            # Transformada de Fourier
            P_canal = np.zeros_like(P_noisy, dtype=np.float32)
            W_canal = np.zeros_like(W_noisy, dtype=np.float32)
            
            for id in range(Nx_idx):
                for j in range(nt):
                    P_canal[id, :, j] = np.real(np.fft.fftshift(np.fft.fft(P_noisy[id, :, j])))
                    W_canal[id, :, j] = np.real(np.fft.fftshift(np.fft.fft(W_noisy[id, :, j])))
                    
            arg = []
            for id in range(Nx_idx):
                arg.append((id, P_canal[id, :, :], W_canal[id, :, :], xi, w0, ni, nt2, n_sens))
            
            resultados = pool.map(inversao, arg)
            
            for id, vel_id, rho_id in resultados:
                Vel_acumulada[:, id] += vel_id
                Rho_acumulada[:, id] += rho_id

    print("Tirando a média")
    Vel_rec = Vel_acumulada / N_tiros
    Rho_rec = Rho_acumulada / N_tiros

    print("Carregando dados originais")
    dados_marmousi = sio.loadmat('marmousi_matrizes.mat')
    vm = np.array(dados_marmousi['Vp'], dtype=np.float32)[:ni, :]
    rhom = np.array(dados_marmousi['Rho'], dtype=np.float32)[:ni, :]

    print("Suavizando")
    sig = 3.0
    vm_suav = gaussian_filter(vm, sigma=sig).astype(np.float32)
    rhom_suav = gaussian_filter(rhom, sigma=sig).astype(np.float32)

    dxm = 1.25
    nxm = vm.shape[1]

    print("Interpolando matrizes invertidas")
    xinv  = np.arange(Nx_idx) * dx 
    xm = np.arange(nxm) * dxm

    vint = np.zeros((ni, nxm), dtype=np.float32) 
    rhoint = np.zeros((ni, nxm), dtype=np.float32)

    for i in range(ni):
        int_v = interp1d(xinv, Vel_rec[i, :], kind='cubic', fill_value='extrapolate')
        vint[i, :] = int_v(xm)
        
        int_rho = interp1d(xinv, Rho_rec[i, :], kind='cubic', fill_value='extrapolate')
        rhoint[i, :] = int_rho(xm)

    print("Calculando matrizes de erro e impedância")
    # Cálculo das Impedâncias
    Zm_suav = rhom_suav * vm_suav
    Zint = rhoint * vint

    abs_erro_v = np.abs(vint - vm_suav)
    rel_erro_v = (abs_erro_v / np.maximum(vm_suav, 1e-10)) * 100

    abs_erro_rho = np.abs(rhoint - rhom_suav)
    rel_erro_rho = (abs_erro_rho / np.maximum(rhom_suav, 1e-10)) * 100

    abs_erro_Z = np.abs(Zint - Zm_suav)
    rel_erro_Z = (abs_erro_Z / np.maximum(Zm_suav, 1e-10)) * 100

    # Cálculo do Erro Médio, Mediana e Máximo
    mean_erro_v, median_erro_v, max_erro_v = np.mean(rel_erro_v), np.median(rel_erro_v), np.max(rel_erro_v)
    mean_erro_rho, median_erro_rho, max_erro_rho = np.mean(rel_erro_rho), np.median(rel_erro_rho), np.max(rel_erro_rho)
    mean_erro_Z, median_erro_Z, max_erro_Z = np.mean(rel_erro_Z), np.median(rel_erro_Z), np.max(rel_erro_Z)

    print(f"VELOCIDADE -> Médio: {mean_erro_v:.2f}% | Mediana: {median_erro_v:.2f}% | Máximo: {max_erro_v:.2f}%")
    print(f"DENSIDADE  -> Médio: {mean_erro_rho:.2f}% | Mediana: {median_erro_rho:.2f}% | Máximo: {max_erro_rho:.2f}%")
    print(f"IMPEDÂNCIA -> Médio: {mean_erro_Z:.2f}% | Mediana: {median_erro_Z:.2f}% | Máximo: {max_erro_Z:.2f}%")
    
    # Salvando estatísticas em um arquivo txt
    nome_arquivo_txt = f'estatisticas_erro_{N_tiros}_SNR{SNR_dB}.txt'
    with open(nome_arquivo_txt, 'w', encoding='utf-8') as f:
        f.write(f"Resultados da Inversão Sísmica - {N_tiros} tiro(s) | SNR: {SNR_dB} dB\n")
        f.write("-" * 65 + "\n")
        f.write(f"VELOCIDADE -> Erro Médio: {mean_erro_v:.2f}% | Mediana: {median_erro_v:.2f}% | Máximo: {max_erro_v:.2f}%\n")
        f.write(f"DENSIDADE  -> Erro Médio: {mean_erro_rho:.2f}% | Mediana: {median_erro_rho:.2f}% | Máximo: {max_erro_rho:.2f}%\n")
        f.write(f"IMPEDÂNCIA -> Erro Médio: {mean_erro_Z:.2f}%   | Mediana: {median_erro_Z:.2f}% | Máximo: {max_erro_Z:.2f}%\n")
    print(f"Estatísticas de erro salvas no arquivo: {nome_arquivo_txt}")

    xkm1 = xm[0] / 1000
    xkm2 = xm[-1] / 1000
    zkm = (ni * dxm) / 1000
    eixo = [xkm1, xkm2, zkm, 0] 

    # Visualização - Velocidade
    print("Gerando Imagem - Velocidade")
    fig_v, axes_v = plt.subplots(3, 1, figsize=(14, 12))
    min_vel, max_vel = np.min(vm_suav), np.max(vm_suav)

    im1 = axes_v[0].imshow(vm_suav, aspect='auto', cmap='jet', vmin=min_vel, vmax=max_vel, extent=eixo)
    axes_v[0].set_title('Marmousi Suavizado - Velocidade', fontweight='bold')
    axes_v[0].set_ylabel('Profundidade (km)')
    fig_v.colorbar(im1, ax=axes_v[0], label='Velocidade (m/s)')

    im2 = axes_v[1].imshow(vint, aspect='auto', cmap='jet', vmin=min_vel, vmax=max_vel, extent=eixo)
    axes_v[1].set_title(f'Inversão com Interpolação - Velocidade (SNR {SNR_dB}dB)', fontweight='bold')
    axes_v[1].set_ylabel('Profundidade (km)')
    fig_v.colorbar(im2, ax=axes_v[1], label='Velocidade (m/s)')

    im3 = axes_v[2].imshow(rel_erro_v, aspect='auto', cmap='turbo', vmin=0, vmax=100, extent=eixo)
    axes_v[2].set_title('Erro (%)', fontweight='bold')
    axes_v[2].set_xlabel('X (km)')
    axes_v[2].set_ylabel('Profundidade (km)')
    fig_v.colorbar(im3, ax=axes_v[2], label='Erro (%)')

    fig_v.tight_layout()
    fig_v.savefig(f'erro_marmousi_velocidade_{N_tiros}_SNR{SNR_dB}.png', dpi=300)
    plt.close(fig_v)

    # Visualização - Densidade
    print("Gerando Imagem - Densidade")
    fig_r, axes_r = plt.subplots(3, 1, figsize=(14, 12))
    min_rho, max_rho = np.min(rhom_suav), np.max(rhom_suav)

    im4 = axes_r[0].imshow(rhom_suav, aspect='auto', cmap='viridis', vmin=min_rho, vmax=max_rho, extent=eixo)
    axes_r[0].set_title('Marmousi Suavizado - Densidade', fontweight='bold')
    axes_r[0].set_ylabel('Profundidade (km)')
    fig_r.colorbar(im4, ax=axes_r[0], label='Densidade (kg/m³)')

    im5 = axes_r[1].imshow(rhoint, aspect='auto', cmap='viridis', vmin=min_rho, vmax=max_rho, extent=eixo)
    axes_r[1].set_title(f'Inversão com Interpolação - Densidade (SNR {SNR_dB}dB)', fontweight='bold')
    axes_r[1].set_ylabel('Profundidade (km)')
    fig_r.colorbar(im5, ax=axes_r[1], label='Densidade (kg/m³)')

    im6 = axes_r[2].imshow(rel_erro_rho, aspect='auto', cmap='turbo', vmin=0, vmax=100, extent=eixo)
    axes_r[2].set_title('Erro (%)', fontweight='bold')
    axes_r[2].set_xlabel('X (km)')
    axes_r[2].set_ylabel('Profundidade (km)')
    fig_r.colorbar(im6, ax=axes_r[2], label='Erro (%)')

    fig_r.tight_layout()
    fig_r.savefig(f'erro_marmousi_densidade_{N_tiros}_SNR{SNR_dB}.png', dpi=300)
    plt.close(fig_r)

    # Visualização - Impedância
    print("Gerando Imagem - Impedância")
    fig_z, axes_z = plt.subplots(3, 1, figsize=(14, 12))
    min_Z, max_Z = np.min(Zm_suav), np.max(Zm_suav)

    im7 = axes_z[0].imshow(Zm_suav, aspect='auto', cmap='magma', vmin=min_Z, vmax=max_Z, extent=eixo)
    axes_z[0].set_title('Marmousi Suavizado - Impedância', fontweight='bold')
    axes_z[0].set_ylabel('Profundidade (km)')
    fig_z.colorbar(im7, ax=axes_z[0], label='Impedância (kg/(m²s))')

    im8 = axes_z[1].imshow(Zint, aspect='auto', cmap='magma', vmin=min_Z, vmax=max_Z, extent=eixo)
    axes_z[1].set_title(f'Inversão com Interpolação - Impedância (SNR {SNR_dB}dB)', fontweight='bold')
    axes_z[1].set_ylabel('Profundidade (km)')
    fig_z.colorbar(im8, ax=axes_z[1], label='Impedância (kg/(m²s))')

    im9 = axes_z[2].imshow(rel_erro_Z, aspect='auto', cmap='turbo', vmin=0, vmax=100, extent=eixo)
    axes_z[2].set_title('Erro (%)', fontweight='bold')
    axes_z[2].set_xlabel('X (km)')
    axes_z[2].set_ylabel('Profundidade (km)')
    fig_z.colorbar(im9, ax=axes_z[2], label='Erro (%)')

    fig_z.tight_layout()
    fig_z.savefig(f'erro_marmousi_impedancia_{N_tiros}_SNR{SNR_dB}.png', dpi=300)
    plt.close(fig_z)

    print("Pronto!")
