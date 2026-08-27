import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
import scipy.io as sio
import torch

# Seleciona o dispositivo de GPU (CUDA) se disponível
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def inversao_layer_peeling_gpu(P_canal, W_canal, ni, nt2, Z_min, Z_max):
    Nx, n_sens, _ = P_canal.shape
    Zrec = torch.zeros((Nx, n_sens, ni), device=device, dtype=torch.float32)
    
    Pinv_prev = torch.zeros((Nx, n_sens, nt2), device=device, dtype=torch.float32)
    Winv_prev = torch.zeros((Nx, n_sens, nt2), device=device, dtype=torch.float32)
    
    # Superfície
    Pinv_prev[:, :, 0:nt2-2:2] = P_canal[:, :, 2:nt2:2]
    Winv_prev[:, :, 0:nt2-2:2] = W_canal[:, :, 2:nt2:2]
    
    W0 = Winv_prev[:, :, 0]
    P0 = Pinv_prev[:, :, 0]
    
    valid0 = (W0 != 0) & torch.isfinite(W0)
    Z0 = torch.where(valid0, P0 / W0, torch.tensor(Z_min, device=device))
    Z0 = torch.clamp(Z0, Z_min, Z_max)
    Zrec[:, :, 0] = Z0
    
    Zinv_prev = Z0
    
    # Layer-Peeling vetorizado no espaço (Nx, n_sens)
    for i in range(1, ni):
        Pinv_curr = torch.zeros_like(Pinv_prev)
        Winv_curr = torch.zeros_like(Winv_prev)
        
        Zi = torch.where((Zinv_prev == 0) | (~torch.isfinite(Zinv_prev)), torch.tensor(Z_min, device=device), Zinv_prev)
        Zi_exp = Zi.unsqueeze(-1)
        
        js = torch.arange(i, nt2 - i, 2, device=device)
        if len(js) > 0:
            jm1, jp1 = js - 1, js + 1
            
            a = Winv_prev[:, :, jm1] + Winv_prev[:, :, jp1]
            b = Winv_prev[:, :, jm1] - Winv_prev[:, :, jp1]
            c_calc = Pinv_prev[:, :, jm1] + Pinv_prev[:, :, jp1]
            d = Pinv_prev[:, :, jm1] - Pinv_prev[:, :, jp1]
            
            Winv_curr[:, :, js] = 0.5 * (a + d / Zi_exp)
            Pinv_curr[:, :, js] = 0.5 * (Zi_exp * b + c_calc)
        
        W_ii = Winv_curr[:, :, i]
        P_ii = Pinv_curr[:, :, i]
        
        valid_i = (W_ii != 0) & torch.isfinite(W_ii)
        Znovo = P_ii / W_ii
        
        Z_atual = torch.where(valid_i & torch.isfinite(Znovo), torch.clamp(Znovo, Z_min, Z_max), Zi)
        Zrec[:, :, i] = Z_atual
        
        Zinv_prev = Z_atual
        Pinv_prev = Pinv_curr
        Winv_prev = Winv_curr
        
    return Zrec

def otimizar_parametros_gpu(Zrec, xi2, w0, lim, n_iters=500):
    Z_medido = Zrec.permute(0, 2, 1) # (Nx, ni, n_sens)
    Nx, ni, n_sens = Z_medido.shape
    
    rho_min, rho_max = lim[0]
    c_min, c_max = lim[1]
    Z0_min, Z0_max = rho_min * c_min, rho_max * c_max
    
    validos = torch.abs(Z_medido) > 1e-5
    escala_Z = torch.median(torch.abs(Z_medido), dim=-1, keepdim=True).values
    escala_Z = torch.where(~torch.isfinite(escala_Z) | (escala_Z < 1e-12), torch.tensor(1.0, device=device), escala_Z)
    
    # 1. Extrai Z0 diretamente da incidência normal (xi = 0, canal central)
    Z0_inc_normal = Zrec[:, n_sens // 2, :].clamp(Z0_min + 1e-3, Z0_max - 1e-3)
    
    # 2. Inicialização exata do u_Z via inversão do Sigmoid
    p_init = (Z0_inc_normal - Z0_min) / (Z0_max - Z0_min)
    u_Z_init = torch.log(p_init / (1.0 - p_init))
    
    u_Z = u_Z_init.clone().detach().requires_grad_(True)
    u_c = torch.zeros((Nx, ni), device=device, requires_grad=True)
    
    optimizer = torch.optim.Adam([u_Z, u_c], lr=0.03)
    xi2_gpu = xi2.unsqueeze(0).unsqueeze(0)
    
    for _ in range(n_iters):
        optimizer.zero_grad()
        
        Z0 = Z0_min + (Z0_max - Z0_min) * torch.sigmoid(u_Z)
        c = c_min + (c_max - c_min) * torch.sigmoid(u_c)
        rho_virtual = Z0 / c
        
        c_3d = c.unsqueeze(-1)
        rho_3d = rho_virtual.unsqueeze(-1)
        
        kz2 = torch.clamp((w0 / c_3d)**2 - xi2_gpu, min=1e-10)
        kz = torch.sqrt(kz2)
        Z_teo = (rho_3d * w0) / kz
        
        res = (Z_medido - Z_teo) / escala_Z
        loss = torch.where(validos, res**2, torch.zeros_like(res)).sum()
        
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        Z0_final = Z0_min + (Z0_max - Z0_min) * torch.sigmoid(u_Z)
        c_final = torch.clamp(c_min + (c_max - c_min) * torch.sigmoid(u_c), c_min, c_max)
        rho_final = torch.clamp(Z0_final / c_final, rho_min, rho_max)
        
    return c_final.T.cpu().numpy(), rho_final.T.cpu().numpy(), Z0_final.T.cpu().numpy()


if __name__ == '__main__':
    print("Carregando dados PW")
    dados = np.load('dados_marmousi_P_W_completos.npz')
    
    P_real = torch.from_numpy(dados['P_real']).to(device).float()
    W_real = torch.from_numpy(dados['W_real']).to(device).float()
    
    Nx_idx, n_sens, nt = P_real.shape
    ni = 2800                  
    nt2 = nt 
    dx = 50.0
    w0 = 2 * np.pi * 50
    
    xi_np = (2 * np.pi / (n_sens * dx)) * np.arange(-(n_sens//2), (n_sens//2) + 1, dtype=np.float32)
    xi2_gpu = torch.from_numpy(xi_np**2).to(device)

    # Injetando ruído
    SNR_dB = 20
    N_tiros = 25
    fator_ruido = 10.0 ** (-SNR_dB / 20.0)

    Vel_acumulada = np.zeros((ni, Nx_idx), dtype=np.float32)
    Rho_acumulada = np.zeros((ni, Nx_idx), dtype=np.float32)
    Z0_acumulada = np.zeros((ni, Nx_idx), dtype=np.float32)

    lim = [(900, 6000), (1400, 4000)]
    Z_min, Z_max = lim[0][0] * lim[1][0], lim[0][1] * lim[1][1]

    print(f"Executando no dispositivo: {device}. Iniciando loop de {N_tiros} tiros com SNR = {SNR_dB} dB")

    for tiro in range(1, N_tiros + 1):
        print(f"Loop: {tiro}/{N_tiros}")
        
        rms_P = torch.sqrt(torch.mean(P_real**2, dim=(1,2), keepdim=True))
        rms_W = torch.sqrt(torch.mean(W_real**2, dim=(1,2), keepdim=True))
        
        ruido_P = torch.randn_like(P_real)
        ruido_W = torch.randn_like(W_real)
        
        P_noisy = P_real + (fator_ruido * rms_P) * ruido_P
        W_noisy = W_real + (fator_ruido * rms_W) * ruido_W

        P_canal = torch.real(torch.fft.fftshift(torch.fft.fft(P_noisy, dim=1), dim=1))
        W_canal = torch.real(torch.fft.fftshift(torch.fft.fft(W_noisy, dim=1), dim=1))

        Zrec = inversao_layer_peeling_gpu(P_canal, W_canal, ni, nt2, Z_min, Z_max)
        
        vel_tiro, rho_tiro, Z0_tiro = otimizar_parametros_gpu(Zrec, xi2_gpu, w0, lim, n_iters=500)
        
        Vel_acumulada += vel_tiro
        Rho_acumulada += rho_tiro
        Z0_acumulada += Z0_tiro

    print("Tirando a média")
    Vel_rec = Vel_acumulada / N_tiros
    Rho_rec = Rho_acumulada / N_tiros
    Z0_rec_media = Z0_acumulada / N_tiros

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
    xinv = np.arange(Nx_idx) * dx 
    xm = np.arange(nxm) * dxm

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

    print("Calculando matrizes de erro")
    Zm_suav = rhom_suav * vm_suav

    abs_erro_v = np.abs(vint - vm_suav)
    rel_erro_v = (abs_erro_v / np.maximum(vm_suav, 1e-10)) * 100

    abs_erro_rho = np.abs(rhoint - rhom_suav)
    rel_erro_rho = (abs_erro_rho / np.maximum(rhom_suav, 1e-10)) * 100

    abs_erro_Z = np.abs(Zint - Zm_suav)
    rel_erro_Z = (abs_erro_Z / np.maximum(Zm_suav, 1e-10)) * 100

    mean_erro_v, median_erro_v, max_erro_v = np.mean(rel_erro_v), np.median(rel_erro_v), np.max(rel_erro_v)
    mean_erro_rho, median_erro_rho, max_erro_rho = np.mean(rel_erro_rho), np.median(rel_erro_rho), np.max(rel_erro_rho)
    mean_erro_Z, median_erro_Z, max_erro_Z = np.mean(rel_erro_Z), np.median(rel_erro_Z), np.max(rel_erro_Z)

    print("-" * 65)
    print(f"VELOCIDADE -> Médio: {mean_erro_v:.2f}% | Mediana: {median_erro_v:.2f}% | Máximo: {max_erro_v:.2f}%")
    print(f"DENSIDADE  -> Médio: {mean_erro_rho:.2f}% | Mediana: {median_erro_rho:.2f}% | Máximo: {max_erro_rho:.2f}%")
    print(f"IMPEDÂNCIA -> Médio: {mean_erro_Z:.2f}% | Mediana: {median_erro_Z:.2f}% | Máximo: {max_erro_Z:.2f}%")
    print("-" * 65)
    
    nome_arquivo_txt = f'estatisticas_m2_erro_{N_tiros}_SNR{SNR_dB}.txt'
    with open(nome_arquivo_txt, 'w', encoding='utf-8') as f:
        f.write(f"Resultados da Inversão Sísmica GPU - {N_tiros} tiro(s) | SNR: {SNR_dB} dB\n")
        f.write("-" * 65 + "\n")
        f.write(f"VELOCIDADE -> Erro Médio: {mean_erro_v:.2f}% | Mediana: {median_erro_v:.2f}% | Máximo: {max_erro_v:.2f}%\n")
        f.write(f"DENSIDADE  -> Erro Médio: {mean_erro_rho:.2f}% | Mediana: {median_erro_rho:.2f}% | Máximo: {max_erro_rho:.2f}%\n")
        f.write(f"IMPEDÂNCIA -> Erro Médio: {mean_erro_Z:.2f}%   | Mediana: {median_erro_Z:.2f}% | Máximo: {max_erro_Z:.2f}%\n")
    print(f"Estatísticas de erro salvas no arquivo: {nome_arquivo_txt}")

    xkm1 = xm[0] / 1000
    xkm2 = xm[-1] / 1000
    zkm = (ni * dxm) / 1000
    eixo = [xkm1, xkm2, zkm, 0] 

    # Plot - Velocidade
    fig_v, axes_v = plt.subplots(3, 1, figsize=(14, 12))
    min_vel, max_vel = np.min(vm_suav), np.max(vm_suav)

    im1 = axes_v[0].imshow(vm_suav, aspect='auto', cmap='jet', vmin=min_vel, vmax=max_vel, extent=eixo)
    axes_v[0].set_title('Marmousi Suavizado - Velocidade', fontweight='bold')
    axes_v[0].set_ylabel('Profundidade (km)')
    fig_v.colorbar(im1, ax=axes_v[0], label='Velocidade (m/s)')

    im2 = axes_v[1].imshow(vint, aspect='auto', cmap='jet', vmin=min_vel, vmax=max_vel, extent=eixo)
    axes_v[1].set_title(f'Inversão com Interpolação GPU - Velocidade (SNR {SNR_dB}dB)', fontweight='bold')
    axes_v[1].set_ylabel('Profundidade (km)')
    fig_v.colorbar(im2, ax=axes_v[1], label='Velocidade (m/s)')

    im3 = axes_v[2].imshow(rel_erro_v, aspect='auto', cmap='turbo', vmin=0, vmax=100, extent=eixo)
    axes_v[2].set_title('Erro (%)', fontweight='bold')
    axes_v[2].set_xlabel('X (km)')
    axes_v[2].set_ylabel('Profundidade (km)')
    fig_v.colorbar(im3, ax=axes_v[2], label='Erro (%)')

    fig_v.tight_layout()
    fig_v.savefig(f'erro_marmousi_velocidade_m2_{N_tiros}_SNR{SNR_dB}.png', dpi=300)
    plt.close(fig_v)

    # Plot - Densidade
    fig_r, axes_r = plt.subplots(3, 1, figsize=(14, 12))
    min_rho, max_rho = np.min(rhom_suav), np.max(rhom_suav)

    im4 = axes_r[0].imshow(rhom_suav, aspect='auto', cmap='viridis', vmin=min_rho, vmax=max_rho, extent=eixo)
    axes_r[0].set_title('Marmousi Suavizado - Densidade', fontweight='bold')
    axes_r[0].set_ylabel('Profundidade (km)')
    fig_r.colorbar(im4, ax=axes_r[0], label='Densidade (kg/m³)')

    im5 = axes_r[1].imshow(rhoint, aspect='auto', cmap='viridis', vmin=min_rho, vmax=max_rho, extent=eixo)
    axes_r[1].set_title(f'Inversão com Interpolação GPU - Densidade (SNR {SNR_dB}dB)', fontweight='bold')
    axes_r[1].set_ylabel('Profundidade (km)')
    fig_r.colorbar(im5, ax=axes_r[1], label='Densidade (kg/m³)')

    im6 = axes_r[2].imshow(rel_erro_rho, aspect='auto', cmap='turbo', vmin=0, vmax=100, extent=eixo)
    axes_r[2].set_title('Erro (%)', fontweight='bold')
    axes_r[2].set_xlabel('X (km)')
    axes_r[2].set_ylabel('Profundidade (km)')
    fig_r.colorbar(im6, ax=axes_r[2], label='Erro (%)')

    fig_r.tight_layout()
    fig_r.savefig(f'erro_marmousi_densidade_m2_{N_tiros}_SNR{SNR_dB}.png', dpi=300)
    plt.close(fig_r)

    # Plot - Impedância
    fig_z, axes_z = plt.subplots(3, 1, figsize=(14, 12))
    min_Z, max_Z = np.min(Zm_suav), np.max(Zm_suav)

    im7 = axes_z[0].imshow(Zm_suav, aspect='auto', cmap='magma', vmin=min_Z, vmax=max_Z, extent=eixo)
    axes_z[0].set_title('Marmousi Suavizado - Impedância', fontweight='bold')
    axes_z[0].set_ylabel('Profundidade (km)')
    fig_z.colorbar(im7, ax=axes_z[0], label='Impedância (kg/(m²s))')

    im8 = axes_z[1].imshow(Zint, aspect='auto', cmap='magma', vmin=min_Z, vmax=max_Z, extent=eixo)
    axes_z[1].set_title(f'Inversão Direta com Interpolação GPU - Impedância (SNR {SNR_dB}dB)', fontweight='bold')
    axes_z[1].set_ylabel('Profundidade (km)')
    fig_z.colorbar(im8, ax=axes_z[1], label='Impedância (kg/(m²s))')

    im9 = axes_z[2].imshow(rel_erro_Z, aspect='auto', cmap='turbo', vmin=0, vmax=100, extent=eixo)
    axes_z[2].set_title('Erro (%)', fontweight='bold')
    axes_z[2].set_xlabel('X (km)')
    axes_z[2].set_ylabel('Profundidade (km)')
    fig_z.colorbar(im9, ax=axes_z[2], label='Erro (%)')

    fig_z.tight_layout()
    fig_z.savefig(f'erro_marmousi_impedancia_m2_{N_tiros}_SNR{SNR_dB}.png', dpi=300)
    plt.close(fig_z)

    print("Pronto!")