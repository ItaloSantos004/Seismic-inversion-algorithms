clear; clc; close all;

M = 4; % ordem de Higdon

Nx = 150; % dominio
Nz = 200;

h = 10.0;
c = 1500.0;
dt = 0.004;
S = c * dt / h;
tempo = 430;

m = max(1, M);
U = zeros(Nx, Nz, tempo + m);

% Parametros da fonte
t0 = 0.1;
s_gauss = 0.02; % Renomeado para s_gauss para não confundir com a função s(t)
x0 = round(Nx/2);
z0 = 2;

% =========================================================================
% NOVIDADE: SETUP DO RECEPTOR (GEOFONE) PARA COMPARAÇÃO
% =========================================================================
xr = 100; % Posição X do receptor na malha
zr = 100; % Posição Z do receptor na malha

% Distância física real (em metros) entre a fonte e o receptor
r = sqrt(((xr - x0) * h)^2 + ((zr - z0) * h)^2);

% Vetores para guardar o histórico da onda (Sismogramas)
traco_fdtd = zeros(1, tempo);
traco_ana  = zeros(1, tempo);

% Matriz de Higdon
c1 = (S - 1) / (S + 1);
c2 = [1, -c1; c1, -1];
c3 = 1;

if M > 0
    for i = 1:M
        c3 = conv2(c3, c2);
    end
end

figure('Position', [100, 100, 1000, 500]);

% =========================================================================
% LAÇO PRINCIPAL
% =========================================================================
for n = 1:tempo
    t = n * dt;
    p = n + m;

    % 1. INTERIOR DA MALHA FDTD
    U(2:Nx-1, 2:Nz-1, p+1) = 2*U(2:Nx-1, 2:Nz-1, p) - U(2:Nx-1, 2:Nz-1, p-1) + ...
        (S^2) * (U(3:Nx, 2:Nz-1, p) - 2*U(2:Nx-1, 2:Nz-1, p) + U(1:Nx-2, 2:Nz-1, p) + ...
                 U(2:Nx-1, 3:Nz, p) - 2*U(2:Nx-1, 2:Nz-1, p) + U(2:Nx-1, 1:Nz-2, p));

    % 2. INJEÇÃO DA FONTE FDTD
    f = exp(-((t - t0) / s_gauss)^2);
    U(x0, z0, p+1) = U(x0, z0, p+1) + f;

    % 3. CONDIÇÕES DE CONTORNO
    U(:, 1, p+1) = 0; % Superficie Rigida

    if M == 0
        U(1, :, p+1) = 0; U(Nx, :, p+1) = 0; U(:, Nz, p+1) = 0;
    else
        U(1, :, p+1) = 0; U(Nx, :, p+1) = 0; U(:, Nz, p+1) = 0;
        for i = 1:M+1
            for j = 1:M+1
                if i==1 && j==1; continue; end
                coef = c3(i,j);
                tn = p - i + 2;

                U(1, :, p+1)  = U(1, :, p+1)  - coef * U(j, :, tn);
                U(Nx, :, p+1) = U(Nx, :, p+1) - coef * U(Nx - j + 1, :, tn);
                U(:, Nz, p+1) = U(:, Nz, p+1) - coef * U(:, Nz - j + 1, tn);
            end
        end
    end

    % =========================================================================
    % 4. GRAVAÇÃO DOS TRAÇOS (O TESTE DE VALIDAÇÃO)
    % =========================================================================
    % Salva o valor que o FDTD calculou para o receptor neste instante
    traco_fdtd(n) = U(xr, zr, p+1);

    % Calcula a Solução Analítica (Integral de Convolução 2D)
    if t > r/c
        % Cria o vetor de tempo tau até um pentelhésimo antes do limite
        % para evitar a divisão por zero na singularidade da raiz
        tau = linspace(0, t - r/c - 1e-6, 400);

        % O sinal da fonte avaliado no passado (tau)
        s_tau = exp(-((tau - t0) / s_gauss).^2);

        % O integrando da Equação 2.21 do artigo
        integrando = s_tau ./ sqrt(c^2 * (t - tau).^2 - r^2);

        % Integração numérica via trapézios multiplicada pela constante
        traco_ana(n) = (1 / (2*pi*c)) * trapz(tau, integrando);
    else
        traco_ana(n) = 0; % A onda ainda não chegou
    end

    % =========================================================================
    % 5. ANIMAÇÃO DO CAMPO (Opcional, comentada para rodar mais rápido)
    % =========================================================================
    % Se quiser ver a onda andando, descomente as linhas abaixo.
    % if mod(n, 10) == 0
    %     subplot(1,2,1);
    %     imagesc(U(:,:,p)'); caxis([-0.02 0.05]); colormap(jet);
    %     title(sprintf('FDTD 2D - Tempo: %d', n)); drawnow;
    % end
end

% =========================================================================
% PLOTAGEM FINAL: FDTD vs ANALÍTICO
% =========================================================================
% Nota sobre Fator de Escala: A injeção direta 'U = U + f' no FDTD
% não gera a mesma amplitude absoluta de uma fonte contínua ideal (delta de Dirac).
% Para sobrepor perfeitamente e avaliar a *fase*, normalizamos os dois sinais.

traco_fdtd_norm = traco_fdtd / max(abs(traco_fdtd));
traco_ana_norm  = traco_ana / max(abs(traco_ana));

tempo_vetor = (1:tempo) * dt;

plot(tempo_vetor, traco_fdtd_norm, 'b-', 'LineWidth', 2);
hold on;
plot(tempo_vetor, traco_ana_norm, 'r--', 'LineWidth', 2);
grid on;
set(gca, 'GridAlpha', 0.4);
legend('Numérico (FDTD)', 'Analítico (Função de Green)', 'Location', 'Best');
title('Validação do Sismograma: FDTD vs Equação 2.21');
xlabel('Tempo (s)');
ylabel('Amplitude Normalizada');
