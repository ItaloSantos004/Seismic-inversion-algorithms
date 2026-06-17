% =========================================================================
% SIMULAÇÃO 2D: EXPLOSÃO NA SUPERFÍCIE COM MEDIDOR DE REFLEXÃO
% =========================================================================
clear; clc; close all;

% 1. Parâmetros de Controle do Usuário
% 0 = Bordas Rígidas | 1 = Higdon 1ª Ordem | 2 = Higdon 2ª Ordem
ordem_absorcao = 2;

% 2. Parâmetros do Domínio e Malha
Nx = 150;
Nz = 150;
h = 10.0;
c = 1500.0;

% 3. Parâmetros de Tempo
dt = 0.004;
S = c * dt / h;
PassosTempo = 500; % Aumentei um pouco para dar tempo da onda sair totalmente

% Inicialização
U_past = zeros(Nx, Nz);
U_curr = zeros(Nx, Nz);
U_next = zeros(Nx, Nz);

% 4. Configuração da Explosão (Fonte Gaussiana)
t0 = 0.1;
sigma = 0.02;
x_src = round(Nx/2);
z_src = 2;

% Variáveis para o Medidor de Energia
max_energia = 1e-10; % Evitar divisão por zero no início

c1 = (S - 1) / (S + 1);

figure('Position', [100, 100, 800, 600]);

% =========================================================================
% LAÇO PRINCIPAL
% =========================================================================
for n = 1:PassosTempo
    t = n * dt;

    % A. ATUALIZAÇÃO DO INTERIOR
    U_next(2:Nx-1, 2:Nz-1) = 2*U_curr(2:Nx-1, 2:Nz-1) - U_past(2:Nx-1, 2:Nz-1) + ...
        (S^2) * (U_curr(3:Nx, 2:Nz-1) - 2*U_curr(2:Nx-1, 2:Nz-1) + U_curr(1:Nx-2, 2:Nz-1) + ...
                 U_curr(2:Nx-1, 3:Nz) - 2*U_curr(2:Nx-1, 2:Nz-1) + U_curr(2:Nx-1, 1:Nz-2));

    % B. INJEÇÃO DA EXPLOSÃO
    src = exp(-((t - t0) / sigma)^2);
    U_next(x_src, z_src) = U_next(x_src, z_src) + src;

    % Condição de superfície livre rígida (o "splash")
    U_next(:, 1) = U_curr(:, 2);

    % C. CONDIÇÕES DE TRANSPARÊNCIA (Higdon)
    if ordem_absorcao == 0
        U_next(1, :) = 0; U_next(Nx, :) = 0; U_next(:, Nz) = 0;

    elseif ordem_absorcao == 1
        U_next(1, :) = U_curr(2, :) + c1 * (U_next(2, :) - U_curr(1, :));
        U_next(Nx, :) = U_curr(Nx-1, :) + c1 * (U_next(Nx-1, :) - U_curr(Nx, :));
        U_next(:, Nz) = U_curr(:, Nz-1) + c1 * (U_next(:, Nz-1) - U_curr(:, Nz));

    elseif ordem_absorcao == 2
        U_next(1, :) = 2*c1*U_curr(1, :) - (c1^2)*U_past(1, :) ...
            - 2*c1*U_next(2, :) + 2*(1+c1^2)*U_curr(2, :) - 2*c1*U_past(2, :) ...
            - (c1^2)*U_next(3, :) + 2*c1*U_curr(3, :) - U_past(3, :);

        U_next(Nx, :) = 2*c1*U_curr(Nx, :) - (c1^2)*U_past(Nx, :) ...
            - 2*c1*U_next(Nx-1, :) + 2*(1+c1^2)*U_curr(Nx-1, :) - 2*c1*U_past(Nx-1, :) ...
            - (c1^2)*U_next(Nx-2, :) + 2*c1*U_curr(Nx-2, :) - U_past(Nx-2, :);

        U_next(:, Nz) = 2*c1*U_curr(:, Nz) - (c1^2)*U_past(:, Nz) ...
            - 2*c1*U_next(:, Nz-1) + 2*(1+c1^2)*U_curr(:, Nz-1) - 2*c1*U_past(:, Nz-1) ...
            - (c1^2)*U_next(:, Nz-2) + 2*c1*U_curr(:, Nz-2) - U_past(:, Nz-2);
    end

    % D. MEDIDOR DE REFLEXÃO / ENERGIA DO DOMÍNIO
    % Calcula a soma dos quadrados das amplitudes
    energia_atual = sum(U_next(:).^2);

    % Atualiza o pico de energia histórico
    if energia_atual > max_energia
        max_energia = energia_atual;
    end

    % Calcula quantos % da onda ainda está dentro da malha
    perc_restante = (energia_atual / max_energia) * 100;

    % E. ATUALIZAÇÃO DA MEMÓRIA
    U_past = U_curr;
    U_curr = U_next;

    % F. PLOTAGEM ANIMADA
    if mod(n, 4) == 0
        imagesc(U_curr');
        caxis([-0.02 0.05]);
        colormap(jet);
        colorbar;
        % O título agora atua como um HUD de instrumentação
        title(sprintf('Passo: %d | Ordem: %d | Energia no Domínio: %.2f%%', n, ordem_absorcao, perc_restante));
        xlabel('Distância X');
        ylabel('Profundidade Z');
        drawnow;
    end
end
