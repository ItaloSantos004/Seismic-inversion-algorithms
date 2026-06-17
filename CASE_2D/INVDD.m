clear; clc; close all;

% =========================================================================
% 1. PARÂMETROS DA GRADE 2D E DO MEIO
% =========================================================================
nt = 300;     % tempo (reduzido para rodar mais rápido no teste)
ni = 100;     % profundidade (número de camadas)

nx = 21;      % Número de sensores na superfície (Canais)
dx = 40;      % Espaçamento entre sensores (m)
omega0 = 2*pi*50; % Frequência angular central assumida (50 Hz)

% Em vez de apenas Z, agora o meio real tem Densidade (rho) e Velocidade (c)
rho_real = ones(1, ni) * 1000;
c_real   = ones(1, ni) * 1500;

% Simulando suas interfaces, mas agora com rho e c separados
rho_real(31:60) = 2000;  c_real(31:60) = 3000;
rho_real(61:end) = 750;  c_real(61:end) = 1000;

% Impedância de incidência normal verdadeira (apenas para o plot final)
Z_normal_real = rho_real .* c_real;

% =========================================================================
% 2. O FRONT-END DA TRANSFORMADA DE FOURIER (Canais de Número de Onda)
% =========================================================================
% Vetor de número de onda horizontal (xi)
xi = (2*pi / (nx * dx)) * (-floor(nx/2):floor(nx/2));

% =========================================================================
% 3. PROBLEMA DIRETO (GERANDO DADOS DE SUPERFÍCIE PARA CADA CANAL)
% =========================================================================
% matrizes para guardar os dados da superfície (xi, tempo)
P2_canais = zeros(nx, nt);
W2_canais = zeros(nx, nt);

fonte2 = zeros(1, nt);
fonte2(3) = 1; % O mesmo spike do seu código original

disp('Executando Forward 1D para cada canal (Gerando dados sintéticos)...');
for m = 1:nx
    % Impedância OBLÍQUA real para o canal m: Z = (rho*c) / sqrt(1 - (c*xi/w)^2)
    Z_m = zeros(1, ni);
    for i = 1:ni
        kz = sqrt((omega0/c_real(i))^2 - xi(m)^2); % Num de onda vertical
        if isreal(kz) && kz > 0
            Z_m(i) = rho_real(i) * omega0 / kz;
        else
            Z_m(i) = Z_m(i-1); % Onda evanescente (evita crash no sintético)
        end
    end

    % --- INÍCIO DO SEU CÓDIGO FORWARD 1D (adaptado para usar Z_m) ---
    P2 = zeros(ni, nt); W2 = zeros(ni, nt);
    for j = 2:nt
        if mod(1 + j, 2) == 0 % superficie
            P2(1, j) = fonte2(j);
            U2 = Z_m(1)*W2(2, j-1) - P2(2, j-1);
            W2(1, j) = (P2(1, j) + U2) / Z_m(1);
        end
        for i = 2:ni-1 % pontos dentro
            if mod(i + j, 2) == 0
                D2 = Z_m(i-1)*W2(i-1, j-1) + P2(i-1, j-1);
                U2 = Z_m(i)*W2(i+1, j-1) - P2(i+1, j-1);
                W2(i, j) = (D2 + U2) / (Z_m(i-1) + Z_m(i));
                P2(i, j) = (Z_m(i)*D2 - Z_m(i-1)*U2) / (Z_m(i-1) + Z_m(i));
            end
        end
        if mod(ni + j, 2) == 0 % fundo
            D2 = Z_m(ni-1)*W2(ni-1, j-1) + P2(ni-1, j-1);
            W2(ni, j) = D2 / (2 * Z_m(ni-1));
            P2(ni, j) = D2 / 2;
        end
    end
    % --- FIM DO SEU CÓDIGO FORWARD 1D ---

    % Armazena APENAS a superfície lida por este canal
    P2_canais(m, :) = P2(1, :);
    W2_canais(m, :) = W2(1, :);
end
% Na vida real, P2_canais e W2_canais vêm da aplicação de fft() nos dados do sensor.

% =========================================================================
% 4. O PROBLEMA INVERSO 2D (DESCASCAMENTO CAMADA POR CAMADA)
% =========================================================================
nt2 = 2 * ni;
Z_reconstruido_2D = zeros(nx, ni); % Vai guardar a impedancia de cada camada p/ cada canal

disp('Executando Inversão DC-wp (Descascamento) por canal...');
for m = 1:nx
    % --- INÍCIO DO SEU CÓDIGO DE INVERSÃO 1D ---
    P4 = zeros(ni, nt2);
    W4 = zeros(ni, nt2);
    Z4 = zeros(1, ni);

    for j = 1:2:nt2
        P4(1, j) = P2_canais(m, j + 2);
        W4(1, j) = W2_canais(m, j + 2);
    end

    Z4(1) = P4(1, 1) / W4(1, 1);

    for i = 2:ni
        for j = i:2:(nt2 - i + 1)
            a = W4(i-1, j-1) + W4(i-1, j+1);
            b = W4(i-1, j-1) - W4(i-1, j+1);
            c = P4(i-1, j-1) + P4(i-1, j+1);
            d = P4(i-1, j-1) - P4(i-1, j+1);

            W4(i, j) = 0.5 * (a + d / Z4(i-1));
            P4(i, j) = 0.5 * (Z4(i-1) * b + c);
        end
        Z4(i) = P4(i, i) / W4(i, i);
    end
    % --- FIM DO SEU CÓDIGO DE INVERSÃO 1D ---

    Z_reconstruido_2D(m, :) = Z4; % Guarda a assinatura oblíqua desta camada
end

% =========================================================================
% 5. CONECTANDO OS CANAIS (SEPARANDO RHO E C)
% =========================================================================
% A impedância oblíqua segue a regra: Z = (rho * c) / sqrt(1 - c^2*xi^2 / w^2)
% Elevando ao quadrado e invertendo, temos uma equação de RETA:
% 1/Z^2 = [ 1 / (rho^2 * c^2) ] + [ -1 / (rho^2 * w^2) ] * xi^2
% y     =          A            +          B             * x

disp('Calculando Regressão Linear para extrair Densidade e Velocidade...');
rho_est = zeros(1, ni);
c_est   = zeros(1, ni);

x_reg = (xi(:).^2); % Eixo x da regressão

for i = 1:ni
    y_reg = 1 ./ (Z_reconstruido_2D(:, i).^2);

    % Regressão linear simples (Polinômio de grau 1)
    % p(1) é o B (inclinação), p(2) é o A (interseção)
    p = polyfit(x_reg, y_reg, 1);
    B = min(p(1), -1e-20); % Garante que seja negativo (física)
    A = max(p(2),  1e-20); % Garante que seja positivo

    % Recuperando a física a partir dos coeficientes da reta
    rho_est(i) = sqrt( -1 / (B * omega0^2) );
    c_est(i)   = sqrt(  1 / (A * rho_est(i)^2) );
end

% A Impedância normal final estimada é o produto das estimativas
Z_normal_est = rho_est .* c_est;

% =========================================================================
% 6. VISUALIZAÇÃO
% =========================================================================
figure;

% PLOT 1: DENSIDADE
subplot(3, 1, 1);
stairs(1:ni, rho_real, 'b', 'LineWidth', 2, 'DisplayName', 'Real'); hold on;
stairs(1:ni, rho_est, 'r--', 'LineWidth', 2, 'DisplayName', 'Estimado (2D)');
ylabel('Densidade \rho (kg/m^3)');
title('Recuperação Separada: Densidade');
legend; grid on; axis tight;

% PLOT 2: VELOCIDADE DO SOM
subplot(3, 1, 2);
stairs(1:ni, c_real, 'b', 'LineWidth', 2); hold on;
stairs(1:ni, c_est, 'r--', 'LineWidth', 2);
ylabel('Velocidade c (m/s)');
title('Recuperação Separada: Velocidade do Som');
grid on; axis tight;

% PLOT 3: IMPEDÂNCIA NORMAL
subplot(3, 1, 3);
stairs(1:ni, Z_normal_real, 'b', 'LineWidth', 2); hold on;
stairs(1:ni, Z_normal_est, 'r--', 'LineWidth', 2);
xlabel('Camada i (Tempo de Trânsito \zeta)');
ylabel('Impedância Normal Z_0');
title('Impedância Acústica (O que a Inversão 1D daria)');
grid on; axis tight;
