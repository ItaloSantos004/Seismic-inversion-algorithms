clear; clc; close all;

% =========================================================================
% 1. DEFINIÇÃO DO DOMÍNIO E MODELO DE IMPEDÂNCIA
% =========================================================================
ni = 400;
nt = 3 * ni;

Z = ones(1, ni);
Z(1:120)   = 2000 * 1500;
Z(121:250) = 2200 * 3000;
Z(251:end) = 750 * 1200;

fonte2 = zeros(1, nt);
fonte2(3) = 1;

% ESCOLHA DE UMA ÚNICA AMPLITUDE
A = 5 * 10^-8;
z0 = 0;
s = 5;

fprintf('Calculando os campos de onda...\n');

% =========================================================================
% 2. CASO 1: PROBLEMA DIRETO SEM PERTURBAÇÃO (A = 0)
% =========================================================================
P_sem = zeros(ni, nt);
W_sem = zeros(ni, nt);

for j = 2:nt
    if mod(1 + j, 2) == 0
        P_sem(1, j) = fonte2(j);
        U_sem = Z(1)*W_sem(2, j-1) - P_sem(2, j-1);
        W_sem(1, j) = (P_sem(1, j) + U_sem) / Z(1);
    end
    for i = 2:ni-1
        if mod(i + j, 2) == 0
            D_sem = Z(i-1)*W_sem(i-1, j-1) + P_sem(i-1, j-1);
            U_sem = Z(i)*W_sem(i+1, j-1) - P_sem(i+1, j-1);
            W_sem(i, j) = (D_sem + U_sem) / (Z(i-1) + Z(i));
            P_sem(i, j) = (Z(i)*D_sem - Z(i-1)*U_sem) / (Z(i-1) + Z(i));
        end
    end
    if mod(ni + j, 2) == 0
        D_sem = Z(ni-1)*W_sem(ni-1, j-1) + P_sem(ni-1, j-1);
        W_sem(ni, j) = D_sem / (2 * Z(ni-1));
        P_sem(ni, j) = D_sem / 2;
    end
end

% =========================================================================
% 3. CASO 2: PROBLEMA DIRETO COM PERTURBAÇÃO
% =========================================================================
P_com = zeros(ni, nt);
W_com = zeros(ni, nt);

% Condição inicial Gaussiana (Perturbação)
for i = 1:ni
    W_com(i, 1) = A * exp(-((i - z0).^2) / (2 * s^2));
end

for j = 2:nt
    if mod(1 + j, 2) == 0
        P_com(1, j) = fonte2(j);
        U_com = Z(1)*W_com(2, j-1) - P_com(2, j-1);
        W_com(1, j) = (P_com(1, j) + U_com) / Z(1);
    end
    for i = 2:ni-1
        if mod(i + j, 2) == 0
            D_com = Z(i-1)*W_com(i-1, j-1) + P_com(i-1, j-1);
            U_com = Z(i)*W_com(i+1, j-1) - P_com(i+1, j-1);
            W_com(i, j) = (D_com + U_com) / (Z(i-1) + Z(i));
            P_com(i, j) = (Z(i)*D_com - Z(i-1)*U_com) / (Z(i-1) + Z(i));
        end
    end
    if mod(ni + j, 2) == 0
        D_com = Z(ni-1)*W_com(ni-1, j-1) + P_com(ni-1, j-1);
        W_com(ni, j) = D_com / (2 * Z(ni-1));
        P_com(ni, j) = D_com / 2;
    end
end

% =========================================================================
% 4. PLOTAGEM DE W NA SUPERFÍCIE (i = 1)
% =========================================================================

% Filtrando apenas os índices ímpares do tempo onde a superfície é calculada
tempo = 1:2:nt;
W_sem_superficie = W_sem(1, tempo);
W_com_superficie = W_com(1, tempo);

figure;

plot(tempo, W_sem_superficie, 'b', 'LineWidth', 2);
hold on;
plot(tempo, W_com_superficie, 'r', 'LineWidth', 2);
hold off;

title(['Superfície | Perturbação A = ', num2str(A)]);
xlabel('T');
ylabel('A');
grid on;

fprintf('Gráfico gerado com sucesso!\n');
