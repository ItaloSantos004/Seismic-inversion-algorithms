clear; clc; close all;

%dominio
ni = 400;
nt = 3 * ni;

Z = ones(1, ni);
Z(1:120)   = 2000 * 1500;
Z(121:250) = 2200 * 3000;
Z(251:end) = 750 * 1200;

fonte2 = zeros(1, nt);
fonte2(3) = 1;

sigmas = [1, 2, 5, 10, 20];
cores = ['k', 'r', 'm', 'c', 'g'];

figure;
stairs(1:ni, Z, 'b', 'LineWidth', 3);
hold on;

fprintf('Inicio');

A = 5e-6;
z0 = 0;

for k = 1:length(sigmas)

    s = sigmas(k);

    P2 = zeros(ni, nt);
    W2 = zeros(ni, nt);

    for i = 1:ni
        W2(i, 1) = A * exp(-((i - z0).^2) / (2 * s^2));
    end

    % PROBLEMA DIRETO
    for j = 2:nt
        if mod(1 + j, 2) == 0
            P2(1, j) = fonte2(j);
            U2 = Z(1)*W2(2, j-1) - P2(2, j-1);
            W2(1, j) = (P2(1, j) + U2) / Z(1);
        end
        for i = 2:ni-1
            if mod(i + j, 2) == 0
                D2 = Z(i-1)*W2(i-1, j-1) + P2(i-1, j-1);
                U2 = Z(i)*W2(i+1, j-1) - P2(i+1, j-1);
                W2(i, j) = (D2 + U2) / (Z(i-1) + Z(i));
                P2(i, j) = (Z(i)*D2 - Z(i-1)*U2) / (Z(i-1) + Z(i));
            end
        end
        if mod(ni + j, 2) == 0
            D2 = Z(ni-1)*W2(ni-1, j-1) + P2(ni-1, j-1);
            W2(ni, j) = D2 / (2 * Z(ni-1));
            P2(ni, j) = D2 / 2;
        end
    end

    % INVERSÃO 1D
    nt2 = 2 * ni;
    P4 = zeros(ni, nt2); W4 = zeros(ni, nt2); Z4 = zeros(1, ni);

    for j = 1:2:nt2
        P4(1, j) = P2(1, j + 2);
        W4(1, j) = W2(1, j + 2);
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

    stairs(1:ni, Z4, cores(k), 'LineWidth', 1.5);
end

fprintf('Fim');

hold off;
axis([1 ni 0 1.2e7]);
xlabel('Xi');
ylabel('Z');
title('Variando sigma');
grid on;
