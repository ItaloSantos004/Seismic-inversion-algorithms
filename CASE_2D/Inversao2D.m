clear; clc; close all;

%parametros
nt = 300;
ni = 100; %camadas
nx = 21; %sensores
dx = 40; %distancia dos sensores
omega0 = 2*pi*50; %frequencia angular

%interfaces
rho_real = ones(1, ni) * 1000;
c_real   = ones(1, ni) * 1500;

rho_real(31:60) = 2000;  c_real(31:60) = 3000;
rho_real(61:end) = 1500;  c_real(61:end) = 2500;

Z_normal_real = rho_real .* c_real; %impedancia real para comparação

xi = (2*pi / (nx * dx)) * (-floor(nx/2):floor(nx/2)); %xi_m

%gerando dados dos sensores no dominio xi_m
P2_teorico = zeros(nx, nt);
W2_teorico = zeros(nx, nt);

fonte2 = zeros(1, nt);
fonte2(3) = 1; %pulso

disp('Inicio');
for m = 1:nx

    Z_m = zeros(1, ni); %impedancia obliqua
    for i = 1:ni
        kz = sqrt((omega0/c_real(i))^2 - xi(m)^2);
        if isreal(kz) && kz > 0
            Z_m(i) = rho_real(i) * omega0 / kz;
        else
            Z_m(i) = Z_m(i-1); %para lidar com ondas evanescentes
        end
    end

    %propagação direta
    P2 = zeros(ni, nt); W2 = zeros(ni, nt);
    for j = 2:nt
        if mod(1 + j, 2) == 0 %superficie
            P2(1, j) = fonte2(j);
            U2 = Z_m(1)*W2(2, j-1) - P2(2, j-1);
            W2(1, j) = (P2(1, j) + U2) / Z_m(1);
        end
        for i = 2:ni-1 %interno
            if mod(i + j, 2) == 0
                D2 = Z_m(i-1)*W2(i-1, j-1) + P2(i-1, j-1);
                U2 = Z_m(i)*W2(i+1, j-1) - P2(i+1, j-1);
                W2(i, j) = (D2 + U2) / (Z_m(i-1) + Z_m(i));
                P2(i, j) = (Z_m(i)*D2 - Z_m(i-1)*U2) / (Z_m(i-1) + Z_m(i));
            end
        end
        if mod(ni + j, 2) == 0 %fundo
            D2 = Z_m(ni-1)*W2(ni-1, j-1) + P2(ni-1, j-1);
            W2(ni, j) = D2 / (2 * Z_m(ni-1));
            P2(ni, j) = D2 / 2;
        end
    end

    P2_teorico(m, :) = P2(1, :);
    W2_teorico(m, :) = W2(1, :);
end

%tranformada inversa para voltar pro dominio em x
p_sensores = real(ifft(ifftshift(P2_teorico, 1), [], 1)); %dados reias dos sensores
w_sensores = real(ifft(ifftshift(W2_teorico, 1), [], 1));

%AGORA É A INVERSÃO

%voltamos nosso dominio para os angulos xi_m
P2_canais = real(fftshift(fft(p_sensores, [], 1), 1));
W2_canais = real(fftshift(fft(w_sensores, [], 1), 1));

%inversão
nt2 = 2 * ni; % Tempo causal necessário
Z_reconstruido_2D = zeros(nx, ni);

for m = 1:nx
    P4 = zeros(ni, nt2);
    W4 = zeros(ni, nt2);
    Z4 = zeros(1, ni);

    %superficie
    for j = 1:2:nt2
        P4(1, j) = P2_canais(m, j + 2);
        W4(1, j) = W2_canais(m, j + 2);
    end

    Z4(1) = P4(1, 1) / W4(1, 1);

    for i = 2:ni %recursão
        for j = i:2:(nt2 - i + 1)
            a = W4(i-1, j-1) + W4(i-1, j+1);
            b = W4(i-1, j-1) - W4(i-1, j+1);
            c = P4(i-1, j-1) + P4(i-1, j+1);
            d = P4(i-1, j-1) - P4(i-1, j+1);

            W4(i, j) = 0.5 * (a + d / Z4(i-1));
            P4(i, j) = 0.5 * (Z4(i-1) * b + c);
        end

        Z4(i) = P4(i, i) / W4(i, i); %impedancia
    end

    Z_reconstruido_2D(m, :) = Z4;
end

%agora podemos separar a densidade e a velocidade fazendo regressao linear
rho_est = zeros(1, ni);
c_est   = zeros(1, ni);

x_reg = (xi(:).^2);

for i = 1:ni
    y_reg = 1 ./ (real(Z_reconstruido_2D(:, i)).^2); %toma a parte real para evitar ruido complexo

    p = polyfit(x_reg, y_reg, 1); %ajuste da reta

    B = min(p(1), -1e-20); %para manter a inclinação sempre negativa
    A = max(p(2),  1e-20); %para manter a interseção sempre positiva

    rho_est(i) = sqrt( -1 / (B * omega0^2) ); %recuperando os parametros
    c_est(i)   = sqrt(  1 / (A * rho_est(i)^2) );
end

Z_normal_est = rho_est .* c_est; %impedancia recuperada
disp('Fim');

%visualização
figure;

%densidade
subplot(3, 1, 1);
stairs(1:ni, rho_real, 'b', 'LineWidth', 2, 'DisplayName', 'Real'); hold on;
stairs(1:ni, rho_est, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Inversão');
ylabel('Densidade');
title('Densidade');
legend; grid on; axis tight;

%velocidade
subplot(3, 1, 2);
stairs(1:ni, c_real, 'b', 'LineWidth', 2); hold on;
stairs(1:ni, c_est, 'r--', 'LineWidth', 1.5);
ylabel('Velocidade');
title('Velocidade');
grid on; axis tight;

%impedancia
subplot(3, 1, 3);
stairs(1:ni, Z_normal_real, 'b', 'LineWidth', 2); hold on;
stairs(1:ni, Z_normal_est, 'r--', 'LineWidth', 1.5);
xlabel('camada i (zeta)');
ylabel('Impedância');
title('Impedância');
grid on; axis tight;
