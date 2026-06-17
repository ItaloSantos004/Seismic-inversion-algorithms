clear; clc; close all;

%Dominio
L = 1000;

ni = 400;
nt = 4 * ni;
dz = L / ni;

z = (1:ni) * dz;

%perfil de impedancia
Z = 2e6 + 20 * (z - L/2).^2;

%Z = 2e6 + 8000 * z;

%Z = 2e6 + 6e6 * exp(-((z - 500).^2) / 10000);

%Z = 4e6 + 1.5e6 * sin(2 * pi * z / 250);

P1 = zeros(ni, nt);
W1 = zeros(ni, nt);

P2 = zeros(ni, nt);
W2 = zeros(ni, nt);

%fonte
t0 = 100;
s = 15;
fonte1 = 1000 * exp(-((1:nt) - t0).^2 / (2*s^2));

fonte2 = zeros(1, nt);
fonte2(3) = 1;

%direto
for j = 2:nt

    if mod(1 + j, 2) == 0 % superficie
        P1(1, j) = fonte1(j);
        U1 = Z(1)*W1(2, j-1) - P1(2, j-1);
        W1(1, j) = (P1(1, j) + U1) / Z(1);

        P2(1, j) = fonte2(j);
        U2 = Z(1)*W2(2, j-1) - P2(2, j-1);
        W2(1, j) = (P2(1, j) + U2) / Z(1);
    end

    for i = 2:ni-1 % pontos dentro
        if mod(i + j, 2) == 0
            D1 = Z(i-1)*W1(i-1, j-1) + P1(i-1, j-1);
            U1 = Z(i)*W1(i+1, j-1) - P1(i+1, j-1);
            W1(i, j) = (D1 + U1) / (Z(i-1) + Z(i));
            P1(i, j) = (Z(i)*D1 - Z(i-1)*U1) / (Z(i-1) + Z(i));

            D2 = Z(i-1)*W2(i-1, j-1) + P2(i-1, j-1);
            U2 = Z(i)*W2(i+1, j-1) - P2(i+1, j-1);
            W2(i, j) = (D2 + U2) / (Z(i-1) + Z(i));
            P2(i, j) = (Z(i)*D2 - Z(i-1)*U2) / (Z(i-1) + Z(i));
        end
    end

    if mod(ni + j, 2) == 0 % fundo
        D1 = Z(ni-1)*W1(ni-1, j-1) + P1(ni-1, j-1);
        W1(ni, j) = D1 / (2 * Z(ni-1));
        P1(ni, j) = D1 / 2;

        D2 = Z(ni-1)*W2(ni-1, j-1) + P2(ni-1, j-1);
        W2(ni, j) = D2 / (2 * Z(ni-1));
        P2(ni, j) = D2 / 2;
    end
end

%inversao
nt2 = 2 * ni;

P4 = zeros(ni, nt2);
W4 = zeros(ni, nt2);
Z4 = zeros(1, ni);

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

%visualização
figure;
plot(z, Z, 'b', 'LineWidth', 1.5);
hold on;
stairs(z, Z4, 'r', 'LineWidth', 1.5);

hold off;

axis([1 L 0 1.2e7]);
xlabel('m');
ylabel('Z');
title('Perfil continuo');
grid on;
