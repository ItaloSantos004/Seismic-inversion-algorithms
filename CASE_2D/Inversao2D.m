clear; clc; close all;

%parametros
nt = 300;
ni = 100; %camadas
nx = 21; %sensores
dx = 40; %distancia dos sensores
w = 2*pi*50; %frequencia angular

%interfaces
preal = ones(1, ni) * 1000;
creal   = ones(1, ni) * 1500;

preal(31:60) = 2000;  creal(31:60) = 3000;
preal(61:end) = 1500;  creal(61:end) = 2500;

Zreal = preal .* creal; %impedancia real para comparação

xi = (2*pi / (nx * dx)) * (-floor(nx/2):floor(nx/2)); %xi_m

%gerando dados dos sensores no dominio xi_m
Pteo = zeros(nx, nt);
Wteo = zeros(nx, nt);

fonte = zeros(1, nt);
fonte(3) = 1; %pulso

disp('Inicio');
for m = 1:nx

    Zm = zeros(1, ni); %impedancia obliqua
    for i = 1:ni
        kz = sqrt((w/creal(i))^2 - xi(m)^2);
        if isreal(kz) && kz > 0
            Zm(i) = preal(i) * w / kz;
        else
            Zm(i) = Zm(i-1); %para lidar com ondas evanescentes
        end
    end

    %propagação direta
    P = zeros(ni, nt);
    W = zeros(ni, nt);
    for j = 2:nt
        if mod(1 + j, 2) == 0 %superficie
            P(1, j) = fonte(j);
            U = Zm(1)*W(2, j-1) - P(2, j-1);
            W(1, j) = (P(1, j) + U) / Zm(1);
        end
        for i = 2:ni-1 %interno
            if mod(i + j, 2) == 0
                D = Zm(i-1)*W(i-1, j-1) + P(i-1, j-1);
                U = Zm(i)*W(i+1, j-1) - P(i+1, j-1);
                W(i, j) = (D + U) / (Zm(i-1) + Zm(i));
                P(i, j) = (Zm(i)*D - Zm(i-1)*U) / (Zm(i-1) + Zm(i));
            end
        end
        if mod(ni + j, 2) == 0 %fundo
            D = Zm(ni-1)*W(ni-1, j-1) + P(ni-1, j-1);
            W(ni, j) = D / (2 * Zm(ni-1));
            P(ni, j) = D / 2;
        end
    end

    Pteo(m, :) = P(1, :);
    Wteo(m, :) = W(1, :);
end

%tranformada inversa para voltar pro dominio em x
Psensor = real(ifft(ifftshift(Pteo, 1), [], 1)); %dados reias dos sensores
Wsensor = real(ifft(ifftshift(Wteo, 1), [], 1));

%AGORA É A INVERSÃO

%voltamos nosso dominio para os angulos xi_m
Pcanal = real(fftshift(fft(Psensor, [], 1), 1));
Wcanal = real(fftshift(fft(Wsensor, [], 1), 1));

%inversão
nt2 = 2 * ni; %tempo necessario
Zrec = zeros(nx, ni); %impedancia reconstruida camada x angulo

for m = 1:nx
    Pinv = zeros(ni, nt2);
    Winv = zeros(ni, nt2);
    Zinv = zeros(1, ni);

    %superficie
    for j = 1:2:nt2
        Pinv(1, j) = Pcanal(m, j + 2);
        Winv(1, j) = Wcanal(m, j + 2);
    end

    Zinv(1) = Pinv(1, 1) / Winv(1, 1);

    for i = 2:ni %recursão
        for j = i:2:(nt2 - i + 1)
            a = Winv(i-1, j-1) + Winv(i-1, j+1);
            b = Winv(i-1, j-1) - Winv(i-1, j+1);
            c = Pinv(i-1, j-1) + Pinv(i-1, j+1);
            d = Pinv(i-1, j-1) - Pinv(i-1, j+1);

            Winv(i, j) = 0.5 * (a + d / Zinv(i-1));
            Pinv(i, j) = 0.5 * (Zinv(i-1) * b + c);
        end

        Zinv(i) = Pinv(i, i) / Winv(i, i); %impedancia
    end

    Zrec(m, :) = Zinv;
end

%agora podemos separar a densidade e a velocidade fazendo regressao linear
prec = zeros(1, ni);
crec   = zeros(1, ni);

x = (xi(:).^2);

for i = 1:ni
    y = 1 ./ (real(Zrec(:, i)).^2); %toma a parte real para evitar ruido complexo

    p = polyfit(x, y, 1); %ajuste da reta

    prec(i) = sqrt( -1 / (p(1) * w^2) ); %recuperando os parametros
    crec(i)   = sqrt(  1 / (p(2) * prec(i)^2) );
end

Zrec2 = prec .* crec; %impedancia recuperada
disp('Fim');

%visualização
figure;

%densidade
subplot(3, 1, 1);
stairs(1:ni, preal, 'b', 'LineWidth', 2, 'DisplayName', 'Real'); hold on;
stairs(1:ni, prec, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Inversão');
ylabel('Densidade');
title('Densidade');
legend; grid on; axis tight;

%velocidade
subplot(3, 1, 2);
stairs(1:ni, creal, 'b', 'LineWidth', 2); hold on;
stairs(1:ni, crec, 'r--', 'LineWidth', 1.5);
ylabel('Velocidade');
title('Velocidade');
grid on; axis tight;

%impedancia
subplot(3, 1, 3);
stairs(1:ni, Zreal, 'b', 'LineWidth', 2); hold on;
stairs(1:ni, Zrec2, 'r--', 'LineWidth', 1.5);
xlabel('camada i (zeta)');
ylabel('Impedância');
title('Impedância');
grid on; axis tight;
