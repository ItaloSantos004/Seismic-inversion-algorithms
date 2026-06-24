clear; clc; close all;

%parametros
nt = 300;
ni = 100; %camadas
nx = 101; %sensores
dx = 40; %espaçamento 
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
            if i == 1
                Zm(i) = preal(i) * creal(i);
            else
                Zm(i) = Zm(i-1); %para lidar com ondas evanescentes internas
            end
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

disp('Inicio da inversão')

%injeção de ruido
erro = 0.0;

Psensor = Psensor + erro * max(abs(Psensor(:))) * randn(size(Psensor));
Wsensor = Wsensor + erro * max(abs(Wsensor(:))) * randn(size(Wsensor));

%voltamos nosso dominio para os angulos xi_m
Pcanal = real(fftshift(fft(Psensor, [], 1), 1));
Wcanal = real(fftshift(fft(Wsensor, [], 1), 1)); 

nt2 = 2 * ni; %tempo necessario

%matrizes 3D para propagar os sensores ao mesmo tempo
Pinv = zeros(nx, ni, nt2);
Winv = zeros(nx, ni, nt2);

prec = zeros(1, ni); %valores reconstruidos
crec = zeros(1, ni);
Zuni = zeros(nx, 1); %impedancia unificada

xi2 = (xi(:).^2);
chute_rho = 1000;
chute_c = 1500;
options = optimset('Display', 'off', 'TolX', 1e-4, 'TolFun', 1e-4);

%Superficie
Zcam = zeros(nx, 1); %impedancia dos sensores na camada

for m = 1:nx
    for j = 1:2:nt2
        Pinv(m, 1, j) = Pcanal(m, j + 2);
        Winv(m, 1, j) = Wcanal(m, j + 2);
    end
    Zcam(m) = Pinv(m, 1, 1) / Winv(m, 1, 1);
end

%minimização
validos = abs(real(Zcam)) > 1e-5; 

if sum(validos) < 2 
    prec(1) = chute_rho;
    crec(1) = chute_c;
else
    xi3 = xi2(validos);
    Z_medido = real(Zcam(validos)); 
    
    %minimização
    fun_min = @(p) sum( abs( Z_medido - (p(1)*w) ./ sqrt( (w/p(2)).^2 - xi3 ) ).^2 ) ...
                        + 1e12 * (p(1) <= 0) ...      
                        + 1e12 * (p(2) <= 0) ...      
                        + 1e12 * ((w/p(2))^2 <= max(xi3)); 
                        
    pc = fminsearch(fun_min, [chute_rho, chute_c], options); %par pressao e velocidade
    prec(1) = pc(1);
    crec(1) = pc(2);
    chute_rho = prec(1);
    chute_c = crec(1);
end

%impedancia unificada para a camada 1
for m = 1:nx
    kzuni = sqrt((w/crec(1))^2 - xi(m)^2);
    if isreal(kzuni) && kzuni > 0
        Zuni(m) = prec(1) * w / kzuni;
    else
        Zuni(m) = prec(1) * crec(1);
    end
end

%mesma coisa para as demais camadas
for i = 2:ni
    
    Zcam = zeros(nx, 1);
    
    for m = 1:nx %utlizamos a impedancia unificada
        for j = i:2:(nt2 - i + 1)
            a = Winv(m, i-1, j-1) + Winv(m, i-1, j+1);
            b = Winv(m, i-1, j-1) - Winv(m, i-1, j+1);
            c = Pinv(m, i-1, j-1) + Pinv(m, i-1, j+1);
            d = Pinv(m, i-1, j-1) - Pinv(m, i-1, j+1);

            Winv(m, i, j) = 0.5 * (a + d / Zuni(m));
            Pinv(m, i, j) = 0.5 * (Zuni(m) * b + c);
        end
        Zcam(m) = Pinv(m, i, i) / Winv(m, i, i);
    end
    
    %minimização
    Zcam = Zcam;
    validos = abs(real(Zcam)) > 1e-5; 
    
    if sum(validos) < 2 
        prec(i) = prec(i-1);
        crec(i) = crec(i-1);
    else
        xi3 = xi2(validos);
        Z_medido = real(Zcam(validos)); 
        
        fun_min = @(p) sum( abs( Z_medido - (p(1)*w) ./ sqrt( (w/p(2)).^2 - xi3 ) ).^2 ) ...
                            + 1e12 * (p(1) <= 0) ...      
                            + 1e12 * (p(2) <= 0) ...      
                            + 1e12 * ((w/p(2))^2 <= max(xi3)); 
                            
        pc = fminsearch(fun_min, [chute_rho, chute_c], options);
        prec(i) = pc(1);
        crec(i) = pc(2);
        chute_rho = prec(i);
        chute_c = crec(i);
    end
    
    %calcula a impedancia unificada para ser usada
    for m = 1:nx
        kzuni = sqrt((w/crec(i))^2 - xi(m)^2);
        if isreal(kzuni) && kzuni > 0
            Zuni(m) = prec(i) * w / kzuni;
        else
            Zuni(m) = prec(i) * crec(i);
        end
    end
end

Zrec2 = prec .* crec; %impedancia recuperada

disp('Fim');

%visualização
fig = figure;

%densidade
subplot(3, 1, 1);
stairs(1:ni, preal, 'b', 'LineWidth', 2.5); hold on;
stairs(1:ni, prec, 'r', 'LineWidth', 1.5);
ylabel('Densidade');
title('Densidade');
grid on; axis tight;

%velocidade
subplot(3, 1, 2);
stairs(1:ni, creal, 'b', 'LineWidth', 2.5); hold on;
stairs(1:ni, crec, 'r', 'LineWidth', 1.5);
ylabel('Velocidade');
title('Velocidade');
grid on; axis tight;

%impedancia
subplot(3, 1, 3);
stairs(1:ni, Zreal, 'b', 'LineWidth', 2.5); hold on;
stairs(1:ni, Zrec2, 'r', 'LineWidth', 1.5);
xlabel('camada i (zeta)');
ylabel('Impedância');
title('Impedância');
grid on; axis tight;

%salvando a imagem
nome = sprintf('Resultado_Erro_%.3f.png', erro);
print(fig, nome, '-dpng', '-r300'); 
disp('Imagem salva');