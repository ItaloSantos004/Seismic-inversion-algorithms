clear; clc; close all;

%parametros
nt = 300;     
ni = 100;     
nx = 41;      
dx = 40;      
w = 2*pi*50;  

%interfaces
preal = ones(1, ni) * 1000;
creal = ones(1, ni) * 1500;
preal(31:60) = 2000;  creal(31:60) = 3000;
preal(61:end) = 1500;  creal(61:end) = 2500;
Zreal = preal .* creal; 

xi = (2*pi / (nx * dx)) * (-floor(nx/2):floor(nx/2)); 
xi2 = (xi(:).^2); 

%limites para o otimizador
lb = [900, 1400];  
ub = [6000, 4000]; 

opt_fmin = optimset('MaxFunEvals', 1000, 'MaxIter', 1000, 'Display', 'off');

%injeção do ruido SNR + loop da quantidade de vezes
%roda tudo e no fim tira a media
SNR_dB = 10; 
N_tiros = 100; 

prec_tiros = zeros(N_tiros, ni);
crec_tiros = zeros(N_tiros, ni);

disp('Inicio do Loop Grande');

for tiro = 1:N_tiros
    disp(['Loop: ', num2str(tiro), '/', num2str(N_tiros)]);
    
    %modelagem direta 
    Pteo = zeros(nx, nt); 
    Wteo = zeros(nx, nt);
    fonte = zeros(1, nt); 
    fonte(3) = 1; 

    for m = 1:nx
        Zm = zeros(1, ni); 
        for i = 1:ni
            kz = sqrt((w/creal(i))^2 - xi(m)^2);
            if isreal(kz) && kz > 0; Zm(i) = preal(i) * w / kz; else; if i == 1; Zm(i) = preal(i) * creal(i); else; Zm(i) = Zm(i-1); end; end
        end
        P = zeros(ni, nt); W = zeros(ni, nt);
        for j = 2:nt
            if mod(1 + j, 2) == 0; P(1, j) = fonte(j); W(1, j) = (P(1, j) + Zm(1)*W(2, j-1) - P(2, j-1)) / Zm(1); end
            for i = 2:ni-1 
                if mod(i + j, 2) == 0
                    D = Zm(i-1)*W(i-1, j-1) + P(i-1, j-1); U = Zm(i)*W(i+1, j-1) - P(i+1, j-1);
                    W(i, j) = (D + U) / (Zm(i-1) + Zm(i)); P(i, j) = (Zm(i)*D - Zm(i-1)*U) / (Zm(i-1) + Zm(i));
                end
            end
            if mod(ni + j, 2) == 0; D = Zm(ni-1)*W(ni-1, j-1) + P(ni-1, j-1); W(ni, j) = D / (2 * Zm(ni-1)); P(ni, j) = D / 2; end
        end
        Pteo(m, :) = P(1, :); Wteo(m, :) = W(1, :);
    end

    Psensor = real(ifft(ifftshift(Pteo, 1), [], 1)); 
    Wsensor = real(ifft(ifftshift(Wteo, 1), [], 1));
    rms_P = sqrt(mean(Psensor(:).^2));
    rms_W = sqrt(mean(Wsensor(:).^2));
    fator_ruido = 10^(-SNR_dB / 20);
    
    Psensor = Psensor + (fator_ruido * rms_P) * randn(size(Psensor));
    Wsensor = Wsensor + (fator_ruido * rms_W) * randn(size(Wsensor));

    Pcanal = real(fftshift(fft(Psensor, [], 1), 1)); Wcanal = real(fftshift(fft(Wsensor, [], 1), 1)); 

    %agora começa a inversão
    nt2 = 2 * ni; 
    Pinv = zeros(nx, ni, nt2); Winv = zeros(nx, ni, nt2);
    prec = zeros(1, ni); crec = zeros(1, ni);
    Zuni = zeros(nx, 1); 
    
    chute_rho = 1000; 
    chute_c = 1500;

    %Superfície
    Zcam = zeros(nx, 1); 
    for m = 1:nx
        for j = 1:2:nt2; Pinv(m, 1, j) = Pcanal(m, j + 2); Winv(m, 1, j) = Wcanal(m, j + 2); end
        Zcam(m) = Pinv(m, 1, 1) / Winv(m, 1, 1);
    end

    validos = abs(real(Zcam)) > 1e-5 & isfinite(Zcam); 
    if sum(validos) < 2 
        prec(1) = chute_rho; crec(1) = chute_c;
    else
        xi3 = xi2(validos); Z_medido = real(Zcam(validos)); 
        fun_min = @(p) sum( abs( Z_medido - (p(1)*w) ./ sqrt( (w/p(2)).^2 - xi3 ) ).^2 );
        
        try
            %uso sqp pois achei mais preciso, caso não de pra usar uso a minimização abaixo
            pc = sqp([chute_rho, chute_c], fun_min, [], [], lb, ub);
        catch
            fun_min_fmin = @(p) sum( abs( Z_medido - (p(1)*w) ./ sqrt( max((w/p(2)).^2 - xi3, 1e-10) ) ).^2 ) ...
                                + 1e10 * (p(1) < lb(1)) + 1e10 * (p(1) > ub(1)) ...
                                + 1e10 * (p(2) < lb(2)) + 1e10 * (p(2) > ub(2));
            pc = fminsearch(fun_min_fmin, [chute_rho, chute_c], opt_fmin);
        end
                       
        prec(1) = pc(1); crec(1) = pc(2); chute_rho = prec(1); chute_c = crec(1);
    end

    for m = 1:nx
        kzuni = sqrt((w/crec(1))^2 - xi(m)^2);
        if isreal(kzuni) && kzuni > 0; Zuni(m) = prec(1) * w / kzuni; else; Zuni(m) = prec(1) * crec(1); end
    end

    %loop inversao demais camadas
    for i = 2:ni
        Zcam = zeros(nx, 1);
        for m = 1:nx 
            for j = i:2:(nt2 - i + 1)
                a = Winv(m, i-1, j-1) + Winv(m, i-1, j+1); b = Winv(m, i-1, j-1) - Winv(m, i-1, j+1);
                c = Pinv(m, i-1, j-1) + Pinv(m, i-1, j+1); d = Pinv(m, i-1, j-1) - Pinv(m, i-1, j+1);
                Winv(m, i, j) = 0.5 * (a + d / Zuni(m)); Pinv(m, i, j) = 0.5 * (Zuni(m) * b + c);
            end
            Zcam(m) = Pinv(m, i, i) / Winv(m, i, i);
        end
        
        validos = abs(real(Zcam)) > 1e-5 & isfinite(Zcam); 
        if sum(validos) < 2 
            prec(i) = prec(i-1); crec(i) = crec(i-1);
        else
            xi3 = xi2(validos); Z_medido = real(Zcam(validos)); 
            fun_min = @(p) sum( abs( Z_medido - (p(1)*w) ./ sqrt( (w/p(2)).^2 - xi3 ) ).^2 );
            
            try
                pc = sqp([chute_rho, chute_c], fun_min, [], [], lb, ub);
            catch
                fun_min_fmin = @(p) sum( abs( Z_medido - (p(1)*w) ./ sqrt( max((w/p(2)).^2 - xi3, 1e-10) ) ).^2 ) ...
                                    + 1e10 * (p(1) < lb(1)) + 1e10 * (p(1) > ub(1)) ...
                                    + 1e10 * (p(2) < lb(2)) + 1e10 * (p(2) > ub(2));
                pc = fminsearch(fun_min_fmin, [chute_rho, chute_c], opt_fmin);
            end
                           
            prec(i) = pc(1); crec(i) = pc(2); chute_rho = prec(i); chute_c = crec(i);
        end
        
        for m = 1:nx
            kzuni = sqrt((w/crec(i))^2 - xi(m)^2);
            if isreal(kzuni) && kzuni > 0; Zuni(m) = prec(i) * w / kzuni; else; Zuni(m) = prec(i) * crec(i); end
        end
    end
    
    prec_tiros(tiro, :) = prec; crec_tiros(tiro, :) = crec;
end

disp('Tirando a Média Final');


prec_final = mean(prec_tiros, 1);
crec_final = mean(crec_tiros, 1);
Zrec_final = prec_final .* crec_final; 

fig = figure;
subplot(3, 1, 1);
stairs(1:ni, preal, 'b', 'LineWidth', 2.5); hold on;
stairs(1:ni, prec_final, 'r', 'LineWidth', 1.5);
ylabel('Densidade'); 
title('Densidade');
grid on; axis tight;

subplot(3, 1, 2);
stairs(1:ni, creal, 'b', 'LineWidth', 2.5); hold on;
stairs(1:ni, crec_final, 'r', 'LineWidth', 1.5);
ylabel('Velocidade'); 
title('Velocidade');
grid on; axis tight;

subplot(3, 1, 3);
stairs(1:ni, Zreal, 'b', 'LineWidth', 2.5); hold on;
stairs(1:ni, Zrec_final, 'r', 'LineWidth', 1.5);
xlabel('Camada i (Profundidade)'); 
ylabel('Impedancia'); 
title('Impedancia');
grid on; axis tight;

nome = sprintf('Resultado_%dvezes_SNR%ddB.png', N_tiros, SNR_dB);
print(fig, nome, '-dpng', '-r300'); 
disp('Imagem salva');