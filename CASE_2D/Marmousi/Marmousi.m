clear; clc; close all;

file_vp = 'MODEL_P-WAVE_VELOCITY_1.25m.segy';
file_rho = 'MODEL_DENSITY_1.25m.segy';

%função para ler os dados
function [matrix, dx, dz] = ler_marmousi_ibm(filename)
    fid = fopen(filename, 'r', 'ieee-be'); 
    if fid == -1
        error(['erro: ', filename]);
    end

    fseek(fid, 3200, 'bof');
    fseek(fid, 3220, 'bof');
    ns = fread(fid, 1, 'int16'); 

    if ns <= 0 || ns > 10000 
        ns = 2800; 
    end

    fseek(fid, 0, 'eof');
    file_size = ftell(fid);
    trace_size = 240 + (ns * 4);
    ntraces = floor((file_size - 3600) / trace_size);

    disp(['Dimensões: ', num2str(ns), ' linhas x ', num2str(ntraces), ' colunas']);

    matrix = zeros(ns, ntraces);
    fseek(fid, 3600, 'bof');

    for i = 1:ntraces
        fseek(fid, 240, 'cof'); 

        raw_bytes = fread(fid, ns, 'uint32');
        
        S = bitshift(raw_bytes, -31);               
        C = bitand(bitshift(raw_bytes, -24), 127);     
        F = bitand(raw_bytes, 16777215);     

        coluna_real = (1 - 2.*S) .* (F / 16777216) .* (16.^(C - 64));
        
        matrix(:, i) = coluna_real;
    end

    fclose(fid);
    dx = 1.25; 
    dz = 1.25; 
end

disp('Extraindo os dados');

%extraindo os dados
[Vp, dx, dz]  = ler_marmousi_ibm(file_vp);
[Rho, dx, dz] = ler_marmousi_ibm(file_rho);

Rho = Rho * 1000;

%convertendo para 32 bits
Vp = single(Vp);
Rho = single(Rho);
dx = single(dx);
dz = single(dz);

disp('Salvando matrizes');
save('-v7', 'marmousi_matrizes.mat', 'Vp', 'Rho', 'dx', 'dz');

%visualização
disp('Gerando a imagem');

x_km = (0:(size(Vp, 2)-1)) * double(dx) / 1000;
z_km = (0:(size(Vp, 1)-1)) * double(dz) / 1000;

fig = figure;

%velocidade
subplot(2, 1, 1);
imagesc(x_km, z_km, Vp);
colormap(gca, 'jet'); 
cb1 = colorbar; 
title(cb1, 'Velocidade (m/s)', 'FontWeight', 'bold');
title('Marmousi - Velocidade', 'FontSize', 14);
xlabel('X (km)'); 
ylabel('Profundidade (km)');
axis tight;

%densidade
subplot(2, 1, 2);
imagesc(x_km, z_km, Rho);
colormap(gca, 'viridis'); 
cb2 = colorbar; 
title(cb2, 'Densidade (kg/m^3)', 'FontWeight', 'bold');
title('Marmousi - Densidade', 'FontSize', 14);
xlabel('X (km)'); 
ylabel('Profundidade (km)');
axis tight;

%salvando a imagem
nome = 'Modelo_Original_Marmousi.png';

print(fig, nome, '-dpng', '-r300');

disp('Imagem salva');