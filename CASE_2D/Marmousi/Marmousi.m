% =========================================================================
% SCRIPT: IMPORTAÇÃO MARMOUSI2 COM DECODIFICADOR IBM FLOAT
% =========================================================================
clear; clc; close all;

file_vp = 'MODEL_P-WAVE_VELOCITY_1.25m.segy';
file_rho = 'MODEL_DENSITY_1.25m.segy';

% Função interna para ler a matriz contornando o formato IBM
function [matrix, dx, dz] = ler_marmousi_ibm(filename)
    fid = fopen(filename, 'r', 'ieee-be'); 
    if fid == -1
        error(['CRÍTICO: Não foi possível abrir o arquivo: ', filename]);
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

    disp(['-> Lendo e Decodificando (IBM Float): ', filename]);
    disp(['   Dimensões: ', num2str(ns), ' linhas x ', num2str(ntraces), ' colunas']);

    matrix = zeros(ns, ntraces);
    fseek(fid, 3600, 'bof');

    for i = 1:ntraces
        fseek(fid, 240, 'cof'); 
        
        % 1. Lê a coluna como inteiros puros de 32 bits (preserva os bits originais)
        raw_bytes = fread(fid, ns, 'uint32');
        
        % 2. Separa os bits (Sinal, Expoente e Fração)
        S = bitshift(raw_bytes, -31);                     % 1 bit de sinal
        C = bitand(bitshift(raw_bytes, -24), 127);        % 7 bits de expoente
        F = bitand(raw_bytes, 16777215);                  % 24 bits de fração (2^24 - 1)
        
        % 3. Aplica a equação do IBM Float (Base 16)
        % 16^6 = 16777216
        coluna_real = (1 - 2.*S) .* (F / 16777216) .* (16.^(C - 64));
        
        matrix(:, i) = coluna_real;
    end

    fclose(fid);
    dx = 1.25; 
    dz = 1.25; 
end

disp('===================================================');
disp(' INICIANDO EXTRAÇÃO DE DADOS (Aguarde...)');
disp('===================================================');

% 3. Executa a leitura
[Vp, dx, dz]  = ler_marmousi_ibm(file_vp);
[Rho, dx, dz] = ler_marmousi_ibm(file_rho);

% CONVERSÃO DE UNIDADE (g/cm³ para kg/m³)
Rho = Rho * 1000;

% =========================================================================
% OTIMIZAÇÃO E EXPORTAÇÃO PARA O PYTHON
% =========================================================================
% Converte para 32-bits (single) para economizar RAM e acelerar o Numba
Vp = single(Vp);
Rho = single(Rho);
dx = single(dx);
dz = single(dz);

disp('Salvando matrizes no formato .mat (Versão 7 para compatibilidade com Python)...');
% O -v7 garante que a biblioteca scipy.io do Python consiga ler o arquivo
save('-v7', 'marmousi_matrizes.mat', 'Vp', 'Rho', 'dx', 'dz');
disp('Salvo com sucesso! Pode fechar este script.');

% =========================================================================
% VISUALIZAÇÃO
% =========================================================================
disp('===================================================');
disp(' DADOS CARREGADOS E SALVOS! GERANDO IMAGEM...');
disp('===================================================');

x_km = (0:(size(Vp, 2)-1)) * double(dx) / 1000;
z_km = (0:(size(Vp, 1)-1)) * double(dz) / 1000;

fig = figure('Position', [100, 100, 1200, 700], 'Name', 'Marmousi2 - Modelo Acústico Absoluto');

% --- Gráfico Superior: Velocidade P ---
subplot(2, 1, 1);
imagesc(x_km, z_km, Vp);
colormap(gca, 'jet'); 
cb1 = colorbar; 
title(cb1, 'V_p (m/s)', 'FontWeight', 'bold');
title('Marmousi - Velocidade', 'FontSize', 14);
xlabel('X (km)'); 
ylabel('Profundidade (km)');
axis tight;

% --- Gráfico Inferior: Densidade ---
subplot(2, 1, 2);
imagesc(x_km, z_km, Rho);
colormap(gca, 'viridis'); % Ajustado para funcionar perfeitamente no Octave
cb2 = colorbar; 
title(cb2, 'Densidade (kg/m^3)', 'FontWeight', 'bold');
title('Marmousi - Densidade', 'FontSize', 14);
xlabel('X (km)'); 
ylabel('Profundidade (km)');
axis tight;

% =========================================================================
% SALVAR FIGURA
% =========================================================================
nome_arquivo = 'Marmousi2_Modelo_Acustico.png';

print(fig, nome_arquivo, '-dpng', '-r300');  % PNG com 300 dpi
% Opcionalmente, também salvar em PDF:
% print(fig, 'Marmousi2_Modelo_Acustico.pdf', '-dpdf');

disp(['-> Figura salva como: ', nome_arquivo]);

disp('-> Processo Concluído com Sucesso!');