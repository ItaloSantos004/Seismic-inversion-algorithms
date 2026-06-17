clear; clc; close all;

% Grade Espacial
z = -50:1:500;
nz = length(z);
dz = z(2) - z(1);

% Camadas e Propriedades
interfaces_z = [100, 200, 300, 400];
c_vals = [1500, 1700, 2500, 4000, 4500];
p_vals = [1000, 1300, 1800, 2500, 3000];

C = ones(1, nz) * c_vals(1);
P = ones(1, nz) * p_vals(1);

for k = 1:length(interfaces_z) % Preenchendo a grade com as velocidades e densidades
    C(z >= interfaces_z(k)) = c_vals(k+1);
    P(z >= interfaces_z(k)) = p_vals(k+1);
end

% Tempo
dt = dz / (10 * max(C));
t = 0:dt:0.5;
nt = length(t);

U = zeros(nt, nz);

% Condição Inicial (Pulso Gaussiano)
z0 = 0;
U(1, :) = exp( -(z-z0).^2 / 100);
U(2, :) = exp( -(z-z0).^2 / 100);

% --- IDENTIFICAÇÃO CORRETA DAS INTERFACES ---
is_interface = ismember(z, interfaces_z);

% Índices dos pontos regulares (excluindo bordas e interfaces)
idx_reg = find(~is_interface);
idx_reg = idx_reg(idx_reg > 1 & idx_reg < nz);

% Índices dos pontos de interface
idx_int = find(is_interface);

% Coeficientes
alpha = (C .* dt / dz);
coef = (alpha - 1) ./ (alpha + 1);
gamma = (C .* dt / dz).^2;

% Loop Principal no Tempo (Esquema Explícito)
for n = 2:nt-1
    % Pontos regulares (Equação da Onda Padrão)
    U(n+1, idx_reg) = 2*U(n, idx_reg) - U(n-1, idx_reg) + ...
        gamma(idx_reg) .* (U(n, idx_reg+1) - 2*U(n, idx_reg) + U(n, idx_reg-1));

    % Pontos de interface (Condição de continuidade)
    if ~isempty(idx_int)
        pm = P(idx_int-1); cm = C(idx_int-1);
        pp = P(idx_int+1); cp = C(idx_int+1);

        num = ( (U(n, idx_int+1) - U(n, idx_int))./(pp*dz) ) - ...
              ( (U(n, idx_int) - U(n, idx_int-1))./(pm*dz) );
        den = (dz/2) * ( (1./(pp.*cp.^2)) + (1./(pm.*cm.^2)) );

        u_tt_int = num ./ den;
        U(n+1, idx_int) = 2*U(n, idx_int) - U(n-1, idx_int) + (dt^2 * u_tt_int);
    end

    % Condições de contorno (Absorventes simples de 1ª ordem)
    U(n+1, 1) = U(n, 2) + coef(1) * (U(n+1, 2) - U(n, 1));
    U(n+1, nz) = U(n, nz-1) + coef(nz) * (U(n+1, nz-1) - U(n, nz));
end

% ==========================================
% VISUALIZAÇÃO
% ==========================================

% 1. Animação da Propagação
figure('Color', 'w', 'Name', 'Animação da Onda');
for n = 1:80:nt
    plot(z, U(n, :), 'LineWidth', 1.5, 'Color', 'b');
    hold on;
    for k = 1:length(interfaces_z)
        line([interfaces_z(k) interfaces_z(k)], [-0.5 1.2], 'Color', 'r', 'LineStyle', '--');
    end
    hold off;
    axis([min(z) max(z) -0.5 1.2]);
    title(['Simulação - Tempo: ', num2str(t(n), '%.3f'), 's']);
    xlabel('Profundidade (z)');
    ylabel('Amplitude (U)');
    drawnow;
end

% 2. Gráfico de Intensidade (Tempo vs Profundidade) - Corrigido para Octave
figure('Color', 'w', 'Name', 'Mapa de Intensidade (Wavefield)');

% Downsampling: desenhando 1 a cada 20 passos para não estourar a memória de vídeo do Octave
passo = 20;
t_plot = t(1:passo:end);
U_plot = U(1:passo:end, :);

imagesc(z, t_plot, U_plot);
colormap(jet);
colorbar;
xlabel('Profundidade (z)');
ylabel('Tempo (t) [s]');
title('Evolução do Campo de Onda no Espaço-Tempo');

hold on;
% Linhas das interfaces no mapa de intensidade
for k = 1:length(interfaces_z)
    line([interfaces_z(k) interfaces_z(k)], [0 max(t_plot)], 'Color', 'w', 'LineStyle', '--', 'LineWidth', 1);
end
hold off;

% 3. Gráfico Fixo: Comparação Propriedades vs Snapshot Final - Corrigido para Octave
figure('Color', 'w', 'Name', 'Snapshot e Propriedades');

% Subplot para Velocidade
subplot(2,1,1);
plot(z, C, 'k', 'LineWidth', 2);
grid on;
ylabel('Velocidade (m/s)');
title('Perfil de Propriedades do Meio');
xlim([min(z) max(z)]);

% Subplot para o Snapshot Final da Onda
subplot(2,1,2);
plot(z, U(end, :), 'b', 'LineWidth', 1.5);
hold on;
% Desenhando as linhas com 'line' no lugar do 'xline'
yl = [-0.5 1.2]; % Forçando os limites verticais
for k = 1:length(interfaces_z)
    line([interfaces_z(k) interfaces_z(k)], yl, 'Color', 'r', 'LineStyle', '--');
end
grid on;
xlabel('Profundidade (z)');
ylabel('Amplitude U(z, t_{final})');
title('Snapshot da Onda no Último Instante de Tempo');
xlim([min(z) max(z)]);
ylim(yl);
