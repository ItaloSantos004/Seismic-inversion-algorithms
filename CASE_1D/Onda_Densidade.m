clear; clc; close all;

z = -50:1:500;
nz = length(z);
zc = 200;
[~, idx_c] = min(abs(z - zc));

%CAMADAS
c1 = 1500;
p1 = 1000;

c2 = 3000;
p2 = 2200;


C = zeros(1, nz);
P = zeros(1, nz);
C(z < zc) = c1; C(z >= zc) = c2;
P(z < zc) = p1; P(z >= zc) = p2;

dz = z(2) - z(1);
dt = dz / (10 * max(C));
t = 0:dt:0.5;
nt = length(t);

U = zeros(nt, nz);

%CONDIÇÃO INICIAL
z0 = 0;
U(1, :) = exp( -(z-z0).^2 / 100);
U(2, :) = exp( -(z-z0).^2 / 100);

alpha = (C .* dt / dz);
coef = (alpha - 1) ./ (alpha + 1);

for n = 2:nt-1
    %PONTOS REGULARES
    for i = 2:idx_c-1
        gamma = (c1 * dt / dz)^2;
        U(n+1, i) = 2*U(n, i) - U(n-1, i) + gamma * (U(n, i+1) - 2*U(n, i) + U(n, i-1));
    end

    for i = idx_c+1:nz-1
        gamma = (c2 * dt / dz)^2;
        U(n+1, i) = 2*U(n, i) - U(n-1, i) + gamma * (U(n, i+1) - 2*U(n, i) + U(n, i-1));
    end

    %CAMADAS
    num = ( (U(n, idx_c+1) - U(n, idx_c))/(p2*dz) ) - ( (U(n, idx_c) - U(n, idx_c-1))/(p1*dz) );
    den = (dz/2) * ( (1/(p2*c2^2)) + (1/(p1*c1^2)) );

    u_tt_interface = num / den;

    % Atualização do ponto da interface
    U(n+1, idx_c) = 2*U(n, idx_c) - U(n-1, idx_c) + (dt^2 * u_tt_interface);

    % 3. Condições de Contorno (Bordas)
    U(n+1, 1) = U(n, 2) + coef(1) * (U(n+1, 2) - U(n, 1));
    U(n+1, nz) = U(n, nz-1) + coef(nz) * (U(n+1, nz-1) - U(n, nz));
end

% --- Visualização ---
figure('Color', 'w');
for n = 1:80:nt
    plot(z, U(n, :), 'Color', [0 0.4 0.7], 'LineWidth', 1);
    hold on;
    % Desenha a interface
    line([zc zc], [-0.5 1.2], 'Color', 'k', 'LineStyle', '--');
    hold off;

    axis([min(z) max(z) -0.5 1.2]);
    title(['Tempo: ', num2str(t(n), '%.3f'), 's']);
    xlabel('Posição (z)'); ylabel('Amplitude');
    grid on; drawnow;
end
