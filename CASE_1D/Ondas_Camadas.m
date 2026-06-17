clear; clc; close all;

%Grade
z = -50:1:500;
nz = length(z);
dz = z(2) - z(1);

%Camadas
interfaces_z = [100, 200 300, 400];
c_vals = [1500, 1700, 2500, 4000, 4500];
p_vals = [1000, 1300, 1800, 2500, 3000];

C = ones(1, nz) * c_vals(1);
P = ones(1, nz) * p_vals(1);

for k = 1:length(interfaces_z) %preenchendo nossa grade
    C(z >= interfaces_z(k)) = c_vals(k+1);
    P(z >= interfaces_z(k)) = p_vals(k+1);
end

%Tempo
dt = dz / (10 * max(C));
t = 0:dt:0.5;
nt = length(t);

U = zeros(nt, nz);

%Condição Inicial
z0 = 0;
U(1, :) = exp( -(z-z0).^2 / 100);
U(2, :) = exp( -(z-z0).^2 / 100);

%identificar os indices da interface
is_interface = false(1, nz);
for i = 2:nz-1
    if P(i+1) ~= P(i-1) || C(i+1) ~= C(i-1)
        is_interface(i) = true;
    end
end

%indices dos pontos regulares
idx_reg = find(~is_interface);
idx_reg = idx_reg(idx_reg > 1 & idx_reg < nz);

%pontos interface
idx_int = find(is_interface);

%coeficientes
alpha = (C .* dt / dz);
coef = (alpha - 1) ./ (alpha + 1);
gamma = (C .* dt / dz).^2;

for n = 2:nt-1
    %pontos regulares
    U(n+1, idx_reg) = 2*U(n, idx_reg) - U(n-1, idx_reg) + ...
        gamma(idx_reg) .* (U(n, idx_reg+1) - 2*U(n, idx_reg) + U(n, idx_reg-1));

    %pontos interface
    if ~isempty(idx_int)

        pm = P(idx_int-1); cm = C(idx_int-1);
        pp = P(idx_int+1); cp = C(idx_int+1);

        num = ( (U(n, idx_int+1) - U(n, idx_int))./(pp*dz) ) - ...
              ( (U(n, idx_int) - U(n, idx_int-1))./(pm*dz) );
        den = (dz/2) * ( (1./(pp.*cp.^2)) + (1./(pm.*cm.^2)) );

        u_tt_int = num ./ den;
        U(n+1, idx_int) = 2*U(n, idx_int) - U(n-1, idx_int) + (dt^2 * u_tt_int);
    end

    %condiçoes de contorno
    U(n+1, 1) = U(n, 2) + coef(1) * (U(n+1, 2) - U(n, 1));
    U(n+1, nz) = U(n, nz-1) + coef(nz) * (U(n+1, nz-1) - U(n, nz));
end

%Visualização
figure('Color', 'w');
for n = 1:80:nt
    plot(z, U(n, :), 'LineWidth', 1.5);
    hold on;
    for k = 1:length(interfaces_z)
        line([interfaces_z(k) interfaces_z(k)], [-0.5 1.2], 'Color', 'r', 'LineStyle', '--');
    end
    hold off;
    axis([min(z) max(z) -0.5 1.2]);
    title(['Simulação - Tempo: ', num2str(t(n), '%.3f'), 's']);
    drawnow;
end
