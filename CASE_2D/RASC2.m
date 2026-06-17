clear; clc; close all;

%parametros
M = 4; %ordem
Nx = 150;
Nz = 200;
h = 8.5;
dt = 0.002;
tempo = 800;

m = max(1, M);
U = zeros(Nx, Nz, tempo + m);

%camadas
c_vals = [1500, 2500];
p_vals = [1000, 1800];

interface_meio = round(Nz / 2);

C_matrix = ones(Nx, Nz) * c_vals(1);
Rho_matrix = ones(Nx, Nz) * p_vals(1);

C_matrix(:, interface_meio:end) = c_vals(2);
Rho_matrix(:, interface_meio:end) = p_vals(2);

K_matrix = Rho_matrix .* (C_matrix.^2);

%separando para convolução
%superior
S_sup = c_vals(1) * dt / h;
c1_sup = (S_sup - 1) / (S_sup + 1);
c2_sup = [1, -c1_sup; c1_sup, -1];
c3_sup = 1;

%inferior
S_inf = c_vals(2) * dt / h;
c1_inf = (S_inf - 1) / (S_inf + 1);
c2_inf = [1, -c1_inf; c1_inf, -1];
c3_inf = 1;

%interface
rho_media = (p_vals(1) + p_vals(2)) / 2;
K_media = (p_vals(1)*c_vals(1)^2 + p_vals(2)*c_vals(2)^2) / 2;
c_int = sqrt(K_media / rho_media);

S_int = c_int * dt / h;
c1_int = (S_int - 1) / (S_int + 1);
c2_int = [1, -c1_int; c1_int, -1];
c3_int = 1;

if M > 0
    for idx = 1:M
        c3_sup = conv2(c3_sup, c2_sup);
        c3_inf = conv2(c3_inf, c2_inf);
        c3_int = conv2(c3_int, c2_int);
    end
end

%fonte
t0 = 0.1;
s = 0.02;
x0 = round(Nx/2);
z0 = 2;

figure;

for n = 1:tempo
    t = n * dt;
    p = n + m;

    %interior
    i = 2:Nx-1;
    j = 2:Nz-1;

    rho_x_plus  = 0.5 * (Rho_matrix(i+1, j) + Rho_matrix(i, j));
    rho_x_minus = 0.5 * (Rho_matrix(i, j)   + Rho_matrix(i-1, j));
    rho_z_plus  = 0.5 * (Rho_matrix(i, j+1) + Rho_matrix(i, j));
    rho_z_minus = 0.5 * (Rho_matrix(i, j)   + Rho_matrix(i, j-1));

    D_xx = (1/h^2) * ( (U(i+1, j, p) - U(i, j, p)) ./ rho_x_plus - ...
                       (U(i, j, p) - U(i-1, j, p)) ./ rho_x_minus );

    D_zz = (1/h^2) * ( (U(i, j+1, p) - U(i, j, p)) ./ rho_z_plus - ...
                       (U(i, j, p) - U(i, j-1, p)) ./ rho_z_minus );

    U(i, j, p+1) = 2*U(i, j, p) - U(i, j, p-1) + (dt^2 * K_matrix(i,j)) .* (D_xx + D_zz);

    %fonte
    f = exp(-((t - t0) / s)^2);
    U(x0, z0, p+1) = U(x0, z0, p+1) + f;

    %contorno
    U(:, 1, p+1) = 0;

    if M > 0
        U(1, :, p+1) = 0; U(Nx, :, p+1) = 0; U(:, Nz, p+1) = 0;

        for idxi = 1:M+1
            for idxj = 1:M+1
                if idxi==1 && idxj==1; continue; end

                tn = p - idxi + 2;

                %fundo
                coef_fundo = c3_inf(idxi, idxj);
                U(:, Nz, p+1) = U(:, Nz, p+1) - coef_fundo * U(:, Nz - idxj + 1, tn);

                %laterias
                %superior
                z_sup = 1:(interface_meio - 1);
                coef_lat_sup = c3_sup(idxi, idxj);
                U(1,  z_sup, p+1) = U(1,  z_sup, p+1) - coef_lat_sup * U(idxj, z_sup, tn);
                U(Nx, z_sup, p+1) = U(Nx, z_sup, p+1) - coef_lat_sup * U(Nx - idxj + 1, z_sup, tn);

                %interface
                z_int = interface_meio;
                coef_lat_int = c3_int(idxi, idxj);
                U(1,  z_int, p+1) = U(1,  z_int, p+1) - coef_lat_int * U(idxj, z_int, tn);
                U(Nx, z_int, p+1) = U(Nx, z_int, p+1) - coef_lat_int * U(Nx - idxj + 1, z_int, tn);

                %inferior
                z_inf = (interface_meio + 1):Nz;
                coef_lat_inf = c3_inf(idxi, idxj);
                U(1,  z_inf, p+1) = U(1,  z_inf, p+1) - coef_lat_inf * U(idxj, z_inf, tn);
                U(Nx, z_inf, p+1) = U(Nx, z_inf, p+1) - coef_lat_inf * U(Nx - idxj + 1, z_inf, tn);
            end
        end
    end

    %visualização
    if mod(n, 5) == 0
        imagesc(U(:,:,p+1)');
        caxis([-0.02 0.05]);
        colormap(jet);
        colorbar;

        title(sprintf('Tempo: %d | Ordem %d', n, M));
        xlabel('X'); ylabel('Z');
        drawnow;
    end
end
