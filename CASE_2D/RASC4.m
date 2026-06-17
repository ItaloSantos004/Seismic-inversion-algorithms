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

%camdas
c1 = 1500; p1 = 1000;
c2 = 2500; p2 = 1800;

int = round(Nz / 2);

%dividindo os indices
z_sup_reg = 2 : int - 1; %superior
z_int     = int; %interface
z_inf_reg = int + 1 : Nz - 1; %inferior
i_reg     = 2 : Nx - 1; %eixo x


%constantes
gamma1 = (c1 * dt / h)^2;
gamma2 = (c2 * dt / h)^2;

den = (h/2) * ( (1/(p1*c1^2)) + (1/(p2*c2^2)) );
coef_x = (h/2) * ( (1/p1) + (1/p2) );

%para convolução
%superior
S_sup = c1 * dt / h;
c2_sup = [1, -(S_sup - 1)/(S_sup + 1); (S_sup - 1)/(S_sup + 1), -1];
c3_sup = 1;

%inferior
S_inf = c2 * dt / h;
c2_inf = [1, -(S_inf - 1)/(S_inf + 1); (S_inf - 1)/(S_inf + 1), -1];
c3_inf = 1;

if M > 0
    for idx = 1:M
        c3_sup = conv2(c3_sup, c2_sup);
        c3_inf = conv2(c3_inf, c2_inf);
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

    %superior
    Laplaciano_sup = U(i_reg+1, z_sup_reg, p) - 2*U(i_reg, z_sup_reg, p) + U(i_reg-1, z_sup_reg, p) + ...
                     U(i_reg, z_sup_reg+1, p) - 2*U(i_reg, z_sup_reg, p) + U(i_reg, z_sup_reg-1, p);

    U(i_reg, z_sup_reg, p+1) = 2*U(i_reg, z_sup_reg, p) - U(i_reg, z_sup_reg, p-1) + gamma1 .* Laplaciano_sup;

    %inferior
    Laplaciano_inf = U(i_reg+1, z_inf_reg, p) - 2*U(i_reg, z_inf_reg, p) + U(i_reg-1, z_inf_reg, p) + ...
                     U(i_reg, z_inf_reg+1, p) - 2*U(i_reg, z_inf_reg, p) + U(i_reg, z_inf_reg-1, p);

    U(i_reg, z_inf_reg, p+1) = 2*U(i_reg, z_inf_reg, p) - U(i_reg, z_inf_reg, p-1) + gamma2 .* Laplaciano_inf;

    %interface
    U_int = U(i_reg, z_int, p);

    num_z = ( U(i_reg, z_int+1, p) - U_int ) ./ (p2*h) - ...
            ( U_int - U(i_reg, z_int-1, p) ) ./ (p1*h);

    U_xx = ( U(i_reg+1, z_int, p) - 2*U_int + U(i_reg-1, z_int, p) ) ./ (h^2);
    num_x = coef_x .* U_xx;

    u_tt_int = (num_z + num_x) ./ den;

    U(i_reg, z_int, p+1) = 2*U_int - U(i_reg, z_int, p-1) + (dt^2 .* u_tt_int);

    %fonte
    f_val = exp(-((t - t0) / s)^2);
    U(x0, z0, p+1) = U(x0, z0, p+1) + f_val;

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

                %superior
                coef_lat_sup = c3_sup(idxi, idxj);
                U(1,  z_sup_reg, p+1) = U(1,  z_sup_reg, p+1) - coef_lat_sup * U(idxj, z_sup_reg, tn);
                U(Nx, z_sup_reg, p+1) = U(Nx, z_sup_reg, p+1) - coef_lat_sup * U(Nx - idxj + 1, z_sup_reg, tn);

                %inferior
                z_lower_lat = int : Nz - 1;
                coef_lat_inf = c3_inf(idxi, idxj);
                U(1,  z_lower_lat, p+1) = U(1,  z_lower_lat, p+1) - coef_lat_inf * U(idxj, z_lower_lat, tn);
                U(Nx, z_lower_lat, p+1) = U(Nx, z_lower_lat, p+1) - coef_lat_inf * U(Nx - idxj + 1, z_lower_lat, tn);
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
