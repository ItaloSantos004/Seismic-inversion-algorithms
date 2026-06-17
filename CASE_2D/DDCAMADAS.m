clear; clc; close all;

%parametros
C   = [1500, 2200, 2500, 3000];
RHO = [1000, 1200, 1500, 1900];
interfaces = [60, 130, 210];

%dominio
M = 4; %ordem
Nx = 150;
Nz = 250;
h = 8.5;
dt = 0.002;
tempo = 800;

N_camadas = length(C);

m = max(1, M);
U = zeros(Nx, Nz, tempo + m);
i_reg = 2 : Nx - 1; %eixo x


%mapeando cada indice de z para sua camada
z_camadas = cell(N_camadas, 1); %para armazenar

start_z = 2;
for j = 1:N_camadas-1 %atualizando os indices de cada camada
    z_camadas{j} = start_z : interfaces(j) - 1;
    start_z = interfaces(j) + 1;
end
z_camadas{N_camadas} = start_z : Nz - 1;

%constantes e contorno
gamma = (C .* dt ./ h).^2; %interior

den_int = zeros(N_camadas-1, 1); %para velocidade interface
coef_x_int = zeros(N_camadas-1, 1);
C_int = zeros(N_camadas-1, 1);

for j = 1:N_camadas-1
    den_int(j) = (h/2) * ( (1/(RHO(j)*C(j)^2)) + (1/(RHO(j+1)*C(j+1)^2)) );
    coef_x_int(j) = (h/2) * ( (1/RHO(j)) + (1/RHO(j+1)) );
    C_int(j) = sqrt(coef_x_int(j) / den_int(j)); %velocidade efetiva
end

%construção do operador da borda levando em conta outros angulos
angulos = linspace(0, 60, M);
%inicializa tudo antes da convolução
c3_camadas = cell(N_camadas, 1);
c3_interfaces = cell(N_camadas-1, 1);

for j = 1:N_camadas
    c3_camadas{j} = 1;
end
for j = 1:N_camadas-1
    c3_interfaces{j} = 1;
end

%convolução
if M > 0
    for idx = 1:M
        cos_theta = cosd(angulos(idx));

        %camadas
        for j = 1:N_camadas
            S = (C(j) / cos_theta) * dt / h;
            c2 = [1, -(S - 1)/(S + 1); (S - 1)/(S + 1), -1];
            c3_camadas{j} = conv2(c3_camadas{j}, c2);
        end

        %interfaces
        for j = 1:N_camadas-1
            S = (C_int(j) / cos_theta) * dt / h;
            c2 = [1, -(S - 1)/(S + 1); (S - 1)/(S + 1), -1];
            c3_interfaces{j} = conv2(c3_interfaces{j}, c2);
        end
    end
end

%fonte
t0 = 0.1;
s = 0.02;
x0 = round(Nx/2);
z0 = 2;

figure;
nome_gif = 'simulacao2D.gif';

for n = 1:tempo %preenchendo toda a matriz
    t = n * dt;
    p = n + m;

    %interior das camadas
    for j = 1:N_camadas
        z_reg = z_camadas{j};

        Laplaciano = U(i_reg+1, z_reg, p) - 2*U(i_reg, z_reg, p) + U(i_reg-1, z_reg, p) + ...
                     U(i_reg, z_reg+1, p) - 2*U(i_reg, z_reg, p) + U(i_reg, z_reg-1, p);

        U(i_reg, z_reg, p+1) = 2*U(i_reg, z_reg, p) - U(i_reg, z_reg, p-1) + gamma(j) .* Laplaciano;
    end

    %interfaces
    for j = 1:N_camadas-1
        zi = interfaces(j);
        U_int = U(i_reg, zi, p);

        num_z = ( U(i_reg, zi+1, p) - U_int ) ./ (RHO(j+1)*h) - ...
                ( U_int - U(i_reg, zi-1, p) ) ./ (RHO(j)*h);

        U_xx = ( U(i_reg+1, zi, p) - 2*U_int + U(i_reg-1, zi, p) ) ./ (h^2);
        num_x = coef_x_int(j) .* U_xx;

        u_tt_int = (num_z + num_x) ./ den_int(j);
        U(i_reg, zi, p+1) = 2*U_int - U(i_reg, zi, p-1) + (dt^2 .* u_tt_int);
    end

    %fonte
    f_val = exp(-((t - t0) / s)^2);
    U(x0, z0, p+1) = U(x0, z0, p+1) + f_val;

    U(:, 1, p+1) = 0; %dirichlet

    %bordas
    if M == 0
        U(1, :, p+1) = 0; U(Nx, :, p+1) = 0; U(:, Nz, p+1) = 0;
    else
        U(1, :, p+1) = 0; U(Nx, :, p+1) = 0; U(:, Nz, p+1) = 0;

        for idxi = 1:M+1
            for idxj = 1:M+1
                if idxi==1 && idxj==1; continue; end
                tn = p - idxi + 2;

                %fundo
                coef_fundo = c3_camadas{N_camadas}(idxi, idxj);
                U(:, Nz, p+1) = U(:, Nz, p+1) - coef_fundo * U(:, Nz - idxj + 1, tn);

                %laterias camadas
                for j = 1:N_camadas
                    z_reg = z_camadas{j};
                    coef_lat = c3_camadas{j}(idxi, idxj);
                    U(1,  z_reg, p+1) = U(1,  z_reg, p+1) - coef_lat * U(idxj, z_reg, tn);
                    U(Nx, z_reg, p+1) = U(Nx, z_reg, p+1) - coef_lat * U(Nx - idxj + 1, z_reg, tn);
                end

                %laterias interface
                for j = 1:N_camadas-1
                    zi = interfaces(j);
                    coef_lat = c3_interfaces{j}(idxi, idxj);
                    U(1,  zi, p+1) = U(1,  zi, p+1) - coef_lat * U(idxj, zi, tn);
                    U(Nx, zi, p+1) = U(Nx, zi, p+1) - coef_lat * U(Nx - idxj + 1, zi, tn);
                end
            end
        end
    end

    %visualização
    if mod(n, 5) == 0
        imagesc(U(:,:,p+1)');
        caxis([-0.02 0.05]);
        colormap(jet);
        colorbar;
        title(sprintf('Tempo: %d', n));
        xlabel('X'); ylabel('Z');

        hold on;
        for j = 1:length(interfaces) %indentificando as interfaces
            plot([1 Nx], [interfaces(j) interfaces(j)], 'w--', 'LineWidth', 1);
        end
        hold off;

        drawnow;

        frame = getframe(gcf);
        im = frame2im(frame);
        [imind, cm] = rgb2ind(im); % Converte para o formato de cores do GIF

        % 2. Grava no arquivo
        if n == 5 % Se for o primeiro frame (já que mod(n,5) começa no 5)
            imwrite(imind, cm, nome_gif, 'gif', 'Loopcount', inf, 'DelayTime', 0.15);
        else % Se for os frames seguintes, anexa (append) ao arquivo existente
            imwrite(imind, cm, nome_gif, 'gif', 'WriteMode', 'append', 'DelayTime', 0.15);
        end
    end
end
