clear; clc; close all;

M = 4; %ordem

Nx = 150; %dominio
Nz = 200;

h = 8.5;
c = 1500.0;
dt = 0.004;
S = c * dt / h;
tempo = 350;


m = max(1, M);
U = zeros(Nx, Nz, tempo + m);

%parametros da fonte
t0 = 0.1;
s = 0.02;
x0 = round(Nx/2);
z0 = 50;

%criando o termo elevado a M
c1 = (S - 1) / (S + 1);
c2 = [1, -c1; c1, -1];
c3 = 1;

if M > 0
    for i = 1:M
        c3 = conv2(c3, c2);
    end
end

figure('Position', [100, 100, 800, 600]);


for n = 1:tempo
    t = n * dt;

    p = n + m; %para ficar certo os indices na matriz

    %interior
    U(2:Nx-1, 2:Nz-1, p+1) = 2*U(2:Nx-1, 2:Nz-1, p) - U(2:Nx-1, 2:Nz-1, p-1) + ...
        (S^2) * (U(3:Nx, 2:Nz-1, p) - 2*U(2:Nx-1, 2:Nz-1, p) + U(1:Nx-2, 2:Nz-1, p) + ...
                 U(2:Nx-1, 3:Nz, p) - 2*U(2:Nx-1, 2:Nz-1, p) + U(2:Nx-1, 1:Nz-2, p));

    %fonte
    f = exp(-((t - t0) / s)^2);
    U(x0, z0, p+1) = U(x0, z0, p+1) + f;

    %condição para ter a superficie rigida
    U(:, 1, p+1) = 0;

    %agora é a condição de transparencia
    if M == 0
        U(1, :, p+1) = 0; U(Nx, :, p+1) = 0; U(:, Nz, p+1) = 0;
    else
        U(1, :, p+1) = 0; U(Nx, :, p+1) = 0; U(:, Nz, p+1) = 0;

        for i = 1:M+1
            for j = 1:M+1
                if i==1 && j==1; continue; end
                c = c3(i,j);
                tn = p - i + 2;

                U(1, :, p+1) = U(1, :, p+1) - c * U(j, :, tn); %esquerda

                U(Nx, :, p+1) = U(Nx, :, p+1) - c * U(Nx - j + 1, :, tn); %direita

                U(:, Nz, p+1) = U(:, Nz, p+1) - c * U(:, Nz - j + 1, tn); %fundo
            end
        end
    end


    if mod(n, 5) == 0
        imagesc(U(:,:,p)'); %plota no instante p, pois é onde começa
        caxis([-0.02 0.05]);
        colormap(jet);
        colorbar;
        title(sprintf('tempo: %d | Ordem %d', n, M));
        xlabel('X'); ylabel('Z');
        drawnow;
    end
end
