clear; clc; close all;

M = 4;

%dominio
Nx = 200;
Nz = 200;

h = 8.5;
c = 1500.0;
dt = 0.004;
S = c * dt / h;
tempo = 800;

m = max(1, M);
U = zeros(Nx, Nz, tempo + m);

%fonte
t0 = 0.1;
s0 = 0.02;
x0 = round(Nx/2);
z0 = 2;

%ponto para comparação
xr = 100;
zr = 100;

%distancia ate a fonte
dist1 = sqrt(((xr - x0) * h)^2 + ((zr - z0) * h)^2);

%fonte fantasma
zf = 0;
dist2 = sqrt(((xr - x0) * h)^2 + ((zr - zf) * h)^2); %distancia

orig = zeros(1, tempo);
conv  = zeros(1, tempo);

%matrizes
c1 = (S - 1) / (S + 1);
c2 = [1, -c1; c1, -1];
c3 = 1;

if M > 0
    for i = 1:M
        c3 = conv2(c3, c2);
    end
end

figure;

%preenchendo
for n = 1:tempo
    t = n * dt;
    p = n + m;

    %interior
    U(2:Nx-1, 2:Nz-1, p+1) = 2*U(2:Nx-1, 2:Nz-1, p) - U(2:Nx-1, 2:Nz-1, p-1) + ...
        (S^2) * (U(3:Nx, 2:Nz-1, p) - 2*U(2:Nx-1, 2:Nz-1, p) + U(1:Nx-2, 2:Nz-1, p) + ...
                 U(2:Nx-1, 3:Nz, p) - 2*U(2:Nx-1, 2:Nz-1, p) + U(2:Nx-1, 1:Nz-2, p));

    %fonte
    f = exp(-((t - t0) / s0)^2);
    U(x0, z0, p+1) = U(x0, z0, p+1) + f;

    U(:, 1, p+1) = 0; %dirichlet

    %paredes
    if M == 0
        U(1, :, p+1) = 0; U(Nx, :, p+1) = 0; U(:, Nz, p+1) = 0;
    else
        U(1, :, p+1) = 0; U(Nx, :, p+1) = 0; U(:, Nz, p+1) = 0;

        for i = 1:M+1
            for j = 1:M+1
                if i==1 && j==1; continue; end
                coef = c3(i,j);
                tn = p - i + 2;

                U(1, :, p+1)  = U(1, :, p+1)  - coef * U(j, :, tn);
                U(Nx, :, p+1) = U(Nx, :, p+1) - coef * U(Nx - j + 1, :, tn);
                U(:, Nz, p+1) = U(:, Nz, p+1) - coef * U(:, Nz - j + 1, tn);
            end
        end
    end

    %linhas para comparação
    orig(n) = U(xr, zr, p+1);

    if t > dist1/c %integral onda direta
        dt1 = linspace(0, t - dist1/c - 1e-6, 400);
        fdt = exp(-((dt1 - t0) / s0).^2);
        int1 = fdt ./ sqrt(c^2 * (t - dt1).^2 - dist1^2);
        a1 = (1 / (2*pi*c)) * trapz(dt1, int1);
    else
        a1 = 0;
    end

    if t > dist2/c %integral onda refletida
        dt2 = linspace(0, t - dist2/c - 1e-6, 400);
        fdt1 = exp(-((dt2 - t0) / s0).^2);
        int2 = fdt1 ./ sqrt(c^2 * (t - dt2).^2 - dist2^2);
        a2 = (1 / (2*pi*c)) * trapz(dt2, int2);
    else
        a2 = 0;
    end

    conv(n) = a1 - a2;
end

%normalizando
o1 = orig / max(abs(orig));
o2  = conv / max(abs(conv));

tempo1 = (1:tempo) * dt;

plot(tempo1, o1, 'b-', 'LineWidth', 2);
hold on;
plot(tempo1, o2, 'r--', 'LineWidth', 2);
grid on;
title('Comparação');
xlabel('Tempo');
ylabel('Amplitude');
