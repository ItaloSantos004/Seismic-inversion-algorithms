clear; clc; close all;

%Espaço que estamos trabalhando
z = -50:1:500;
nz = length(z);
z0 = 0; %Inicio da fonte

c = 1500 * ones(size(z)); %Velocidade da onda

%Tempo
dz = z(2) - z(1);
dt = dz / (10 * max(c));
t = 0:dt:0.5;
nt = length(t);

%Inicializando a matriz tempo x espaço
U = zeros(nt, nz);

%Condições iniciais
U0 = exp( -(z-z0).^2 / 50);
Um1 = exp( -(z-z0).^2 / 50);

%Preenche os espaços da matrix ja dados
U(1, :) = Um1;
U(2, :) = U0;

%Diferenças Finitas, usando stencil em cruz
gamma = (c * dt / dz).^2;

for n = 2:nt-1
    for i = 2:nz-1
        U(n+1, i) = 2*U(n, i) - U(n-1, i) + gamma(i)*(U(n, i+1) - 2*U(n, i) + U(n, i-1));
    end

    %Condições de Dirichlet nas bordas
    U(n+1, 1) = 0;
    U(n+1, nz) = 0;
end

%Simulação
figure;
for n = 1:50:nt
    plot(z, U(n, :), 'linewidth', 1.5);
    axis([min(z) max(z) -0.6 1.2]);
    title(['Propagação da Onda (Condições de Dirichlet) - Tempo: ', num2str(t(n), '%.3f'), 's']);
    xlabel('Posição (z)');
    ylabel('Amplitude');
    grid on;
    drawnow;
end
