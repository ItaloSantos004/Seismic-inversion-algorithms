clear; clc; close all;

%Espaço que estamos trabalhando
z = -50:1:500;
nz = length(z);
z0 = 0; %Inicio da fonte
zc = 200; %onde a camada muda

%Velocidade nas camadas
c = zeros(size(z));
c1 = 1500; %meio 1
c2 = 3000; %meio 2
c(z < zc) = c1;
c(z >= zc) = c2;

%Tempo(CFL)
dz = z(2) - z(1);
dt = dz / (10 * max(c));
t = 0:dt:0.4;
nt = length(t);

U = zeros(nt, nz);

%Condições iniciais
U(1, :) = exp( -(z-z0).^2 / 50);
U(2, :) = exp( -(z-z0).^2 / 50);

%Diferenças finitas e utilizando deslocamento de grade
gamma = (c * dt / dz).^2;
alpha = (c * dt / dz); %paraxial

coef = (alpha - 1) ./ (alpha + 1); %coeficiente de ajuste

for n = 2:nt-1
    for i = 2:nz-1
        U(n+1, i) = 2*U(n, i) - U(n-1, i) + gamma(i)*(U(n, i+1) - 2*U(n, i) + U(n, i-1));
    end

    %Bordas com deslocamento de grade
    U(n+1, 1) = U(n, 2) + coef(1) * (U(n+1, 2) - U(n, 1));

    U(n+1, nz) = U(n, nz-1) + coef(nz) * (U(n+1, nz-1) - U(n, nz));
end

%Salvando tudo
%save('matriz_onda_camadas.txt', 'U');
%fprintf('\nMatriz completa salva no arquivo: matriz_onda_camadas.txt\n');

%Simulação
figure;
for n = 1:50:nt
    plot(z, U(n, :), 'b', 'linewidth', 1.5);
    hold on;
    line([zc zc], [-1 1.5], 'Color', 'k', 'LineStyle', '--');
    hold off;
    axis([min(z) max(z) -0.5 1.5]);
    title(['Propagação da Onda (Com camadas) - Tempo: ', num2str(t(n), '%.3f'), 's']);
    xlabel('Posição (z)');
    ylabel('Amplitude');
    grid on;
    drawnow;
end
