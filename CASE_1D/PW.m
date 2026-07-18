clear; clc; close all;

nt = 1600; %tempo
ni = 400;  %profundidade

%propriedades do meio
C = ones(1, ni);
Rho = ones(1, ni);

C(1:120)   = 1500; Rho(1:120)   = 1000;
C(121:250) = 2500; Rho(121:250) = 1800;
C(251:end) = 4000; Rho(251:end) = 2500;

%Impedância
Z = C .* Rho; 

%Visualização
figure;

%perfil de velocidade
subplot(1, 3, 1);
plot(1:ni, C, 'b', 'LineWidth', 2);
title('Velocidade (c)');
xlabel('Profundidade (Xi)');
ylabel('Velocidade (m/s)');
ylim([min(C)-500, max(C)+500]);
grid on;

%perfil de densidade
subplot(1, 3, 2);
plot(1:ni, Rho, 'r', 'LineWidth', 2);
title('Densidade (\rho)');
xlabel('Profundidade (Xi)');
ylabel('Densidade (kg/m^3)');
ylim([min(Rho)-500, max(Rho)+500]);
grid on;

%perfil de impedancia
subplot(1, 3, 3);
plot(1:ni, Z, 'k', 'LineWidth', 2);
title('Impedância Acústica (Z)');
xlabel('Profundidade (Xi)');
ylabel('Impedância (kg/(m^2 \cdot s))');
ylim([min(Z)-1e6, max(Z)+1e6]);
grid on;

P = zeros(ni, nt);
W = zeros(ni, nt);

%fonte
t0 = 100;
s = 15;
fonte = 1000 * exp(-((1:nt) - t0).^2 / (2*s^2));


for j = 2:nt

    if mod(1 + j, 2) == 0 %superficie
        P(1, j) = fonte(j);
        U = Z(1)*W(2, j-1) - P(2, j-1);
        W(1, j) = (P(1, j) + U) / Z(1);
    end

    for i = 2:ni-1 %pontos dentro
        if mod(i + j, 2) == 0
            D = Z(i-1)*W(i-1, j-1) + P(i-1, j-1);
            U = Z(i)*W(i+1, j-1) - P(i+1, j-1);

            W(i, j) = (D + U) / (Z(i-1) + Z(i));
            P(i, j) = (Z(i)*D - Z(i-1)*U) / (Z(i-1) + Z(i));
        end
    end

    if mod(ni + j, 2) == 0 %fundo
        D = Z(ni-1)*W(ni-1, j-1) + P(ni-1, j-1);
        W(ni, j) = D / (2 * Z(ni-1));
        P(ni, j) = D / 2;
    end
end

%linha da interface
intf = find(diff(Z) ~= 0);

%visualização de P e W
figure;

for j = 1:5:nt

    %pressao
    subplot(2, 1, 1);
    plot(1:ni, P(:, j), 'b', 'LineWidth', 1.5);
    hold on;
    for k = 1:length(intf)
        line([intf(k) intf(k)], [-1200 1200], 'Color', 'red', 'LineStyle', '--');
    end
    hold off;
    axis([1 ni -1200 1200]);
    title(['Propagação j = ', num2str(j)]);
    ylabel('P');
    grid on;

    %velocidade
    subplot(2, 1, 2);
    plot(1:ni, W(:, j), 'g', 'LineWidth', 1.5);
    hold on;
    for k = 1:length(intf)
        line([intf(k) intf(k)], [-1.5e-3 1.5e-3], 'Color', 'red', 'LineStyle', '--');
    end
    hold off;
    axis([1 ni -1.5e-3 1.5e-3]);
    ylabel('W');
    xlabel('xi');
    grid on;

    drawnow;
end