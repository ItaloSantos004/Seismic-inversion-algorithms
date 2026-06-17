clear; clc; close all;

nt = 1600; %tempo
ni = 400;  %profundidade

%impedancia
Z = ones(1, ni);
Z(1:120)   = 1500 * 1000;
Z(121:250) = 2500 * 1800;
Z(251:end) = 4000 * 2500;

P = zeros(ni, nt);
W = zeros(ni, nt);

%fonte
fonte = zeros(1, nt);

fonte(3) = 1000;

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

%visualização
intf = find(diff(Z) ~= 0); %indices da interface

figure;

for j = 1:5:nt

    %pressão
    subplot(2, 1, 1);
    plot(1:ni, P(:, j), 'b', 'LineWidth', 1);
    hold on;
    for k = 1:length(intf)
        line([intf(k) intf(k)], [-1200 1200], 'Color', 'r', 'LineStyle', '--');
    end
    hold off;
    axis([1 ni -1200 1200]);
    title(['Propagação j =', num2str(j)]);
    ylabel('P');
    grid on;

    %velocidade
    subplot(2, 1, 2);
    plot(1:ni, W(:, j), 'g', 'LineWidth', 1);
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
