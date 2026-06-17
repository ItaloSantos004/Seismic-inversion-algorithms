clear; clc; close all;

a = linspace(0, 90, 500);
ar = a * (pi / 180);

R = abs((cos(ar) - 1) ./ (cos(ar) + 1));

R1 = (R .^ 1) * 100;
R2 = (R .^ 2) * 100;
R3 = (R .^ 3) * 100;
R4 = (R .^ 4) * 100;

figure;
hold on;

plot(a, R1, 'LineWidth', 1.5);
plot(a, R2, 'LineWidth', 1.5);
plot(a, R3, 'LineWidth', 1.5);
plot(a, R4, 'LineWidth', 1.5);

grid on;

axis([0 90 0 100]);
xlabel('alpha', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Porcentagem', 'FontSize', 12, 'FontWeight', 'bold');
title('Teste', 'FontSize', 14);
