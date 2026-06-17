clear; clc; close all;

M = 3; %ordem

Nx = 150;
Nz = 200;

h = 10.0;
c = 1500.0;
dt = 0.004;
S = c * dt / h;
PassosTempo = 600;

% 2. O Tensor de Histórico (Máquina do Tempo)
% =========================================================================
% O FDTD precisa de pelo menos 3 níveis para o interior (Futuro, Presente, Passado).
% Se a ordem de absorção M for maior que 2, precisamos de M+1 níveis.
NiveisTempo = max(3, M + 1);
U = zeros(Nx, Nz, NiveisTempo);

% Fonte Gaussiana
t0 = 0.1; sigma = 0.02;
x_src = round(Nx/2); z_src = 2;

% 3. O Motor Gerador de Coeficientes de Higdon
% =========================================================================
c1 = (S - 1) / (S + 1);

% Matriz Base de 1ª Ordem
% Linhas: Tempo (1=Futuro, 2=Presente) | Colunas: Espaço (1=Borda, 2=Interior)
C_base = [1, -c1; c1, -1];

% Matriz Nula caso Ordem seja 0 (Dirichlet)
C_higdon = 1;

if M > 0
    % Eleva o polinômio à potência M através de convoluções 2D
    for i = 1:M
        C_higdon = conv2(C_higdon, C_base);
    end
end

figure('Position', [100, 100, 800, 600]);

% =========================================================================
% LAÇO PRINCIPAL
% =========================================================================
for n = 1:PassosTempo
    t = n * dt;

    % A. ATUALIZAÇÃO DO INTERIOR (Onda 2D)
    % U(:,:,1) = Futuro | U(:,:,2) = Presente | U(:,:,3) = Passado
    U(2:Nx-1, 2:Nz-1, 1) = 2*U(2:Nx-1, 2:Nz-1, 2) - U(2:Nx-1, 2:Nz-1, 3) + ...
        (S^2) * (U(3:Nx, 2:Nz-1, 2) - 2*U(2:Nx-1, 2:Nz-1, 2) + U(1:Nx-2, 2:Nz-1, 2) + ...
                 U(2:Nx-1, 3:Nz, 2) - 2*U(2:Nx-1, 2:Nz-1, 2) + U(2:Nx-1, 1:Nz-2, 2));

    % B. INJEÇÃO DA EXPLOSÃO
    src = exp(-((t - t0) / sigma)^2);
    U(x_src, z_src, 1) = U(x_src, z_src, 1) + src;

    % Superfície Rígida
    U(:, 1, 1) = U(:, 2, 2);

    % C. CONDIÇÕES DE TRANSPARÊNCIA GENÉRICAS
    if M == 0
        U(1, :, 1) = 0; U(Nx, :, 1) = 0; U(:, Nz, 1) = 0;
    else
        % Limpa as bordas antes de aplicar o somatório
        U(1, :, 1) = 0; U(Nx, :, 1) = 0; U(:, Nz, 1) = 0;

        % Laço Dinâmico: Aplica os pesos da Matriz de Convolução
        for i = 1:M+1
            for j = 1:M+1
                % Pula o coeficiente (1,1) pois é o próprio futuro na borda
                if i==1 && j==1; continue; end

                peso = C_higdon(i,j);

                % ESQUERDA: O interior cresce positivamente (j)
                U(1, :, 1) = U(1, :, 1) - peso * U(j, :, i);

                % DIREITA: O interior cresce negativamente (Nx - j + 1)
                U(Nx, :, 1) = U(Nx, :, 1) - peso * U(Nx - j + 1, :, i);

                % FUNDO: O interior cresce negativamente no eixo Z (Nz - j + 1)
                U(:, Nz, 1) = U(:, Nz, 1) - peso * U(:, Nz - j + 1, i);
            end
        end
    end



    % E. ATUALIZAÇÃO DA MÁQUINA DO TEMPO (Shift do Tensor)
    % Empurra tudo para o passado: o Presente vira Passado, o Futuro vira Presente.
    for k = NiveisTempo:-1:2
        U(:,:,k) = U(:,:,k-1);
    end

    % F. PLOTAGEM ANIMADA
    if mod(n, 4) == 0
        imagesc(U(:,:,2)');
        caxis([-0.02 0.05]);
        colormap(jet); colorbar;
        title(sprintf('Passo: %d | Ordem %d', n, M));
        xlabel('Distância X'); ylabel('Profundidade Z');
        drawnow;
    end
end
