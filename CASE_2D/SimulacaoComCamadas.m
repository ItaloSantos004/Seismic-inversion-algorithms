clear; clc; close all;

%parametros
c   = [1500, 2200, 2500, 3000]; %velocidade
d = [1000, 1200, 1500, 1900]; %densidade
int = [60, 130, 210]; %indice da interface

%dominio
M = 4; %ordem
Nx = 150;
Nz = 250;
h = 8.5;
dt = 0.002;
tempo = 800;

Ncam = length(c); %numero de camadas

m = max(1, M);
U = zeros(Nx, Nz, tempo + m);
x = 2 : Nx - 1; %eixo x


%mapeando cada indice de z para sua camada
zcam = cell(Ncam, 1); %para armazenar

iz = 2;
for j = 1:Ncam-1 %atualizando os indices de cada camada
    zcam{j} = iz : int(j) - 1;
    iz = int(j) + 1;
end
zcam{Ncam} = iz : Nz - 1;

%constantes e contorno
gamma = (c .* dt ./ h).^2; %interior

%para velocidade efetiva
dint = zeros(Ncam-1, 1); %denominador
coefx = zeros(Ncam-1, 1); %coeficiente u_xx
cint = zeros(Ncam-1, 1); 

for j = 1:Ncam-1
    dint(j) = (h/2) * ( (1/(d(j)*c(j)^2)) + (1/(d(j+1)*c(j+1)^2)) );
    coefx(j) = (h/2) * ( (1/d(j)) + (1/d(j+1)) );
    cint(j) = sqrt(coefx(j) / dint(j)); %velocidade efetiva
end

%construção do operador da borda levando em conta outros angulos
angulos = linspace(0, 60, M);
%inicializa tudo antes da convolução
convcam = cell(Ncam, 1);
convint = cell(Ncam-1, 1);

for j = 1:Ncam
    convcam{j} = 1;
end
for j = 1:Ncam-1
    convint{j} = 1;
end

%convolução
if M > 0
    for i = 1:M
        cost = cosd(angulos(i));

        %camadas
        for j = 1:Ncam
            S = (c(j) / cost) * dt / h;
            S2 = [1, -(S - 1)/(S + 1); (S - 1)/(S + 1), -1];
            convcam{j} = conv2(convcam{j}, S2);
        end

        %interfaces
        for j = 1:Ncam-1
            S = (cint(j) / cost) * dt / h;
            S2 = [1, -(S - 1)/(S + 1); (S - 1)/(S + 1), -1];
            convint{j} = conv2(convint{j}, S2);
        end
    end
end

%fonte
t0 = 0.1;
s = 0.02;
x0 = round(Nx/2);
z0 = 2;

figure;
arq = 'simulacao2D.gif';

for n = 1:tempo %preenchendo toda a matriz
    t = n * dt;
    p = n + m;

    %interior das camadas
    for j = 1:Ncam
        z = zcam{j};

        lap = U(x+1, z, p) - 2*U(x, z, p) + U(x-1, z, p) + U(x, z+1, p) - 2*U(x, z, p) + U(x, z-1, p);

        U(x, z, p+1) = 2*U(x, z, p) - U(x, z, p-1) + gamma(j) .* lap;
    end

    %interfaces
    for j = 1:Ncam-1
        Zcam = int(j);
        uint = U(x, Zcam, p);

        numz = ( U(x, Zcam+1, p) - uint ) ./ (d(j+1)*h) - ( uint - U(x, Zcam-1, p) ) ./ (d(j)*h);

        u_xx = ( U(x+1, Zcam, p) - 2*uint + U(x-1, Zcam, p) ) ./ (h^2);
        numx = coefx(j) .* u_xx;

        u_tt = (numz + numx) ./ dint(j);
        U(x, Zcam, p+1) = 2*uint - U(x, Zcam, p-1) + (dt^2 .* u_tt);
    end

    %fonte
    f = exp(-((t - t0) / s)^2);
    U(x0, z0, p+1) = U(x0, z0, p+1) + f;

    U(:, 1, p+1) = 0; %dirichlet

    %bordas
    if M == 0
        U(1, :, p+1) = 0; U(Nx, :, p+1) = 0; U(:, Nz, p+1) = 0;
    else
        U(1, :, p+1) = 0; U(Nx, :, p+1) = 0; U(:, Nz, p+1) = 0;

        for ii = 1:M+1
            for jj = 1:M+1
                if ii==1 && jj==1; continue; end
                tn = p - ii + 2;

                %fundo
                coeff = convcam{Ncam}(ii, jj); %coeficiente do fundo
                U(:, Nz, p+1) = U(:, Nz, p+1) - coeff * U(:, Nz - jj + 1, tn);

                %laterias camadas
                for j = 1:Ncam
                    z = zcam{j};
                    coefl = convcam{j}(ii, jj); %coeficiente lateral
                    U(1,  z, p+1) = U(1,  z, p+1) - coefl * U(jj, z, tn);
                    U(Nx, z, p+1) = U(Nx, z, p+1) - coefl * U(Nx - jj + 1, z, tn);
                end

                %laterias interface
                for j = 1:Ncam-1
                    Zcam = int(j);
                    coefl = convint{j}(ii, jj);
                    U(1,  Zcam, p+1) = U(1,  Zcam, p+1) - coefl * U(jj, Zcam, tn);
                    U(Nx, Zcam, p+1) = U(Nx, Zcam, p+1) - coefl * U(Nx - jj + 1, Zcam, tn);
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
        for j = 1:length(int) %indentificando as interfaces
            plot([1 Nx], [int(j) int(j)], 'w--', 'LineWidth', 1);
        end
        hold off;

        drawnow;

        %gerando o .gif
        frame = getframe(gcf);
        im = frame2im(frame);
        [imind, cm] = rgb2ind(im); 

        if n == 5
            imwrite(imind, cm, arq, 'gif', 'Loopcount', inf, 'DelayTime', 0.15);
        else 
            imwrite(imind, cm, arq, 'gif', 'WriteMode', 'append', 'DelayTime', 0.15);
        end
    end
end
