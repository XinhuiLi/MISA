% myfolder = '\\loki\export\mialab\users\rsilva';
% %myfolder = 'Z:';
% cd([myfolder '\projects\MultivariateICA\MISA'])

% RUN THIS FILE FROM THE FOLDER WHERE IT IS STORED %

% Generation of subspaces:
N = 1000;   % Number of observations

% Subspace 1:
% 2D uncorr. MvLap
d = 2;
mu = zeros(1,d);
R = eye(d);
lmb = .5;
[out{1}, R_{1}, Scov{1}] = mymvlap(mu,R,N,1/lmb);

% Subspace 2:
% 3D uncorr. MvLap
d = 3;
mu = zeros(1,d);
R = eye(d);
lmb = .5;
[out{2}, R_{2}, Scov{2}] = mymvlap(mu,R,N,1/lmb);

% Component 3:
% 3D uncorr. MvLap
d = 3;
mu = zeros(1,d);
R = eye(d);
lmb = .5;
[out{3}, R_{3}, Scov{3}] = mymvlap(mu,R,N,1/lmb);

% Component 4:
% 2D uncorr. MvLap
d = 2;
mu = zeros(1,d);
R = eye(d);
lmb = .5;
[out{4}, R_{4}, Scov{4}] = mymvlap(mu,R,N,1/lmb);

%cd([myfolder '\projects\MultivariateICA\MISA'])

M = 2;
Sgt = cell(1,M);
Sgt{1} = [out{1}(:,1) out{2}(:,1) out{3}(:,1:2)];
Sgt{2} = [out{1}(:,2) out{2}(:,2:3) out{3}(:,3) out{4}];
for ff = 1:M
    Sgt{ff} = Sgt{ff}';
end

save('jointsourcesMISA_case6.mat','Sgt','R_','Scov')