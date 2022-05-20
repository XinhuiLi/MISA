% Case13:
% 1 Dataset, Many ISA-type SCVs, No corr, var = (d+1)
% 28 Components, changing subspace size
% - 7 subspaces, each of size 1, 2, 3, 4, 5, 6, 7, respectively.
% - 7 subspaces, each of size 4.
%% Initialize
% rng(1981)
tag = 'ISA';
cas = num2str(13);
myfolder = '\\loki\export\mialab\users\rsilva';

%% Generate the SCVs
M = 1;
C = 28;
N = 32968; % Number of samples
r = 0;

%% SCVs: MvLap, zero-mean, uncorr
% Desired parameters:
% D = [1, 2, 3, 4, 5, 6, 7];
D = 4*ones(1,7);
out = cell(1,length(D));
for ss = 1:length(D) % SCV dimensionality
    d = D(ss);
    mu = zeros(1,d);    % SCV mean
    if d == 1, R = 1;
    else
        R = toeplitz(r.^linspace(0,1,d));   % SCV correlation
    end
    S = R;
    lmb = .5;
    
    out{ss} = mymvlap(mu,S,N,1/lmb);
end

%% Assign SCVs to datasets
Sgt = cell(1,max(M));
Sgt{M(1)} = [out{:}]';

%% Define subspace composition
S = cell(1,max(M));
tp = mat2cell(ones(1,C(M)), 1, D);
S{M(1)} = blkdiag(tp{:});

%% Save SCVs in .mat file
% Go to data folder
curfolder = pwd; % save current folder
cd([myfolder '\projects\MultivariateICA\MISA\fresh\data\Sgt\case' cas])

% Find out this rep number
t = 1;
while exist(['SCV_' tag '_case' cas '_r' num2str(t,'%03d') '.mat'],'file')
    t = t + 1;
end
save(['SCV_' tag '_case' cas '_r' num2str(t,'%03d') '.mat'],'M', 'S', 'Sgt')

% Return to initial folder
cd(curfolder)