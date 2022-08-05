close all; clear; clc;
% @gsd: generate simulated data
% @gsm: generate simulated mixing matrix

debug = false;

addpath("/Users/xli77/Documents/MISA/scripts");
addpath("/Users/xli77/Documents/MISA/scripts/toy_example/");
addpath(genpath('/Users/xli77/Documents/gift/GroupICATv4.0c'));

% generate data
% simple K works with one or two subspaces
seed=7;
% K=[1 1 1 1 1 1 1 1 1 1 1 1]; %[2,2,1,1,1];
% num_subspace=length(K);
% V=sum(K);
% V = [20000,20000];
% M_Tot=2;
N=3000;
Acond=3; % 1 means orthogonal matrix
SNR=(1+999)/1;

% global sim_siva;
% sim_siva = sim_basic_SIVA(seed,K,V,M_Tot,N,Acond,SNR);
% S = sim_siva.S;
% M = sim_siva.M;
% A = sim_siva.A;
% Y = sim_siva.Y;
% X = sim_siva.genX();

S_ = {[1 2 3], [1 2 3]; ...
      [4 5 6], [4 5 6]; ...
      [7 8 9], [7 8 9]; ...
      [   10], [     ]; ...
      [   11], [     ]; ...
      [   12], [     ]; ...
      [     ], [   10]; ...
      [     ], [   11]; ...
      [     ], [   12]};

V = [20000,20000];

sim_misa = sim_MISA(seed,S_,V,N,Acond,SNR);
S = sim_misa.S;
M = sim_misa.M;
A = sim_misa.A;
Y = sim_misa.Y;
X = sim_misa.genX();

ut = utils;
% TODO try PC=15
num_pc = 12; %sum([S_{:,1}] ~= 0);
[whtM, H] = ut.doMMGPCA(X, num_pc, 'WT');

% Set Kotz parameters to multivariate laplace
K = size(S{1},1);

eta = ones(K,1);
beta = ones(K,1);
lambda = ones(K,1);

%% Set additional parameters for MISA

% Use relative gradient
gradtype = 'relative';

% Enable scale control
sc = 1;

% Turn off preprocessing (still removes the mean of the data)
preX = false;

w0 = ut.stackW({diag(pi/sqrt(3)./std(H,[],2))*eye(size(H,1))});

gica1 = MISAK(w0, 1, {eye(size(H,1))}, {H}, ...
                0.5*ones(num_pc,1), ones(num_pc,1), ones(num_pc,1), ...
                gradtype, sc, preX);

% the whitening matrix is an identity matrix, different from the whitening matrix from PCA
% sphering turns off PCA
[W1,wht] = icatb_runica(H,'weights',gica1.W{1},'ncomps',size(H,1),'sphering', 'off', 'verbose', 'off', 'posact', 'off', 'bias', 'on');
std_W1 = std(W1*H,[],2); % Ignoring wht because Infomax run with 'sphering' 'off' --> wht = eye(comps)
W1 = diag(pi/sqrt(3) ./ std_W1) * W1; 

% RUN GICA using MISA: continuing from Infomax above...
% Could use stochastic optimization, but not doing so because MISA does not implement bias weights (yet)...
% gica1.stochastic_opt('verbose', 'off', 'weights', gica1.W{1}, 'bias', 'off');%, 'block', 1100);
[wout,fval,exitflag,output] = ut.run_MISA(gica1,{W1});
std_gica1_W1 = std(gica1.Y{1},[],2);
gica1.objective(ut.stackW({diag(pi/sqrt(3) ./ std_gica1_W1)*gica1.W{1}})); % update gica1.W{1}

% Combine MISA GICA with whitening matrices to initialize multimodal model
W = cellfun(@(w) w,whtM,'Un',0);
% TODO try ICA+PCA weight and compare with PCA weight
% W = cellfun(@(w) gica1.W{1}*w,whtM,'Un',0);
W = cellfun(@(w,x) diag(pi/sqrt(3) ./ std(w*x,[],2))*w,W,X,'Un',0);

%% Set reconstruction error options:

% % Use normalized MSE
% REtype = 'NMSE';
% 
% % Use the transpose of W as the reconstruction approach
% REapproach = 'PINV'; % 'PINV' for pseudoinverse or W
% % WT: transpose when you know true solution is orthogonal
% % PINV: doesn't work if true solution is orthogonal
% 
% % Tolerance level (0 means least error possible)
% RElambda = 0;
% 
% % Other parameters required by the @MISAKRE API but not used
% REref = {};
% REreftype = 'linearreg';
% REreflambda = {.9};
% rC = {[],[]};


%% Define the starting point (for this problem an orthogonal unmixing matrix, since the features A are orthogonal).

rng(100) % set the seed for reproducibility

w0_new = ut.stackW(W(M));

%% Initialize MISA object

data1 = MISAK(w0_new, M, S, X, ...
                0.5*beta, eta, [], ...
                gradtype, sc, preX);

for mm = M
    W0{mm} = [eye(num_pc),zeros(num_pc,size(Y{M(1)},1)-num_pc)];
end
w0_short = ut.stackW(W0);

% 1: data1.Y = data1.W * X
% 2: data2.Y = data2.W * data1.Y
% By 1 and 2: data2.Y = data2.W * data1.W * X
data2 = MISAK(w0_short, data1.M, data1.S, data1.Y, ...
                0.5*beta, eta, [], ...
                gradtype, sc, preX);

%% Debug only
if debug
    % cross-modality correlation
    figure,
    tmp = corr(data1.Y{1}', data1.Y{2}');
    subplot(1,2,1),imagesc(tmp,[-1 1]);colorbar();
    
    tmp = corr(data2.Y{1}', data2.Y{2}');
    subplot(1,2,2),imagesc(tmp,[-1 1]);colorbar();
    
    % cross-dataset correlation
    figure,
    tmp = corr(data1.Y{1}', data2.Y{1}');
    subplot(1,2,1),imagesc(tmp,[-1 1]);colorbar();
    
    tmp = corr(data1.Y{1}', data1.Y{1}');
    subplot(1,2,2),imagesc(tmp,[-1 1]);colorbar();
    
    % cross-modality self-correlation
    Y_cat = cat(2, Y{1}', Y{2}');
    figure, 
    imagesc(corr(Y_cat),[-1 1]); colorbar();
    
    % cross-modality self-correlation (first subspace)
    Y_cat2 = cat(2, Y{1}(1:3,:)', Y{2}(1:3,:)');
    figure, 
    imagesc(corr(Y_cat2),[-1 1]); colorbar();
end

%% Run MISA: PRE + LBFGS-B + Nonlinear Constraint + Combinatorial Optimization
% execute_full_optimization

% Prep starting point: optimize RE to ensure initial W is in the feasible region
woutW0 = data2.stackW(data2.W);

% Define objective parameters and run optimization
f = @(x) data2.objective(x); 

c = [];
barr = 1; % barrier parameter
m = 1; % number of past gradients to use for LBFGS-B (m = 1 is equivalent to conjugate gradient)
N = size(X(M(1)),2); % Number of observations
Tol = .5*N*1e-9; % tolerance for stopping criteria

% Set optimization parameters and run
optprob = ut.getop(woutW0, f, c, barr, {'lbfgs' m}, Tol);
[wout,fval,exitflag,output] = fmincon(optprob);

% Prep and run combinatorial optimization
aux = {data2.W; data2.objective(ut.stackW(data2.W))};

final_W = cell(1,2);
for mm = M
    final_W{mm} = data2.W{mm} * W{mm}; % data2.W is 12x12, W is 12x20k
end
data1.objective(ut.stackW(final_W))
data1.MISI(A)

tic
data2.combinatorial_optim()

final_W = cell(1,2);
for mm = M
    final_W{mm} = data2.W{mm} * W{mm}; % data2.W is 12x12, data1.W is 12x20k
end
data1.objective(ut.stackW(final_W))
data1.MISI(A)
toc

for ct = 2:5
        data2.combinatorial_optim()
        optprob = ut.getop(ut.stackW(data2.W), f, c, barr, {'lbfgs' m}, Tol);
        [wout,fval,exitflag,output] = fmincon(optprob);
        aux(:,ct) = {data2.W; data2.objective_()};
        
        final_W = cell(1,2);
        for mm = M
            final_W{mm} = data2.W{mm} * W{mm}; % data2.W is 12x12, data1.W is 12x20k
        end
        data1.objective(ut.stackW(final_W))
        data1.MISI(A)
end
[~, ix] = min([aux{2,:}]);

final_W = cell(1,2);
for mm = M
    final_W{mm} = aux{1,ix}{mm} * W{mm}; % data2.W is 12x12, data1.W is 12x20k
end
data1.objective(ut.stackW(final_W));

%% Check results
fprintf("\nFinal MISI: %.4f\n\n", data1.MISI(A))
% typically, a number < 0.1 indicates successful recovery of the sources

%% Visualize recovered (mixing) patterns
% view_results
% figure,imagesc((data2.W{1}*data1.W{1})*sim_misa.A{1},max(max(abs((data2.W{1}*data1.W{1})*sim_misa.A{1}))).*[-1 1]);colorbar();
% figure,imagesc((data2.W{end}*data1.W{end})*sim_misa.A{end},max(max(abs((data2.W{end}*data1.W{end})*sim_misa.A{end}))).*[-1 1]);colorbar();

figure,imagesc(data1.W{1}*sim_misa.A{1},max(max(abs(data1.W{1}*sim_misa.A{1}))).*[-1 1]);colorbar();
figure,imagesc(data1.W{end}*sim_misa.A{end},max(max(abs(data1.W{end}*sim_misa.A{end}))).*[-1 1]);colorbar();
