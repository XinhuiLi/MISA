close all; clear; clc;
% @gsd: generate simulated data
% @gsm: generate simulated mixing matrix

addpath("/Users/xli77/Documents/MISA/scripts");
addpath("/Users/xli77/Documents/MISA/scripts/toy_example/");

% generate data
% simple K works with one or two subspaces
seed=7;
num_subspace=5;
K=[2,2,1,1,1];%2*ones(1,num_subspace);
V=sum(K);
M_Tot=2;
N=20000;
Acond=3; % 1 means orthogonal matrix
SNR=(1+999)/1;

global sim_siva;
sim_siva = sim_basic_SIVA(seed,K,V,M_Tot,N,Acond,SNR);

S = sim_siva.S;
M = sim_siva.M;
A = sim_siva.A;
Y = sim_siva.Y;
X = sim_siva.genX();

% Set Kotz parameters to multivariate laplace
eta = ones(length(K),1);
beta = ones(length(K),1);
lambda = ones(length(K),1);

%% Set additional parameters for MISA

% Use relative gradient
gradtype = 'relative';

% Enable scale control
sc = 1;

% Turn off preprocessing (still removes the mean of the data)
preX = false;


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
W0 = cell(size(A));
for mm = M
    num_comp = size(S{mm},2);
    [u, s, v] = svd(randn(size(A{mm}(:,1:num_comp)')),'econ');
    W0{mm} = u*v';
end

ut = utils;
w0 = ut.stackW(W0(M)); % vectorize unmixing matrix for compatibility with Matlab's optimization toolbox

%% Initialize MISA object
% X_end_copy = repmat(sim_siva.Y(end), 1, 10);
% X_end_copy = cellfun(@(x) x+randn(size(x))*0.1, X_end_copy, 'un', 0);
% 
% X_new = cat(2, X, X_end_copy);
% 
% S_end_copy = repmat(S(end), 1, 10);
% S_new = cat(2, S, S_end_copy);
% 
% W0_end_copy = repmat({eye(size(W0{end}))}, 1, 10);
% W0_new = cat(2, W0, W0_end_copy);
% 
% w0_new = ut.stackW(W0_new(1:15));

%%
data1 = MISAK(w0, M, S, X, ...
                0.5*beta, eta, lambda, ...
                gradtype, sc, preX);

% data1 = MISAK(w0_new, 1:15, S_new, X_new, ...
%                 0.5*beta, eta, [], ...
%                 gradtype, sc, preX);

%% Debug only
% tmp = corr(data1.Y{1}', data1.Y{2}');
% figure,imagesc(tmp,[-1 1]);colorbar();
% 
% tmp = corr(data1.Y{1}', data1.Y{1}');
% figure,imagesc(tmp,[-1 1]);colorbar();

% TODO start from here next time, cross-correlation is not diagonal

%% Run MISA: PRE + LBFGS-B + Nonlinear Constraint + Combinatorial Optimization
% execute_full_optimization

% Prep starting point: optimize RE to ensure initial W is in the feasible region
woutW0 = data1.stackW(data1.W);

% Define objective parameters and run optimization
f = @(x) data1.objective(x); 
c = [];
barr = 1; % barrier parameter
m = 1; % number of past gradients to use for LBFGS-B (m = 1 is equivalent to conjugate gradient)
N = size(X(M(1)),2); % Number of observations
Tol = .5*N*1e-9; % tolerance for stopping criteria

% Set optimization parameters and run
optprob = ut.getop(woutW0, f, c, barr, {'lbfgs' m}, Tol);
[wout,fval,exitflag,output] = fmincon(optprob);

% Prep and run combinatorial optimization
aux = {data1.W; data1.objective(ut.stackW(data1.W))};
data1.MISI(A)

data1.combinatorial_optim()
data1.MISI(A)

for ct = 2:3
        data1.combinatorial_optim()
        optprob = ut.getop(ut.stackW(data1.W), f, c, barr, {'lbfgs' m}, Tol);
        [wout,fval,exitflag,output] = fmincon(optprob);
        aux(:,ct) = {data1.W; data1.objective_()};
        data1.MISI(A)
end
[~, ix] = min([aux{2,:}]);
data1.objective(ut.stackW(aux{1,ix}));

%% Check results
fprintf("\nFinal MISI: %.4f\n\n", data1.MISI(A))
% typically, a number < 0.1 indicates successful recovery of the sources

%% Visualize recovered (mixing) patterns
% view_results
figure,imagesc(data1.W{1}*sim_siva.A{1},max(max(abs(data1.W{1}*sim_siva.A{1}))).*[-1 1]);colorbar();
% figure,imagesc(data1.W{5}*sim_siva.A{5},max(max(abs(data1.W{5}*sim_siva.A{5}))).*[-1 1]);colorbar();
figure,imagesc(data1.W{end}*sim_siva.A{end},max(max(abs(data1.W{end}*sim_siva.A{end}))).*[-1 1]);colorbar();

%%
figure,imagesc(aux{1,1}{1}*sim_siva.A{1},max(max(abs(aux{1,1}{1}*sim_siva.A{1}))).*[-1 1]);colorbar();
figure,imagesc(aux{1,1}{end}*sim_siva.A{end},max(max(abs(aux{1,1}{end}*sim_siva.A{end}))).*[-1 1]);colorbar();

%%
% Y_concat = cat(1, data1.Y{:});
% Y_corr = corr(Y_concat');
% figure,imagesc(Y_corr,[-1 1]);colorbar;

%%
% sim_siva_Y_concat = cat(1, sim_siva.Y{:});
% sim_siva_Y_corr = corr(sim_siva_Y_concat');
% figure,imagesc(sim_siva_Y_corr,[-1 1]);colorbar;