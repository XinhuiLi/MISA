close all; clear; clc;
% @gsd: generate simulated data
% @gsm: generate simulated mixing matrix

addpath("/Users/xli77/Documents/MISA/scripts");
addpath("/Users/xli77/Documents/MISA/scripts/toy_example/");
addpath(genpath('/Users/xli77/Documents/gift/GroupICATv4.0c'));

load("/Users/xli77/Documents/MISA/MISA-data/sMRI-fMRI-DTI/W.mat");
load("/Users/xli77/Documents/MISA/MISA-data/sMRI-fMRI-DTI/X.mat");

%%
% WT = cellfun(@(x) (x'), W, 'un', 0);
% save("/Users/xli77/Documents/MISA-pytorch/simulation_data/mmiva.mat", "X", "W", '-v7.3');
% save("/Users/xli77/Documents/MISA-pytorch/simulation_data/mmiva_wt.mat", "X", "WT", '-v7.3');

%%
K = 30;
M = [1,2,3];
S = {sparse(eye(K)), sparse(eye(K)), sparse(eye(K))};

%%
ut=utils;

K = size(S{M(1)},1);   % Number of subspaces

% Set Kotz parameters to multivariate laplace
eta = ones(K,1);
beta = ones(K,1);
lambda = ones(K,1);

% Use relative gradient
gradtype = 'relative';

% Enable scale control
sc = 1;

% Turn off preprocessing (still removes the mean of the data)
preX = false;

w = ut.stackW(W(M));

data1 = MISAK(w, M, S, X, ...
                0.5*beta, eta, [], ...
                gradtype, sc, preX);

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

%%
figure,imagesc(corr(data1.Y{1}',data1.Y{2}'),max(max(abs(corr(data1.Y{1}',data1.Y{2}')))).*[-1 1]);colorbar();
figure,imagesc(corr(data1.Y{1}',data1.Y{3}'),max(max(abs(corr(data1.Y{1}',data1.Y{3}')))).*[-1 1]);colorbar();
figure,imagesc(corr(data1.Y{2}',data1.Y{3}'),max(max(abs(corr(data1.Y{2}',data1.Y{3}')))).*[-1 1]);colorbar();

%%
% save('/Users/xli77/Documents/MISA/MISA-data/sMRI-fMRI-DTI/aux.mat','aux');
% save('/Users/xli77/Documents/MISA/MISA-data/sMRI-fMRI-DTI/data1.mat','data1','-v7.3');
% Y = data1.Y;
% save('/Users/xli77/Documents/MISA/MISA-data/sMRI-fMRI-DTI/data1_Y.mat','Y','-v7.3');
