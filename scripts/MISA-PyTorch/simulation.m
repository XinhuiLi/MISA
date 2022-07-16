close all; clear; clc;
% @gsd: generate simulated data
% @gsm: generate simulated mixing matrix

% function simulation()

addpath("/Users/xli77/Documents/MISA/scripts");
addpath("/Users/xli77/Documents/MISA/scripts/toy_example/");

% generate data
% simple K works with one or two subspaces
seed=7;
num_subspace=5;
K=ones(1,num_subspace);
V=6; % output dimension
M_Tot=3;
N=1000;
Acond=3; % 1 means orthogonal matrix
SNR=(1+999)/1;

sim_siva = sim_basic_SIVA(seed,K,V,M_Tot,N,Acond,SNR);

S = sim_siva.S;
M = sim_siva.M;
A = sim_siva.A;
Y = sim_siva.Y;
X = sim_siva.genX();

% Set Kotz parameters to multivariate laplace
eta = ones(length(K),1);
beta = 0.5 * ones(length(K),1);
lambda = ones(length(K),1);


%% Set additional parameters for MISA

% Use relative gradient
gradtype = 'relative';

% Enable scale control
sc = 1;

% Turn off preprocessing (still removes the mean of the data)
preX = false;


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

%%
data1 = MISAK(w0, M, S, X, ...
                0.5*beta, eta, lambda, ...
                gradtype, sc, preX);

%%
data1.objective(w0)
