close all; clear; clc;
% @gsd: generate simulated data
% @gsm: generate simulated mixing matrix

addpath("/Users/xli77/Documents/MISA/scripts");
addpath("/Users/xli77/Documents/MISA/scripts/toy_example/");

% generate data
% simple K works with one or two subspaces
seed=7;
% K=[ones(1,20), 2];
K=1:6;
V=sum(K);%2*sum(K);
M_Tot=5;
N=20000;
Acond=3; % 1 means orthogonal matrix
SNR=(1+999)/1;

sim_siva = sim_basic_SIVA(seed,K,V,M_Tot,N,Acond,SNR);

S = sim_siva.S;
M = sim_siva.M;
A = sim_siva.A;
Y = sim_siva.Y;
X = sim_siva.genX();

%%
get_MISA_SIVA_parameters

%% Initialize MISA object
data1 = MISAK(w0, M, S, X, ...
                beta, eta, lambda, ...
                gradtype, sc, preX);
%                 REtype, REapproach, RElambda, ...
%                 REref, REreftype, REreflambda, rC);

%% Debug only
% tmp = corr(data1.Y{1}', data1.Y{2}');
% figure,imagesc(tmp,[-1 1]);colorbar();
% 
% tmp = corr(data1.Y{1}', data1.Y{1}');
% figure,imagesc(tmp,[-1 1]);colorbar();

% TODO start from here next time, cross-correlation is not diagonal

%% Run MISA: PRE + LBFGS-B + Nonlinear Constraint + Combinatorial Optimization
execute_full_optimization

% NOTE: toggle lines 41-43 in @utils/getop.m to see detailed optimization iterations (considerably slower)


%% Check results
fprintf("\nFinal MISI: %.4f\n\n", data1.MISI(A))
% typically, a number < 0.1 indicates successful recovery of the sources


%% Visualize recovered (mixing) patterns
% view_results

