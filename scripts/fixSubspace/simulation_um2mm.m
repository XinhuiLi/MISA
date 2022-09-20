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
% num_subspace=30;
% K=ones(1,num_subspace);
% V=sum(K);
M_Tot=2;
N=3000;
Acond=3; % 1 means orthogonal matrix
SNR=(1+999)/1;

S_ = {[1 2], [1 2]; ...
      [3 4 5], [3 4 5]; ...
      [6 7 8 9], [6 7 8 9]; ...
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

num_pc = 12;

%% run optimization
% unimodal
[data1_um, isi_um, aux_um] = run_unimodal(X, Y, A, S, M, num_pc);

% unimodal + multimodal
[data1_ummm, isi_ummm, aux_ummm] = run_multimodal_unimodal(X, Y, A, S, M, num_pc);

% multimodal
[data1_mm, isi_mm, aux_mm] = run_mmiva(X, Y, A, S, S_, M, num_pc);

%%
outpath = '/Users/xli77/Documents/MISA/results/SIVA/fixedSubspace/um2mm/subspace_struct_234111';
save(fullfile(outpath,'um.mat'),'data1_um','isi_um','aux_um');
save(fullfile(outpath,'ummm.mat'),'data1_ummm','isi_ummm','aux_ummm');
save(fullfile(outpath,'mm.mat'),'data1_mm','isi_mm','aux_mm');

%% Visualize recovered (mixing) patterns
% view_results
n_iter = 11;

figure('Position', [10 10 1000 800]),
subplot(3,3,1);imagesc(data1_um.W{1}*sim_misa.A{1},max(max(abs(data1_um.W{1}*sim_misa.A{1}))).*[-1 1]);colorbar();
subplot(3,3,2);imagesc(data1_um.W{end}*sim_misa.A{end},max(max(abs(data1_um.W{end}*sim_misa.A{end}))).*[-1 1]);colorbar();
subplot(3,3,3);plot(1:n_iter,[aux_um{2,:}],'o-');

subplot(3,3,4);imagesc(data1_ummm.W{1}*sim_misa.A{1},max(max(abs(data1_ummm.W{1}*sim_misa.A{1}))).*[-1 1]);colorbar();
subplot(3,3,5);imagesc(data1_ummm.W{end}*sim_misa.A{end},max(max(abs(data1_ummm.W{end}*sim_misa.A{end}))).*[-1 1]);colorbar();
subplot(3,3,6);plot(1:n_iter,[aux_ummm{2,:}],'o-');

subplot(3,3,7);imagesc(data1_mm.W{1}*sim_misa.A{1},max(max(abs(data1_mm.W{1}*sim_misa.A{1}))).*[-1 1]);colorbar();
subplot(3,3,8);imagesc(data1_mm.W{end}*sim_misa.A{end},max(max(abs(data1_mm.W{end}*sim_misa.A{end}))).*[-1 1]);colorbar();
subplot(3,3,9);plot(1:n_iter,[aux_mm{2,:}],'o-');
