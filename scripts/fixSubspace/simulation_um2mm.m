close all; clear; clc;
% @gsd: generate simulated data
% @gsm: generate simulated mixing matrix

debug = false;

addpath("/Users/xli77/Documents/MISA/scripts");
addpath("/Users/xli77/Documents/MISA/scripts/fixSubspace/");
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

% S_ = {[1 2 3], [1 2 3]; ...
%       [4 5 6], [4 5 6]; ...
%       [7 8 9], [7 8 9]; ...
%       [   10], [     ]; ...
%       [   11], [     ]; ...
%       [   12], [     ]; ...
%       [     ], [   10]; ...
%       [     ], [   11]; ...
%       [     ], [   12]};
% 
% S_ = {[1 2], [1 2]; ...
%       [3 4 5], [3 4 5]; ...
%       [6 7 8 9], [6 7 8 9]};

% S_ = {[1 2], [1 2]; ...
%       [  3], [   ]; ...
%       [  4], [   ]; ...
%       [   ], [  3]; ...
%       [   ], [  4]};

V = [20000,20000];

sim_misa = sim_MISA(seed,S_,V,N,Acond,SNR);
S = sim_misa.S;
M = sim_misa.M;
A = sim_misa.A;
Y = sim_misa.Y;
X = sim_misa.genX();

num_pc = 12;
num_iter = 5;

%% run optimization
% unimodal
% [data1_um, aux_um, isi_um] = run_unimodal(X, Y, A, S, M, num_pc, num_iter);

% unimodal + multimodal
% [data1_ummm, aux_ummm, isi_ummm] = run_multimodal_unimodal(X, Y, A, S, M, num_pc, num_iter);

% multimodal
[data1_mm, aux_mm, isi_mm] = run_mmiva(X, Y, A, S, M, num_pc, num_iter);
% [data1_mm_test, aux_mm_test, isi_mm_test] = run_mmiva_test(X, Y, A, S, M, num_pc, num_iter);

%%
S1 = {[1 2], [1 2]; ...
      [3 4], [3 4]; ...
      [5 6], [5 6]; ...
      [7 8], [7 8]; ...
      [  9], [   ]; ...
      [ 10], [   ]; ...
      [ 11], [   ]; ...
      [ 12], [   ]; ...
      [   ], [  9]; ...
      [   ], [ 10]; ...
      [   ], [ 11]; ...
      [   ], [ 12]};

S1_ = cell(size(M));
for mm = M
    if issparse(S1{mm})
        S1_{mm} = S1{mm};
    else
        ii = [];
        jj = [];
        for ii_ = 1:num_pc
            jj_ = length(S1{ii_,mm});
            if jj_ ~= 0
                jj = [jj S1{ii_,mm}];
                ii = [ii ii_*ones(1,jj_)];
            end
        end
        S1_{mm} = sparse(ii, jj, ones(1,sum([S1{:,mm}] ~= 0)), ...
            num_pc, sum([S1{:,mm}] ~= 0), sum([S1{:,mm}] ~= 0));
    end
end

[data1_um_s1, aux_um_s1, isi_um_s1] = run_unimodal(X, Y, A, S1_, M, num_pc, num_iter);
[data1_ummm_s1, aux_ummm_s1, isi_ummm_s1] = run_multimodal_unimodal(X, Y, A, S1_, M, num_pc, num_iter);

% S2 = [3,3,3,1,1,1];
% S3 = [4,4,1,1,1,1];

%%
% outpath = '/Users/xli77/Documents/MISA/results/SIVA/fixedSubspace/um2mm/subspace_struct_234111';
% outpath = '/Users/xli77/Documents/MISA/results/SIVA/fixedSubspace/um2mm/subspace_struct_333111';
% outpath = '/Users/xli77/Documents/MISA/results/SIVA/fixedSubspace/um2mm/subspace_struct_234';
% outpath = '/Users/xli77/Documents/MISA/results/SIVA/fixedSubspace/um2mm/subspace_struct_333111_sample2k';
% outpath = '/Users/xli77/Documents/MISA/results/SIVA/fixedSubspace/um2mm/subspace_struct_211_sample2k';

% save(fullfile(outpath,'um.mat'),'data1_um','isi_um','aux_um');
% save(fullfile(outpath,'ummm.mat'),'data1_ummm','isi_ummm','aux_ummm');
% save(fullfile(outpath,'mm.mat'),'data1_mm','isi_mm','aux_mm');

% load(fullfile(outpath,'um.mat'),'data1_um','isi_um','aux_um');
% load(fullfile(outpath,'ummm.mat'),'data1_ummm','isi_ummm','aux_ummm');
% load(fullfile(outpath,'mm.mat'),'data1_mm','isi_mm','aux_mm');

%% Visualize recovered (mixing) patterns
% view_results
num_row = 3;
num_col = 4;

figure('Position', [10 10 1200 720]),
subplot(num_row,num_col,1);
imagesc(data1_um.W{1}*sim_misa.A{1},max(max(abs(data1_um.W{1}*sim_misa.A{1}))).*[-1 1]);colorbar();
title("WA, M1");
subplot(num_row,num_col,2);
imagesc(data1_um.W{end}*sim_misa.A{end},max(max(abs(data1_um.W{end}*sim_misa.A{end}))).*[-1 1]);colorbar();
title("WA, M2");
subplot(num_row,num_col,3);
imagesc(cat(1,data1_um.W{:})*cat(2,sim_misa.A{:}),max(max(abs(cat(1,data1_um.W{:})*cat(2,sim_misa.A{:})))).*[-1 1]);colorbar();
title("WA, M1&2");
subplot(num_row,num_col,4);
plot(1:num_iter+1,[aux_um{2,:}],'o-');
title("Loss");

subplot(num_row,num_col,5);
imagesc(data1_ummm.W{1}*sim_misa.A{1},max(max(abs(data1_ummm.W{1}*sim_misa.A{1}))).*[-1 1]);colorbar();
title("WA, M1");
subplot(num_row,num_col,6);
imagesc(data1_ummm.W{end}*sim_misa.A{end},max(max(abs(data1_ummm.W{end}*sim_misa.A{end}))).*[-1 1]);colorbar();
title("WA, M2");
subplot(num_row,num_col,7);
imagesc(cat(1,data1_ummm.W{:})*cat(2,sim_misa.A{:}),max(max(abs(cat(1,data1_ummm.W{:})*cat(2,sim_misa.A{:})))).*[-1 1]);colorbar();
title("WA, M1&2");
subplot(num_row,num_col,8);
plot(1:num_iter+1,[aux_ummm{2,:}],'o-');
title("Loss");

subplot(num_row,num_col,9);
imagesc(data1_mm.W{1}*sim_misa.A{1},max(max(abs(data1_mm.W{1}*sim_misa.A{1}))).*[-1 1]);colorbar();
title("WA, M1");
subplot(num_row,num_col,10);
imagesc(data1_mm.W{end}*sim_misa.A{end},max(max(abs(data1_mm.W{end}*sim_misa.A{end}))).*[-1 1]);colorbar();
title("WA, M2");
subplot(num_row,num_col,11);
imagesc(cat(1,data1_mm.W{:})*cat(2,sim_misa.A{:}),max(max(abs(cat(1,data1_mm.W{:})*cat(2,sim_misa.A{:})))).*[-1 1]);colorbar();
title("WA, M1&2");
subplot(num_row,num_col,12);
plot(1:num_iter+1,[aux_mm{2,:}],'o-');
title("Loss");

% subplot(num_row,num_col,9);
% imagesc(data1_mm_test.W{1}*sim_misa.A{1},max(max(abs(data1_mm_test.W{1}*sim_misa.A{1}))).*[-1 1]);colorbar();
% title("WA, M1");
% subplot(num_row,num_col,10);
% imagesc(data1_mm_test.W{end}*sim_misa.A{end},max(max(abs(data1_mm_test.W{end}*sim_misa.A{end}))).*[-1 1]);colorbar();
% title("WA, M2");
% subplot(num_row,num_col,11);
% imagesc(cat(1,data1_mm_test.W{:})*cat(2,sim_misa.A{:}),max(max(abs(cat(1,data1_mm_test.W{:})*cat(2,sim_misa.A{:})))).*[-1 1]);colorbar();
% title("WA, M1&2");
% subplot(num_row,num_col,12);
% plot(1:num_iter+1,[aux_mm_test{2,:}],'o-');
% title("Loss");

%% MMIVA
num_row = 1;
num_col = 4;

figure('Position', [10 10 1200 200]),
subplot(num_row,num_col,1);
imagesc(data1_mm.W{1}*sim_misa.A{1},max(max(abs(data1_mm.W{1}*sim_misa.A{1}))).*[-1 1]);colorbar();
title("WA, M1");
subplot(num_row,num_col,2);
imagesc(data1_mm.W{end}*sim_misa.A{end},max(max(abs(data1_mm.W{end}*sim_misa.A{end}))).*[-1 1]);colorbar();
title("WA, M2");
subplot(num_row,num_col,3);
imagesc(cat(1,data1_mm.W{:})*cat(2,sim_misa.A{:}),max(max(abs(cat(1,data1_mm.W{:})*cat(2,sim_misa.A{:})))).*[-1 1]);colorbar();
title("WA, M1&2");
subplot(num_row,num_col,4);
plot(1:num_iter+1,[aux_mm{2,:}],'o-');
title("Loss");

%%
% ind = [ [2,12], [5,8,10], [1,3,4,6], [7,9,11] ];
% for mm = M
%     W_shuffled{mm} = data2.W{mm}(ind,:);
% end
% data2_w = ut.stackW(W_shuffled);
% data2.objective(data2_w)
