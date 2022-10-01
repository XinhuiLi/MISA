close all; clear; clc;
% @gsd: generate simulated data
% @gsm: generate simulated mixing matrix

format compact

debug = false;

addpath("/Users/xli77/Documents/MISA/scripts");
addpath("/Users/xli77/Documents/MISA/scripts/toy_example/");
addpath(genpath('/Users/xli77/Documents/gift/GroupICATv4.0c'));

% generate data
% simple K works with one or two subspaces
seed=7;
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

M = [1, 2];

X = load('/Users/xli77/Documents/MISA/MISA-data/sMRI-fMRI/X.mat').X;

% Set Kotz parameters to multivariate laplace
K = size(S_,1);
eta = ones(K,1);
beta = ones(K,1);
lambda = ones(K,1);

S = cell(size(M));
% from gsd.m
for mm = M
    if issparse(S_{mm})
        S{mm} = S_{mm};
    else
        ii = [];
        jj = [];
        for ii_ = 1:K
            jj_ = length(S_{ii_,mm});
            if jj_ ~= 0
                jj = [jj S_{ii_,mm}];
                ii = [ii ii_*ones(1,jj_)];
            end
        end
        S{mm} = sparse(ii, jj, ones(1,sum([S_{:,mm}] ~= 0)), ...
            K, sum([S_{:,mm}] ~= 0), sum([S_{:,mm}] ~= 0));
    end
end

num_pc = 12;
num_iter = 10;

%%
% unimodal
[data1_um, aux_um] = run_unimodal(X, S, M, num_pc, num_iter);

% unimodal + multimodal
[data1_ummm, aux_ummm] = run_multimodal_unimodal(X, S, M, num_pc, num_iter);

%%
outpath = '/Users/xli77/Documents/MISA/results/SIVA/fixedSubspace/um2mm/subspace_struct_234111';

% save(fullfile(outpath,'um_neuroimaging.mat'),'data1_um','aux_um');
% save(fullfile(outpath,'ummm_neuroimaging.mat'),'data1_ummm','aux_ummm');

load(fullfile(outpath,'um_neuroimaging.mat'),'data1_um','aux_um');
load(fullfile(outpath,'ummm_neuroimaging.mat'),'data1_ummm','aux_ummm');

%%
num_row = 2;
num_col = 4;

figure('Position', [10 10 1200 400]),
subplot(num_row,num_col,1);
imagesc(abs(corr(data1_um.Y{1}',data1_um.Y{1}')),[0 1]);colorbar();
title("Y, M1");
subplot(num_row,num_col,2);
imagesc(abs(corr(data1_um.Y{2}',data1_um.Y{2}')),[0 1]);colorbar();
title("Y, M2");
subplot(num_row,num_col,3);
imagesc(abs(corr(data1_um.Y{1}',data1_um.Y{2}')),[0 1]);colorbar();
title("Y, M1&2");
subplot(num_row,num_col,4);
plot(1:num_iter+1,[aux_um{2,:}],'o-');
title("Loss");

subplot(num_row,num_col,5);
imagesc(abs(corr(data1_ummm.Y{1}',data1_ummm.Y{1}')),[0 1]);colorbar();
title("Y, M1");
subplot(num_row,num_col,6);
imagesc(abs(corr(data1_ummm.Y{2}',data1_ummm.Y{2}')),[0 1]);colorbar();
title("Y, M2");
subplot(num_row,num_col,7);
imagesc(abs(corr(data1_ummm.Y{1}',data1_ummm.Y{2}')),[0 1]);colorbar();
title("Y, M1&2");
subplot(num_row,num_col,8);
plot(1:num_iter+1,[aux_ummm{2,:}],'o-');
title("Loss");
