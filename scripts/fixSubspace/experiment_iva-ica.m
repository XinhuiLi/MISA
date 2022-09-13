close all; clear; clc;
% @gsd: generate simulated data
% @gsm: generate simulated mixing matrix

format compact

debug = false;

addpath("/Users/xli77/Documents/MISA/scripts");
addpath("/Users/xli77/Documents/MISA/scripts/toy_example/");
addpath(genpath('/Users/xli77/Documents/gift/GroupICATv4.0c'));

n_iter = 1; % Number of combinatorial optimization

% generate data
% simple K works with one or two subspaces
seed=7;
Acond=3; % 1 means orthogonal matrix
SNR=(1+999)/1;

S_ = {[1 2 3], [1 2 3]; ...
      [4 5 6], [4 5 6]; ...
      [7 8 9], [7 8 9]; ...
      [   10], [     ]; ...
      [   11], [     ]; ...
      [   12], [     ]; ...
      [     ], [   10]; ...
      [     ], [   11]; ...
      [     ], [   12]};

M = [1, 2];

X = load('/Users/xli77/Documents/MISA/MISA-data/sMRI-fMRI/X.mat').X;

ut = utils;
num_pc = 12; %sum([S_{:,1}] ~= 0);

% Set Kotz parameters to multivariate laplace
K = size(S_,1);
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


%%



% TODO run on X(1) and (2) separately
[whtM, H] = ut.doMMGPCA(X(1), num_pc, 'WT');
w0 = ut.stackW({diag(pi/sqrt(3)./std(H,[],2))*eye(size(H,1))});
gica1 = MISAK(w0, 1, {eye(size(H,1))}, {H}, ...
                0.5*ones(num_pc,1), ones(num_pc,1), ones(num_pc,1), ...
                gradtype, sc, preX);
% the whitening matrix is an identity matrix, different from the whitening matrix from PCA
% sphering turns off PCA
[Wica,wht] = icatb_runica(H,'weights',gica1.W{1},'ncomps',size(H,1),'sphering', 'off', 'verbose', 'off', 'posact', 'off', 'bias', 'on');
std_Wica = std(Wica*H,[],2); % Ignoring wht because Infomax run with 'sphering' 'off' --> wht = eye(comps)
Wica = diag(pi/sqrt(3) ./ std_Wica) * Wica; 

% RUN GICA using MISA: continuing from Infomax above...
% Could use stochastic optimization, but not doing so because MISA does not implement bias weights (yet)...
% gica1.stochastic_opt('verbose', 'off', 'weights', gica1.W{1}, 'bias', 'off');%, 'block', 1100);
[wout,fval,exitflag,output] = ut.run_MISA(gica1,{Wica});
std_gica1_W1 = std(gica1.Y{1},[],2);
gica1.objective(ut.stackW({diag(pi/sqrt(3) ./ std_gica1_W1)*gica1.W{1}})); % update gica1.W{1}

% Combine MISA GICA with whitening matrices to initialize multimodal model
W1 = cellfun(@(w) gica1.W{1}*w,whtM,'Un',0);
W1 = cellfun(@(w,x) diag(pi/sqrt(3) ./ std(w*x,[],2))*w,W1,X(1),'Un',0);

[whtM, H] = ut.doMMGPCA(X(2), num_pc, 'WT');
w0 = ut.stackW({diag(pi/sqrt(3)./std(H,[],2))*eye(size(H,1))});
gica1 = MISAK(w0, 1, {eye(size(H,1))}, {H}, ...
                0.5*ones(num_pc,1), ones(num_pc,1), ones(num_pc,1), ...
                gradtype, sc, preX);
[Wica,wht] = icatb_runica(H,'weights',gica1.W{1},'ncomps',size(H,1),'sphering', 'off', 'verbose', 'off', 'posact', 'off', 'bias', 'on');
std_Wica = std(Wica*H,[],2);
Wica = diag(pi/sqrt(3) ./ std_Wica) * Wica; 
[wout,fval,exitflag,output] = ut.run_MISA(gica1,{Wica});
std_gica1_W1 = std(gica1.Y{1},[],2);
gica1.objective(ut.stackW({diag(pi/sqrt(3) ./ std_gica1_W1)*gica1.W{1}}));
W2 = cellfun(@(w) gica1.W{1}*w,whtM,'Un',0);
W2 = cellfun(@(w,x) diag(pi/sqrt(3) ./ std(w*x,[],2))*w,W2,X(2),'Un',0);


%% Define the starting point (for this problem an orthogonal unmixing matrix, since the features A are orthogonal).

rng(100) % set the seed for reproducibility

% TODO stack 2 W
% w0_new = ut.stackW(W(M));
W = cell(1,2);
W(1) = W1;
W(2) = W2;
w0_new = ut.stackW(W(M));

%% Initialize MISA object

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

data1 = MISAK(w0_new, M, S, X, ...
                0.5*beta, eta, [], ...
                gradtype, sc, preX);

for mm = M
    W0{mm} = [eye(num_pc)];
end
w0_short = ut.stackW(W0);

% 1: data1.Y = data1.W * X
% 2: data2.Y = data2.W * data1.Y
% By 1 and 2: data2.Y = data2.W * data1.W * X
data2 = MISAK(w0_short, data1.M, data1.S, data1.Y, ...
                0.5*beta, eta, [], ...
                gradtype, sc, preX);

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
    final_W{mm} = data2.W{mm} * W{mm};
end
data1.objective(ut.stackW(final_W))

% TODO run the loop once
for ct = 1:n_iter
        data2.combinatorial_optim()
        optprob = ut.getop(ut.stackW(data2.W), f, c, barr, {'lbfgs' m}, Tol);
        [wout,fval,exitflag,output] = fmincon(optprob);
        aux(:,ct) = {data2.W; data2.objective_()};
        
        final_W = cell(1,2);
        for mm = M
            final_W{mm} = data2.W{mm} * W{mm};
        end
        data1.objective(ut.stackW(final_W))
end
[~, ix] = min([aux{2,:}]);

final_W = cell(1,2);
for mm = M
    final_W{mm} = aux{1,ix}{mm} * W{mm};
end
data1.objective(ut.stackW(final_W));
data2.objective(ut.stackW(aux{1,ix}));

%%
final_Y = cell(1,2);
for mm = M
    final_Y{mm} = aux{1,ix}{mm} * (W{mm} * data1.X{mm});
end

%%
% figure,imagesc(corr(final_Y{1}',data2.Y{1}'),max(max(abs(corr(final_Y{1}',data2.Y{1}')))).*[-1 1]);colorbar();
% figure,imagesc(corr(final_Y{2}',data2.Y{2}'),max(max(abs(corr(final_Y{2}',data2.Y{2}')))).*[-1 1]);colorbar();

%% Check results
% fprintf("\nFinal MISI: %.4f\n\n", data1.MISI(A))
% typically, a number < 0.1 indicates successful recovery of the sources

%% Visualize recovered (mixing) patterns
% view_results
% figure,plot(1:n_iter,[aux{2,:}],'o-');
figure,imagesc(corr(data2.Y{1}',data2.Y{2}'),max(max(abs(corr(data2.Y{1}',data2.Y{2}')))).*[-1 1]);colorbar();
figure,imagesc(corr(data2.Y{1}',data2.Y{1}'),max(max(abs(corr(data2.Y{1}',data2.Y{1}')))).*[-1 1]);colorbar();
figure,imagesc(corr(data2.Y{2}',data2.Y{2}'),max(max(abs(corr(data2.Y{2}',data2.Y{2}')))).*[-1 1]);colorbar();

% figure,imagesc(corr(data1.Y{1}',data1.Y{2}'),max(max(abs(corr(data1.Y{1}',data1.Y{2}')))).*[-1 1]);colorbar();
% figure,imagesc(corr(data1.Y{1}',data1.Y{1}'),max(max(abs(corr(data1.Y{1}',data1.Y{1}')))).*[-1 1]);colorbar();
% figure,imagesc(corr(data1.Y{2}',data1.Y{2}'),max(max(abs(corr(data1.Y{2}',data1.Y{2}')))).*[-1 1]);colorbar();

% figure,imagesc(corr(data2.X{1}',data2.X{1}'),max(max(abs(corr(data2.X{1}',data2.X{1}')))).*[-1 1]);colorbar();

% data1.objective(ut.stackW(W));
% s = svds(data1.Y{1},12);
% sd = diag(s)';

%%
% save '/Users/xli77/Documents/MISA/results/SIVA/fixedSubspace/neuroimaging/without_scaling/aux.mat' 'aux'
% save '/Users/xli77/Documents/MISA/results/SIVA/fixedSubspace/neuroimaging/without_scaling/data1.mat' 'data1'
