function [data1, isi, aux] = run_mmiva(X, Y, A, S, S_, M, num_pc)
% MMIVA

n_iter = 10; % Number of combinatorial optimization

ut = utils;

% Use relative gradient
gradtype = 'relative';

% Enable scale control
sc = 1;

% Turn off preprocessing (still removes the mean of the data)
preX = false;

M_Tot = length(M);

[whtM, H] = ut.doMMGPCA(X, num_pc, 'WT');

% Set Kotz parameters to multivariate laplace
K = size(S{1},1);

eta = ones(K,1);
beta = ones(K,1);
lambda = ones(K,1);

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
% W = cellfun(@(w) gica1.W{1}*w,whtM,'Un',0);
W = cellfun(@(w,x) diag(pi/sqrt(3) ./ std(w*x,[],2))*w,W,X,'Un',0);

w0_new = ut.stackW(W(M));

data1 = MISAK(w0_new, M, S, X, ...
    0.5*beta, eta, [], ...
    gradtype, sc, preX);

for mm = M
    W0{mm} = [eye(num_pc),zeros(num_pc,size(Y{M(1)},1)-num_pc)];
end
w0_short = ut.stackW(W0);

% MMIVA, S
S2_ = num2cell(repmat((1:num_pc)', 1, M_Tot));

S2 = cell(size(M));
for mm = M
    if issparse(S2_{mm})
        S2{mm} = S2_{mm};
    else
        ii = [];
        jj = [];
        for ii_ = 1:12 % Check with Rogers!!!
            jj_ = length(S2_{ii_,mm});
            if jj_ ~= 0
                jj = [jj S2_{ii_,mm}];
                ii = [ii ii_*ones(1,jj_)];
            end
        end
        S2{mm} = sparse(ii, jj, ones(1,sum([S2_{:,mm}] ~= 0)), ...
            12, sum([S2_{:,mm}] ~= 0), sum([S2_{:,mm}] ~= 0));
        % Check with Rogers!!!
    end
end

eta = ones(12,1)*eta(1);
beta = ones(12,1)*beta(1);
data2 = MISAK(w0_short, data1.M, S2, data1.Y, ...
                0.5*beta, eta, [], ...
                gradtype, sc, preX);

% Prep starting point: optimize RE to ensure initial W is in the feasible region
woutW0 = data2.stackW(data2.W);

% Define objective parameters and run optimization
f = @(x) data2.objective(x);

c = [];
barr = 1; % Barrier parameter
m = 1; % Number of past gradients to use for LBFGS-B (m = 1 is equivalent to conjugate gradient)
N = size(X(M(1)),2); % Number of observations
Tol = .5*N*1e-9; % Tolerance for stopping criteria
isi = zeros(1, n_iter+1);

% MMIVA
% make S_ a sparse matrix sS_
sS_ = cell(size(M));
for mm = M
    if issparse(S_{mm})
        sS_{mm} = S_{mm};
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
        sS_{mm} = sparse(ii, jj, ones(1,sum([S_{:,mm}] ~= 0)), ...
            K, sum([S_{:,mm}] ~= 0), sum([S_{:,mm}] ~= 0));
    end
end

% MMIVA
data2.update(sS_, data2.M, data2.beta(1), [], data2.eta(1));
woutW0 = data2.stackW(data2.W);
optprob = ut.getop(woutW0, f, c, barr, {'lbfgs' m}, Tol);
[wout,fval,exitflag,output] = fmincon(optprob);

% Prep and run combinatorial optimization
aux = {data2.W; data2.objective(ut.stackW(data2.W))};

final_W = cell(1,2);
for mm = M
    final_W{mm} = data2.W{mm} * W{mm}; % data2.W is 12x12, W is 12x20k
end
data1.objective(ut.stackW(final_W))
isi(1) = data1.MISI(A)

for ct = 2:n_iter+1
    data2.combinatorial_optim()
    optprob = ut.getop(ut.stackW(data2.W), f, c, barr, {'lbfgs' m}, Tol);
    [wout,fval,exitflag,output] = fmincon(optprob);
    aux(:,ct) = {data2.W; data2.objective_()};

    final_W = cell(1,2);
    for mm = M
        final_W{mm} = data2.W{mm} * W{mm}; % data2.W is 12x12, data1.W is 12x20k
    end
    data1.objective(ut.stackW(final_W))
    isi(ct) = data1.MISI(A)
end
[~, ix] = min([aux{2,:}]);

final_W = cell(1,2);
for mm = M
    final_W{mm} = aux{1,ix}{mm} * W{mm}; % data2.W is 12x12, data1.W is 12x20k
end
data1.objective(ut.stackW(final_W));

end