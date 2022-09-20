function [data1, isi, aux] = run_unimodal(X, Y, A, S, M, num_pc)
% Unimodal: PCA+ICA+CO

ut = utils;

% Use relative gradient
gradtype = 'relative';

% Enable scale control
sc = 1;

% Turn off preprocessing (still removes the mean of the data)
preX = false;

H_all = cell(1,2);
whtM_all = cell(1,2);
W_r = cell(1,2); % reduced W

for mm = M
    [whtM, H] = ut.doMMGPCA(X(mm), num_pc, 'WT');
    whtM_all(mm) = whtM;
    H_all{mm} = H;

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
    W_r{mm} = gica1.W{1};
end

% Combine MISA GICA with whitening matrices to initialize multimodal model
W = cellfun(@(w) w,whtM_all,'Un',0);
W = cellfun(@(w, wr) wr*w,W,W_r,'Un',0);
W = cellfun(@(w,x) diag(pi/sqrt(3) ./ std(w*x,[],2))*w,W,X,'Un',0);

w0_new = ut.stackW(W(M));

% Set Kotz parameters to multivariate laplace
K = size(S{1},1);
eta = ones(K,1);
beta = ones(K,1);
lambda = ones(K,1);

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

% Prep starting point: optimize RE to ensure initial W is in the feasible region
woutW0 = data2.stackW(data2.W);

% Define objective parameters and run optimization
f = @(x) data2.objective(x);

c = [];
barr = 1; % Barrier parameter
m = 1; % Number of past gradients to use for LBFGS-B (m = 1 is equivalent to conjugate gradient)
N = size(X(M(1)),2); % Number of observations
Tol = .5*N*1e-9; % Tolerance for stopping criteria
n_iter = 10; % Number of combinatorial optimization
isi = zeros(1, n_iter+1);

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