function [isi, aux] = run_optimization(X,W,S,M,A)

ut=utils;
n_iter = 5; % Number of combinatorial optimization
isi = zeros(1, n_iter+1);
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

data1 = MISAK(W, M, S, X, ...
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
data1.MISI(A)
isi(1) = data1.MISI(A);

for ct = 2:n_iter
        data1.combinatorial_optim()
        optprob = ut.getop(ut.stackW(data1.W), f, c, barr, {'lbfgs' m}, Tol);
        [wout,fval,exitflag,output] = fmincon(optprob);
        aux(:,ct) = {data1.W; data1.objective_()};
        data1.MISI(A)
        isi(ct) = data1.MISI(A);
end
[~, ix] = min([aux{2,:}]);
data1.objective(ut.stackW(aux{1,ix}));

end