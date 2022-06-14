% close all; clear; clc;
% @gsd: generate simulated data
% @gsm: generate simulated mixing matrix

function simulation_beta_avg()

addpath("/Users/xli77/Documents/MISA/scripts");
addpath("/Users/xli77/Documents/MISA/scripts/toy_example/");

% generate data
% simple K works with one or two subspaces
seed=7;
K=2*ones(1,10);
V=sum(K);
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
% get_MISA_SIVA_parameters

% size(S{M(1)},2): Number of components
S = mat2cell(repmat((1:size(S{M(1)},2)), M_Tot, 1), ones(1,M_Tot), size(S{M(1)},2))';

S = cellfun(@(s) sparse(s, s, ones(size(s)), length(s), length(s), length(s)),... 
    S, 'un', 0);

K = size(S{M(1)},1);   % Number of subspaces

% Set Kotz parameters to multivariate laplace
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


%% Set reconstruction error options:

% Use normalized MSE
REtype = 'NMSE';

% Use the transpose of W as the reconstruction approach
REapproach = 'PINV'; % 'PINV' for pseudoinverse or W
% WT: transpose when you know true solution is orthogonal
% PINV: doesn't work if true solution is orthogonal

% Tolerance level (0 means least error possible)
RElambda = 0;

% Other parameters required by the @MISAKRE API but not used
REref = {};
REreftype = 'linearreg';
REreflambda = {.9};
rC = {[],[]};


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


%% Initialize MISA object

data1 = MISAK(w0, M, S, X, ...
                0.5*beta, eta, [], ...
                gradtype, sc, preX);

data2 = MISAK(w0, M, S, X, ...
                0.35*beta, eta, [], ...
                gradtype, sc, preX);

data3 = MISAK(w0, M, S, X, ...
                0.65*beta, eta, [], ...
                gradtype, sc, preX);

data4 = MISAK(w0, M, S, X, ...
                beta, eta, [], ...
                gradtype, sc, preX);

%% Run MISA: PRE + LBFGS-B + Nonlinear Constraint + Combinatorial Optimization
% execute_full_optimization

% Prep starting point: optimize RE to ensure initial W is in the feasible region
woutW0 = data1.stackW(data1.W);

% Define objective parameters and run optimization
f = @(x) MISA_mean(x); 
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

data1.combinatorial_optim()
data1.MISI(A)

%% Check results
fprintf("\nFinal MISI: %.4f\n\n", data1.MISI(A))
% typically, a number < 0.1 indicates successful recovery of the sources

%% Visualize recovered (mixing) patterns
% view_results
% figure,imagesc(data1.W{1}*sim_siva.A{1},max(max(abs(data1.W{1}*sim_siva.A{1}))).*[-1 1]);colorbar();
% figure,imagesc(data1.W{end}*sim_siva.A{end},max(max(abs(data1.W{end}*sim_siva.A{end}))).*[-1 1]);colorbar();

    function [J, gJ] = MISA_mean(x)
        % Calculate mean and gradient of x across data objects
        % TODO argmax of gradients
        J = (data1.objective(x)+data2.objective(x)+data3.objective(x)+data4.objective(x))/4;
        if nargout > 1
            [J1, gJ1] = data1.objective(x);
            [J2, gJ2] = data2.objective(x);
            [J3, gJ3] = data3.objective(x);
            [J4, gJ4] = data4.objective(x);
            J = (J1+J2+J3+J4)/4;
            gJ = (gJ1+gJ2+gJ3+gJ4)/4;
        end
    end

end