K = size(S{M(1)},1);   % Number of subspaces

% Set Kotz parameters to multivariate lapace
eta = ones(K,1);
beta = .5*ones(K,1);
lambda = ones(K,1);


%% Set additional parameters for MISA

% Use relative gradient
gradtype = 'relative';

% Enable scale control
sc = 1;

% Turn off preprocessing (still removes the mean of the data)
preX = false;


%% Define the starting point (for this problem an orthogonal unmixing matrix, since the features A are orthogonal).

rng(100) % set the seed for reproducibility
W0 = cell(1,length(M));
W0{1} = [1,2,3;3,2,1;2,1,3];
W0{2} = [3,2,1;2,1,3;1,2,3];
W0{3} = [2,1,3;1,2,3;3,2,1];

ut = utils;
w0 = ut.stackW(W0(M)); % vectorize unmixing matrix for compatibility with Matlab's optimization toolbox

