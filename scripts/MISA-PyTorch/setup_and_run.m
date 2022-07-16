close all; clear; clc;

%%
addpath('./scripts/MISA-PyTorch/');

%% Define subspace structure for the sources (the following matches what was used to generate the sources)
X = cell(1,3);
X{1} = rand(3,3);
X{2} = rand(3,3);
X{3} = rand(3,3);

% Define the number of datasets (here, the number of modalities)
M = 1:length(X);

S = cell(1,3);           % Cell array: each cell contains a matrix K x C(m).
                         % Each k-th row has 0's and 1's to indicate what
                         % source go within the k-th subspace in dataset m

% Modality 1 = dataset 1
S{1} = [1 0 0;... % source 1 into subspace 1
        0 1 0;... % source 2 into subspace 2
        0 0 1];   % source 3 into subspace 3

% Modality 2 = dataset 2
S{2} = [1 0 0;... % source 1 into subspace 1
        0 1 0;... % source 2 into subspace 2
        0 0 1];   % source 3 into subspace 3

% Modality 3 = dataset 3
S{3} = [1 0 0;... % source 1 into subspace 1
        0 1 0;... % source 2 into subspace 2
        0 0 1];   % source 3 into subspace 3

get_MISA_Torch_parameters

%% Initialize MISA object
data1 = MISAK(w0, M, S, X, ...
                beta, eta, lambda, ...
                gradtype, sc, preX);

%% Save variables
save './scripts/MISA-PyTorch/X.mat', 'X';
save './scripts/MISA-PyTorch/W0.mat', 'W0';
