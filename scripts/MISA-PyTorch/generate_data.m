function [sim_siva,S,M,A,Y,X,wpca,w0,w1] = generate_data(seed,n_sample,n_dataset,n_source)

N=n_sample; % number of samples
M_Tot=n_dataset; % number of datasets
K=ones(1,n_source); % number of sources
V=sum(K);
Acond=3; % 1 means orthogonal matrix
SNR=(1+999)/1;

sim_siva = sim_basic_SIVA(seed,K,V,M_Tot,N,Acond,SNR);

S = sim_siva.S;
M = sim_siva.M;
A = sim_siva.A;
Y = sim_siva.Y;
X = sim_siva.genX();

ut = utils;
n_pc = sim_siva.C(1); % same as number of sources
[whtM, H] = ut.doMMGPCA(X, n_pc, 'WT');

% Use relative gradient
gradtype = 'relative';

% Enable scale control
sc = 1;

% Turn off preprocessing (still removes the mean of the data)
preX = false;

w0 = ut.stackW({diag(pi/sqrt(3)./std(H,[],2))*eye(size(H,1))});

gica1 = MISAK(w0, 1, {eye(size(H,1))}, {H}, ...
                0.5*ones(n_pc,1), ones(n_pc,1), ones(n_pc,1), ...
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
% W = cellfun(@(w) w,whtM,'Un',0);
W = cellfun(@(w) gica1.W{1}*w,whtM,'Un',0);
wpca = cellfun(@(w,x) diag(pi/sqrt(3) ./ std(w*x,[],2))*w,W,X,'Un',0);

w0 = cell(size(A));
w1 = cell(size(A));
for mm = M
    n_component = size(S{mm},2);
    [u, s, v] = svd(randn(size(A{mm}(:,1:n_component)')),'econ');
    [u1, s1, v1] = svd(randn(size(A{mm}(:,1:n_component)')),'econ');
    w0{mm} = u*v';
    w1{mm} = u1*v1';
end

end