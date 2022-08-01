function [whtM, H] = doMMGPCA(X, comps, rec_type)

M = 1:length(X);
N = size(X{1},2);

V = cellfun(@(x) size(x,1), X);

outs = cell(size(X));

cvx = zeros(N);

for mm = M
    cvx_ = cov(X{mm});
    cvx = cvx + cvx_./(length(M)*trace(cvx_)/N);
end

% Subject-level PCA reduction...
[H,lambda] = eigs(cvx,comps);

A = cellfun(@(x) sqrt(N./(length(M)*sum(x(:).^2)))*(x*H), X, 'Un', 0);
norm_A = cellfun(@(a) sum(a.^2), A, 'Un', 0)';
norm_A = sqrt(sum(cell2mat(norm_A)));
A = cellfun(@(a) a./repmat(norm_A,size(a,1),1),A,'Un',0);

if strcmpi(rec_type, 'WT')
    whtM = cellfun(@(a) a', A, 'Un', 0);
elseif strcmpi(rec_type, 'PINV')
    whtM = cellfun(@(a) pinv(a), A, 'Un', 0);
elseif strcmpi(rec_type, 'REG')
    % To Do...
end

whtM = cellfun(@(x,w) sqrt(N-1) * sqrt(N./(length(M)*sum(x(:).^2))) * repmat(1./(norm_A'),1,size(w,2)) .* w, X, whtM, 'Un', 0);

H = sqrt(N-1) * H';