%% Case 3: 2 Datasets, 4 IVA-type SCVs, Low corr, var = (d+1)
%% Initialize
rng(1981)
tag = 'IVA';
cas = num2str(12);
myfolder = '\\loki\export\mialab\users\rsilva';

%% Generate the SCVs
N = 32968; % Number of samples
C = 16;
r = 2*10.^(-[Inf 2 3 4 6 9]); %Should have been [Inf 9 6 4 3 2]
% figure
% hold on
% for rr = r
%     plot(rr.^linspace(0,1,10))
%     plot(rr.^linspace(0,1,10), '.r')
% end
for rr = r
    %% SCV 1: 2D MvLap, zero-mean, corr
    % Desired parameters:
    d = 10;              % SCV dimensionality
    mu = zeros(1,d);    % SCV mean
    R = toeplitz(rr.^linspace(0,1,d));   % SCV correlation
    
    % Find Cov matrix S that has det(S) = 1, and underlying R:
    %fun = @(a) det(a*eye(size(R))*R*(a*eye(size(R)))) - 1;
    %a = fzero(fun,1);
    %S = a*eye(size(R))*R*(a*eye(size(R)));
    S = R;
    
    % check if R from S is correct:
    %R_ = diag(sqrt(diag(S)))\S/diag(sqrt(diag(S)));
    
    lmb = .5;
    
    subspace = cell(1, C);
    for cc = 1:C
        subspace{cc} = mymvlap(mu,S,N,1/lmb);
    end
    
    %% Assign SCVs to datasets
    M = 1:d;
    Sgt = cell(1,max(M));
    for ff = M
        Sgt{M(ff)} = [];
        for cc = 1:C
            Sgt{M(ff)} = [Sgt{M(ff)}; subspace{cc}(:,ff)'];
        end
    end
    
    %% Define subspace composition
    S = cell(1,max(M));
    for ff = M
        S{ff} = eye(C);
    end
    
    %% Save SCVs in .mat file
    % Go to data folder
    curfolder = pwd; % save current folder
    cd([myfolder '\projects\MultivariateICA\MISA\fresh\data\Sgt\case' cas])
    
    % Find out this rep number
    t = 1;
    while exist(['SCV_' tag '_case' cas '_r' num2str(t,'%03d') '.mat'],'file')
        t = t + 1;
    end
    save(['SCV_' tag '_case' cas '_r' num2str(t,'%03d') '.mat'],'M', 'S', 'Sgt')
    
    % Return to initial folder
    cd(curfolder)
end