msk=spm_read_vols(spm_vol('\\fm1-dm2\mialab\hcp\groupica\rest\HC_603_S100_G75\rest_hcpMask.img'));
load('\\fm1-dm2\mialab\hcp\groupica\rest\HC_603_S100_G75\rest_hcp_ica.mat')

load('\\fm1-dm2\mialab\users\rsilva\code\display\mycolors.mat')

sz = size(msk);
nmsk = false(sz);
for ss = 1:length(sz)
    if ss == length(sz), id{ss} = (1:8:sz(ss))';
    else id{ss} = (1:2:sz(ss))'; end
    dim(ss) = length(id{ss});
end
nmsk(id{:}) = true(dim);
sampmsk = nmsk(logical(msk));
ricasig = icasig(:,sampmsk);
rmsk = reshape(msk(nmsk),dim);

% Lower res
a = zeros(size(rmsk));
a(logical(rmsk)) = ricasig(53,:);
figure
imagesc(rot90(a(:,:,4)',2), max(abs(a(:)))*[-1 1])
colormap(mycoolhot)
axis equal tight

% High res: for comparison
a = zeros(size(msk));
a(logical(msk)) = icasig(53,:);
figure
imagesc(rot90(a(:,:,23)',2), max(abs(a(:)))*[-1 1])
colormap(mycoolhot)
axis equal tight

dt = diag(sqrt(diag(ricasig*ricasig')));
fmri = dt\ricasig;
cr = fmri*fmri';

figure
imagesc(abs(cr - eye(75)),[0 .8])
colormap hot

%touse = unique([21 27 53 46 64 45 72 50 59 29 24 23 56 29 56 72 55 38 68 55 25 49 58]);
touse = [21 23 24 25 27 29 38 45 46 49 50 53 55 56 58 59 64 68 72];
cr = fmri(touse,:)*fmri(touse,:)';

figure
imagesc(abs(cr - eye(19)),[0 .22])
colormap hot

% Select subset of maps with low corr.
xx = abs(cr-eye(19));

keep = 1:19;
M = length(keep);
while M > 7
    [t ixx] = max(xx(:));
    [a,b] = ind2sub([19 19],ixx);
    keep = setdiff(keep,[a b]);
    M = length(keep);
    xx([a b],:) = zeros(2,19);
    xx(:,[a b]) = zeros(19,2);
end

cr = fmri(touse(keep),:)*fmri(touse(keep),:)';
figure
imagesc(abs(cr - eye(7)),[0 .06])
colormap hot

touse = touse(keep([1 2 4 5 6 7]));

figure
imagesc(abs(fmri(touse,:)*fmri(touse,:)' - eye(6)),[0 .06])
colormap hot

% View signals:
A = [];
for kk = [touse([3 1 6 5 2 4])]
    a = zeros(size(rmsk));
    a(logical(rmsk)) = ricasig(kk,:);
    A = [A rot90(a(:,:,4)',2)];
end
figure
imagesc(A,max(abs(A(:)))*[-1 1])
colormap(mycoolhot)
axis equal tight

% Gram-Schimdt: makes signals orthogonal to each other
out = myGS(ricasig(touse([3 1 6 5 2 4]),:)')'; % the numbers indicate the order in which vectors are taken in GS

A = [];
for kk = 1:6
    a = zeros(size(rmsk));
    a(logical(rmsk)) = out(kk,:);
    A = [A rot90(a(:,:,4)',2)];
end
figure
imagesc(A,max(abs(A(:)))*[-1 1])
colormap(mycoolhot)
axis equal tight

fMRI = out';

save('.\fMRI\orthofeat_fMRI_r.mat', 'fMRI')
% copyfile('\\fm1-dm2\mialab\hcp\groupica\rest\HC_603_S100_G75\rest_hcpMask.img', ...
%         '\\fm1-dm2\mialab\users\rsilva\projects\MultivariateICA\MISA\fMRI\rest_hcpMask.img')
% copyfile('\\fm1-dm2\mialab\hcp\groupica\rest\HC_603_S100_G75\rest_hcpMask.hdr', ... 
%         '\\fm1-dm2\mialab\users\rsilva\projects\MultivariateICA\MISA\fMRI\rest_hcpMask.hdr')