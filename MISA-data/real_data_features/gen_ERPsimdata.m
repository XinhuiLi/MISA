ref_folder = '\\fm1-dm2\mialab\users\dbridwell\fit_example\DATA\erp_fmri\ANALYSIS';
ref_fname = 'AodfMRIEEG_ica_comb_2.mat';
load(fullfile(ref_folder,ref_fname), 'icasig')

dt = diag(sqrt(diag(icasig*icasig')));
erps = dt\icasig;
cr = erps*erps';

% View signals:
figure
for kk = 1:8
    subplot(2,4,kk);
    plot(erps(kk,:));
    set(gca,'ylim',[-.1 .2])
end

figure
imagesc(abs(cr - eye(8)),[0 .5])
colormap hot

touse = [8 3 2 1];

% Gram-Schimdt: makes signals orthogonal to each other
out = myGS(erps(touse,:)')'; % the numbers indicate the order in which vectors are taken in GS

% View signals:
figure
for kk = 1:length(touse)
    subplot(2,4,touse(kk));
    plot(out(kk,:));
    set(gca,'ylim',[-.1 .2])
end

ERP = out';

save('.\ERP\orthofeat_ERP.mat', 'ERP')