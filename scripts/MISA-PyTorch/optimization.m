close all; clear; clc;
% @gsd: generate simulated data
% @gsm: generate simulated mixing matrix

addpath("/Users/xli77/Documents/MISA/scripts");
addpath("/Users/xli77/Documents/MISA/scripts/toy_example/");
addpath(genpath('/Users/xli77/Documents/gift/GroupICATv4.0c'));

outpath = '/Users/xli77/Documents/MISA/MISA-data/torch';
seed_list = [7, 14, 21];
sample_list = [64, 256, 1024, 4096, 16384, 32768];
dataset_list = [2, 12, 32, 100];
source_list = [12, 32, 100];
count=1;

%%
for seed = seed_list
    for n_dataset = dataset_list
        for n_source = source_list
            if (n_dataset==32&&n_source==100) || (n_dataset==100 && n_source==32) || (n_dataset==100 && n_source==100)
                continue
            end
            for n_sample = sample_list
                if n_source > n_sample
                    continue
                end
                if count > 151
                    % error at count=47,99,151 complex value error; stop at 83
                    disp([count, seed, n_sample, n_dataset, n_source]);

                    fn = sprintf('sim-siva_dataset%d_source%d_sample%d_seed%d.mat',n_dataset,n_source,n_sample,seed);
                    load(fullfile(outpath,fn));

                    [isi, aux] = run_optimization(X,wpca,S,M,A);
                    fn = sprintf('out-siva_dataset%d_source%d_sample%d_seed%d_wpca.mat',n_dataset,n_source,n_sample,seed);
                    save(fullfile(outpath,fn),'isi','aux');

                    [isi0, aux0] = run_optimization(X,w0,S,M,A);
                    fn0 = sprintf('out-siva_dataset%d_source%d_sample%d_seed%d_w0.mat',n_dataset,n_source,n_sample,seed);
                    save(fullfile(outpath,fn0),'isi0','aux0');

                    [isi1, aux1] = run_optimization(X,w1,S,M,A);
                    fn1 = sprintf('out-siva_dataset%d_source%d_sample%d_seed%d_w1.mat',n_dataset,n_source,n_sample,seed);
                    save(fullfile(outpath,fn1),'isi1','aux1');

                end
                count=count+1;
            end
        end
    end
end