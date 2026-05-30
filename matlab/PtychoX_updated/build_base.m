scriptPath = fileparts(mfilename('fullpath'));
cd(scriptPath);

clc
clear

disp("Building MATLAB extension for Hash Encoding");

tic

nvcc_flags = [...
    '-std=c++17 ',...
    '-allow-unsupported-compiler ' ...
];

setenv("NVCC_APPEND_FLAGS", nvcc_flags)

include_dirs = {'scripts'};
flags = cellfun(@(dir) ['-I"' fullfile(pwd, dir) '"'], ...
                        include_dirs, 'UniformOutput', false);
flags = [flags, {'-lcuda'}, {'-lcudart'}, {'-lcufft'}];

replace_file = {'fwd_adPtychoX_base.cu', 'bwd_adPtychoX_base.cu'};
output_path = '@adPtychoX_base';
fwd_path = 'scripts/forward/';
bwd_path = 'scripts/backward/';

mexcuda(flags{:}, [fwd_path, replace_file{1}], '-outdir', [output_path, '\private']);
mexcuda(flags{:}, [bwd_path, replace_file{2}], '-outdir', [output_path, '\private']);
mexcuda(flags{:}, 'scripts\buffer\init_ptyX_base_buffers.cu', '-outdir', [output_path, '\private']);
mexcuda(flags{:}, 'scripts\buffer\delete_ptyX_base_buffers.cu', '-outdir', [output_path, '\private']);

time_spend = toc;
disp(['compiling takes:', num2str(time_spend), 's'])



