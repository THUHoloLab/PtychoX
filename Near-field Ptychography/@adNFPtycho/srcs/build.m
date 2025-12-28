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

include_dirs = {...
    '', ...
    'addon'};

flags = cellfun(@(dir) ['-I"' fullfile(pwd, dir) '"'], ...
                        include_dirs, 'UniformOutput', false); 
% use the cuda.lib
flags = [flags,{'-lcuda'},{'-lcudart'},{'-lcufft'}]; 


cu_path  = 'srcs/';

% main_file = {'cuPtycho_Fwd.cu','cuPtycho_Bwd.cu'};
replace_file = {'cuPtycho_fft_Fwd.cu','cuPtycho_fft_Bwd.cu'};

[output_path, ~, ~] = fileparts(pwd);

% mexcuda(flags{:}, main_file{1},'-outdir',[output_path,'\private']);
% mexcuda(flags{:}, main_file{2},'-outdir',[output_path,'\private']);

mexcuda(flags{:}, replace_file{1},'-outdir',[output_path,'\private']);
% mexcuda(flags{:}, replace_file{2},'-outdir',[output_path,'\private']);
% 
% mexcuda(flags{:}, replace_file{1});
% mexcuda(flags{:}, replace_file{2});

time_spend = toc;
disp(['compiling takes:',num2str(time_spend),'s'])