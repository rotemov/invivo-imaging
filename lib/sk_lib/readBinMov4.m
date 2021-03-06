%load a Bin movie using the info file to get the size.
%No default Bin file name, imput the BinName
function [mov, nrow, ncol]=readBinMov4(BinDir, BinName, flipflg)
tic
clc;
disp(['Loading movie from ' BinDir '...']);

% old file format
if isfile(fullfile(BinDir,'experimental_parameters.txt'))
    try
        fid1 = fopen(fullfile(BinDir,'experimental_parameters.txt'));
        Info=textscan(fid1,'%s');
        nrow = str2num(Info{1,1}{6,1});
        ncol = str2num(Info{1,1}{3,1});
        fclose(fid1);
    catch me
        fclose(fid1);
        rethrow(me);
    end
% read from new file
elseif isfile(fullfile(BinDir,'camera-parameters-Flash.txt'))
    try
        fid1 = fopen(fullfile(BinDir,'camera-parameters-Flash.txt'));
        Info=textscan(fid1,'%s');
        ncol = str2num(Info{1,1}{7,1});
        nrow = str2num(Info{1,1}{8,1});
        fclose(fid1);
    catch me
        fclose(fid1);
        rethrow(me);
    end
% if not given a proper file
else
    ME = MException('Unsupported camera parameter file:noSuchFile', 'Only camera-parameters-Flash.txt or experimental_parameters.txt supported.');
    throw(ME)
end

try
    datfile = fullfile(BinDir,[BinName '.bin']);
    % read file into tmp vector
    fid=fopen(datfile);
    % open file
    tmp = fread(fid, '*uint16', 'l');       % uint16, little endian
    fclose(fid);                            % close file
catch me
    fclose(fid);
    rethrow(me);
end
% reshape vector into appropriately oriented, 3D array
nframe = length(tmp)/(nrow*ncol);

%mov = permute(mov, [2 1 3]);
if flipflg
    c
else
    mov = reshape(tmp, [ncol nrow nframe]);
    mov = permute(mov, [2 1 3]);
end

toc
end