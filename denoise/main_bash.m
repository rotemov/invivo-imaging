function [] = main_bash(home, file_name, output)
    % Initialization

    GUI = 0;
    harvard_cannon = 0;
    genpath(fullfile(cd,'..','lib'))
    addpath(genpath(fullfile(cd,'..','lib')));
    plots = fullfile(output, 'plots');

    if ~exist(output,'dir')
        mkdir(output)
        disp('Made output directory')
    end

    if ~exist(plots,'dir')
        mkdir(plots)
        disp('Made plots directory')
    end

    % NoRMCorre image registration

    [fp,n,ext] = fileparts(file_name)
    if strcmp(ext,'.tif')
        mov = loadtiff(fullfile(home, file_name));
    elseif strcmp(ext,'.bin')
        [mov, nr, nc] = readBinMov4(home, n, 0);
    else
        disp("Unsupported format, terminating")
        exit
    end

    [nrows, ncols, nframes] = size(mov)

    movReg = NoRMCorre2(mov,output); % get registered movie
    disp("NoRMCorre done, saving outputs")
    clear mov
    saveastiff(movReg,fullfile(output,'movReg.tif')); % save registered movie
    clear movReg

    % extract motion traces into MAT file
    reg_shifts = returnShifts(output);
    save(fullfile(output,'reg_shifts.mat'),'reg_shifts');
    disp("Saving done")
end