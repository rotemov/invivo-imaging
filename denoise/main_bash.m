function [] = main_bash(data_dir, file_name)
    % Initialization

    GUI = 0;
    data_dir
    file_name
    genpath(fullfile(cd,'..','lib'))
    addpath(genpath(fullfile(cd,'..','lib')));

    harvard_cannon = 0;
    % home = fullfile(cd, data_dir);
    % home = fullfile(cd,'..','demo_data');
    home = data_dir;

    output = fullfile(home,'output');
    plots = fullfile(home, 'plots');

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
    [nrows, ncols, nframes] = size(mov);
    movReg = NoRMCorre2(mov,home); % get registered movie
    disp("NoRMCorre done, saving outputs")
    clear mov
    saveastiff(movReg,fullfile(home,'movReg.tif')); % save registered movie
    clear movReg

    pack

    % extract motion traces into MAT file
    reg_shifts = returnShifts(home);
    save(fullfile(home,'reg_shifts.mat'),'reg_shifts');
    disp("Saving done")
end