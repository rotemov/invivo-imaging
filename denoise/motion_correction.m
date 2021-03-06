function []=motion_correction(home, output)
    moco_home = home

    addpath(genpath(fullfile(cd,'..','lib')));

    if exist(fullfile(home,'reg_shifts.mat'),'file')
        load(fullfile(home,'reg_shifts.mat'));
        xShifts = reg_shifts(1,71:end);
        yShifts = reg_shifts(2,71:end);
        dX = xShifts - mean(xShifts);
        dY = yShifts - mean(yShifts);
        dXhp = dX - smooth(dX, 2000)';  % high pass filter
        dYhp = dY - smooth(dY, 2000)';
        dXs = smooth(dXhp, 5)';  % low-pass filter just to remove some jitter in the tracking.  Not sure if necessary
        dYs = smooth(dYhp, 5)';

        tStart = tic;
        while(~exist(fullfile(output,'PMD_residual.tif'),'file'))
            pause(30);
            if(toc(tStart) > 5 * 60 * 60)
                display(sprintf('Timed out after %.3f s.', toc(tStart)));
                exit;
            end
        end

        tic;
        mov = shiftdim(loadtiff(fullfile(output,'denoised.tif')),2);
        [ySize, xSize, nFrames] = size(mov);
        t = 1:nFrames;

        avgImg = mean(mov,3);
        dmov = mov - avgImg;

        dT = 5000;
        % First column is the start of each epoch, second column is the end
        if dT ~= nFrames; bdry = [(1:dT:nFrames)', [(dT:dT:nFrames) nFrames]'];
        else; bdry = [(1:dT:nFrames)', [nFrames]']; end;
        nepoch = size(bdry, 1);
        out4 = zeros(size(mov));
        for j = 1:nepoch;
            tau = bdry(j,1):bdry(j,2);
            [out4(:,:,tau), ~] = SeeResiduals(dmov(:,:,tau), [dXs(tau); dYs(tau); dXs(tau).^2; dYs(tau).^2; dXs(tau) .* dYs(tau)], 1);
        end;

        saveastiff(single(out4),fullfile(output,'motion_corrected.tif'));
        toc;
    end
end