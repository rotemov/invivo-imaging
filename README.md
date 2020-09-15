# invivo-imaging (Adam Lab HUJI)

Welcome to Yoav Adam's Lab @ HUJI - subversion of the invivo-imaging pipeline, for running the image processing pipeline on the ELSC cluster.

Pipeline was tested on Ubuntu and Windows 10 machines connected to HUJI's network / using Samba VPN https://ca.huji.ac.il/vpn.

Feel free to open issues in the designated area in this page or to contact me directly: rotem.ovadia@mail.huji.ac.il.

## Installation

1. For windows users:

    a. Enable WSL by opening the Windows PowerShell as administrator and running:
    
        dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
    
    b. Reboot the computer and allow updates        
            
    c. Install Ubuntu 18.04 LTS from the Microsoft Store (don't worry this won't change your OS).
    
    d. Open Ubuntu 18.04 LTS and follow the instructions in the terminal to create a user.
    
    e. In the terminal run the commands:
    
        sudo apt update
        sudo apt install sshpass

2. Install Anaconda from https://www.anaconda.com/products/individual (preferably 64-bit).

3. Install python 3.6 from https://www.python.org/downloads/

4.  a. Click Code --> Download ZIP (at the top of this page)
  
    b. Unzip the pipeline where you want it (if you don't have an unzipper use https://www.7-zip.org/)
  
5. Open an Anaconda3 terminal:
    
    a. Run the command:
    
        cd <directory where you installed the pipeline>\invivo-imaging\GUI
        
    For example I installed on my desktop and ran:
    
        cd  C:\Users\yoavalab.user\Desktop\invivo-imaging\GUI
        
    b. Run the command:
    
       conda env create -f invivo-gui.yml
       
## Running the pipeline

1. Open an Anaconda3 prompt and run the commands:

        conda activate invivo-gui
        cd <directory where you installed the pipeline>\invivo-imaging\GUI
        python gui_main.py
    
2. Choose the relevant files and tune the parameters according to the next section, when you are done click "Run".

## Parameter tuning



# Original Docs

Spike-Guided Penalized Matrix Decomposition-Non-negative Matrix Factorization (SGPMD-NMF) pipeline code for in vivo voltage imaging

1.	NoRMCorre correction of x-y motion
2.	Photobleach correction with b-spline fit
3.	PMD denoiser with parameters optimized on simulated data
4.	From each pixel, regress out residual motion artifacts using NoRMCorre-estimated motion trajectories   
5.	Manually crop blood vessels
6.	localNMF demixing on temporally high-pass filtered movie to obtain spatial support of cells
7.  fastHALS iterations on full unfiltered movie to calculate full spatial footprints of cells and background
8.	Update spatial footprint of cells and background by ensuring smoothness of background spatial components around edges of cells
9.	Apply the updated spatial and background footprints to calculate the temporal traces from the full denoised movie

## Instructions for cluster install

Instructions to run the pipeline here: http://bit.ly/sgpmdnmf-steps (Not comprehensive)

Detailed instructions for setup on Harvard's Cannon cluster: http://bit.ly/harvard-sgpmdnmf-steps (Semi comprehensive)

## Dependencies

* [TREFIDE](http://github.com/ikinsella/trefide)

## References

[1] Xie, M., Adam, Y., Fan, L., Böhm, U., Kinsella, I., Zhou, D., Paninski, L., Cohen, A. High fidelity estimates of spikes and subthreshold waveforms from 1-photon voltage imaging in vivo. Submitted (2020)
