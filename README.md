# invivo-imaging (Adam Lab HUJI)

Welcome to Yoav Adam's Lab @ HUJI - subversion of the invivo-imaging pipeline, for running the image processing pipeline on the ELSC cluster.

The original pipeline was developed in the Cohen Lab @ Harvard https://github.com/adamcohenlab/invivo-imaging (click here for the original docs).

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

    f. (recommended) Map the adam-lab network drive following this tutorial: https://support.microsoft.com/en-us/help/4026635/windows-map-a-network-drive (you will need to mail Nizar to get a user).
    
2. Install Anaconda from https://www.anaconda.com/products/individual (64-bit).

3. Install python 3.6 from https://www.python.org/downloads/ (64-bit).

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

* For running remotely with access to the network drive I found the best way is to simply use TeamViewer to run from a Windows 10 computer in the lab.

## Parameter tuning

1. In general the denoising steps (NoRMCoRRe, detrending and motion correction) will work well with the default parameters, if for some reason it fails choose a different parameter window (denoise start frame and number of frames).

2. Demixing will require some iterations. Start with the default parameters and see which cells the algorithm detects.

    (To be continued)
