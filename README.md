# invivo-imaging (Adam Lab HUJI)

Welcome to Yoav Adam's Lab @ HUJI - subversion of the invivo-imaging pipeline, for running the image processing pipeline on the ELSC cluster.

The original pipeline was developed in the Cohen Lab @ Harvard https://github.com/adamcohenlab/invivo-imaging (click here for the original docs).

Pipeline was tested on Ubuntu and Windows 10 machines connected to HUJI's network / using Samba VPN https://ca.huji.ac.il/vpn.

Feel free to open issues in the designated area in this page or to contact me directly: rotem.ovadia@mail.huji.ac.il.

At the moment we don't have a guide for cluster installation as it is a tedious process. For GUI users (anybody 
planning only to use the pipeline and not develop it) please follow the following installation steps to start.

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
 
Great, now you should be all set to run the pipeline!
       
## Running the pipeline

1. Open an Anaconda3 prompt and run the commands:

        conda activate invivo-gui
        cd <directory where you installed the pipeline>\invivo-imaging\GUI
        python gui_main.py
    
2. Choose the relevant files and tune the parameters according to the next section, when you are done click "Run".

* For running remotely with access to the network drive I found the best way is to simply use TeamViewer to run the GUI 
on a Windows 10 computer in the lab.

## Parameter Tuning

Before we jump into the work flow it's good to have a basic understanding of  some of the jargon and what the 
important ones stand for.

###  Important parameters (Not complete)

#### GUI
Graphical user interface, or in simpler terms the windows you interact with as a user.

#### Denoising 
Taking the raw video and making it easier to work with. As we all know an experimental apparatus is never
ideal. In the case of invivo voltage imaging using optogenetics the main problems are registration of the movement
(tiny movements of the mouse or optical machinery), photobleaching (the fluorophore not being activated properly), 
Z-axis movements (AKA defocusing) and the fact that we are sampling a continuous process discretely. The denoising step 
is the pipeline's way to partially fix the artifacts from these problems.

##### NoRMCorre
Registrates the video (after this runs most of the cells mostly stay in place)

##### Detrending
Decreases loss of signal due to photobleaching, discrete sampling and defocusing (using PMD and spline 
detrending, don't worry you don't have to understand what this means).

##### Motion correction
Decreases residual motion artifacts remaining after NoRMCorre and detrending.

##### denoise start frame, number of frames
Some of the steps in denoising don't require the full video. Moreover that they
are not very tractable (even for the cluster) thus we use only part of the video to do this step. It is always recommended
to use a representing part of the video (frames where all of the cells of interest activate), but it not neccessary. 
For large FOVs (more than 800 total pixels per frame) it is recommended to keep the number of frames under 10k and usually 
5k is enough.

### Demixing
Taking the denoised video, identifying the cells and outputing the voltage traces of their activity. In general the demixing stage 
is much less robust than the denoising stage. It will require some iterations to get the parameters right. 

#### demix start frame, number of frames
It is best to demix on the whole video. However some videos are extremely large which means it can take a long time to demix them.
If you don't demix on all the frames it is essential to choose a representing part of the video (frames where all of the cells of interest 
activate). In general any video you demix on half (or more) of the frames from should work fine.

#### superpixels
These are the algorithm's predictions of the regions of interest (ROIs), which means they should have the same shape as the cells.

#### cell diameter
How big the cell is (in pixels in the video) across its longest axis.

#### cutoff point
Each pixel in a super pixel get a grade of how "cell-like" it is, decided by how strongly it activates 
when it's neighbors activate. This means we would expect the central pixels of a ROI to have a higher grade. In order
to identify the cell's borders we give a minimal grade for a pixel to have in order for it to be part of the cell.
These grades also affect how the voltage traces are calculated from each cell.

#### correlation threshold fix
(TODO)



### Workflow

1. In general the denoising steps (NoRMCoRRe, detrending and motion correction) will work well with the default 
parameters, if for some reason it fails choose a different parameter window (denoise start frame and number of frames).
This means that most likely you will check the NoRMCorre, Detrending and Motion correction checkboxes in the main runner
only on your first run.

2. Demixing will require some iterations. If you have a video you already analysed with a similar video 
(same number of cells / microscope parameters / lighting / experimental apparatus) then start from the parameters that 
worked for that video and modify from there. Otherwise start with the default parameters.

3. After each run see which cells the algorithm detects in the superpixels and NMF traces tab. You want the super pixels to
resemble the cells as much as possible and 

3. Increasing the parameters: correlation threshold fix, cutoff point will make the cells smaller / remove unwanted 
backgrounds.

4. Increasing cell diameter will make the identified cells larger (if you see the pipline is missing some of the larger
cells increase this).

5. The logs are meant mostly for the developers, however if you know how tracebacks work don't hesitate to use them to
to figure out what went wrong. If you are experiencing issues you can always email rotem.ovadia@mail.huji.ac.il or open
an issue on github with the job number and I will try to solve it.

(To be continued ...)