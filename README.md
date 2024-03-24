## FICT - FISH Iterative Cell Typing  
### is a package facilitates accurate cell clustering by iteratively incorporating neighborhood information using the Expectation-Maximization algorithm, designed for data generated through the Fluorescence in situ Hybridization (FISH) technique

![Method](https://github.com/haotianteng/FICT/assets/11155295/3dbef51f-46d1-4af0-b518-864f67163af6)

### Installation
```bash
git clone https://github.com/haotianteng/FICT.git
cd FICT
conda activate YOUR_VIR_ENVIRONMENT
python setup.py install
fict --help
```

### Prepare dataset for running
Currently different FISH project using different data format, so 4 scripts are provided for dataset preparation for different datasets in FICT-SAMPLE tutorial repository.  

### Sample code
Please check [FICT-SAMPLE](https://github.com/haotianteng/FICT-SAMPLE) for sample code to run FICT.  

### Citing FICT
If you used FICT in your work, please cite:
> Teng, H., Yuan, Y., & Bar-Joseph, Z. (2021). Clustering Spatial Transcriptomics Data. Bioinformatics.

### Issue report and contact
If you find any problem during using FICT, please don't hesitate to submit an issue through Github, or directly contact us through the email haotiant@andrew.cmu.edu
