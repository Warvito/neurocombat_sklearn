# NeuroCombat-sklearn
[![License: MIT](https://img.shields.io/github/license/Warvito/neurocombat_sklearn)](https://opensource.org/licenses/MIT) 
[![Version](https://img.shields.io/pypi/v/neurocombat-sklearn)](https://pypi.org/project/neurocombat-sklearn/)
[![PythonVersion](https://img.shields.io/pypi/pyversions/neurocombat-sklearn)]()

Implementation of Combat harmonization method in scikit-learn compatible format.


The Combat harmonization/normalization method uses an parametric empirical Bayes framework to robustly adjust data for site/batch effects. 
The scikit-learn compatible format was used to facilitates the use of this harmonization method in machine learning projects. 


This repository is developed by [Walter Hugo Lopez Pinaya](https://scholar.google.com/citations?user=jjT5-HUAAAAJ) at King's College London and community contributors.

## Installation

### Requirements
- Python (>= 3.5)
- [Scikit-Learn](https://scikit-learn.org/) (>= 0.21.0)


### User installation

If you already have a working installation of numpy and scipy,
the easiest way to install neurocombat-sklearn is using ``pip``   :

    pip install neurocombat-sklearn
 

## Citation
If you find this code useful for your research, please cite:

    @article{fortin2018harmonization,
      title={Harmonization of cortical thickness measurements across scanners and sites},
      author={Fortin, Jean-Philippe and Cullen, Nicholas and Sheline, Yvette I and Taylor, Warren D and Aselcioglu, Irem and Cook, Philip A and Adams, Phil and Cooper, Crystal and Fava, Maurizio and McGrath, Patrick J and others},
      journal={Neuroimage},
      volume={167},
      pages={104--120},
      year={2018},
      publisher={Elsevier}
    }
    
    @article{johnson2007adjusting,
      title={Adjusting batch effects in microarray expression data using empirical Bayes methods},
      author={Johnson, W Evan and Li, Cheng and Rabinovic, Ariel},
      journal={Biostatistics},
      volume={8},
      number={1},
      pages={118--127},
      year={2007},
      publisher={Oxford University Press}
    }

### Disclaimer

Based on:
 - https://github.com/ncullen93/neuroCombat
 - https://github.com/nih-fmrif/nielson_abcd_2018
 - https://github.com/Jfortin1/ComBatHarmonization

