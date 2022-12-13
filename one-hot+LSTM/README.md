DeepGenGrep: a general deep learning-based predictor for multiple genomic signals and regions


INSTALLATION
To run DeepGenGrep:
First install scikit-learn (http://scikit-learn.org/), tensorflow-gpu (https://pypi.org/project/tensorflow-gpu/) 


EXAMPLE:
All datasets employed are included in this project. These datasets are collected from the following literature.

Kalkatawi, M., et al. DeepGSR: an optimized deep-learning structure for the recognition of genomic signals and regions. Bioinformatics 2019;35(7):1125-1132. 

Train DeepGenGrep on human genome for TIS recognition as follows:

python DeepGenGrep.py -gsr TIS -org hs -len 600

Train DeepGenGrep on human genome for the recognition of PAS AATAAA signal as follows:

python DeepGenGrep.py -gsr PAS_AATAAA -org hs -len 600

Parameters and description are as follows:
--gsr: genome signal region (TIS, PAS, PAS_AATAAA, Splice_acc, Splice_don)
-org: organism name, hs (human), mm (mouse), bt (bovine), dm (Drosophila melanogaster)
-len: the length of input sequence

The input dataset was first randomly split into the benchmark (75%) and independent test (25%) datasets. Then, the benchmark dataset was further divided into the training and validation datasets with a ratio of 8:2.

Trained model is saved in the ‘Model’ fold, and the evaluated results is saved in ‘Results’ fold 


CONTACTS


If you have any queries, please email to liuqzhong@nwsuafe.du.cn
