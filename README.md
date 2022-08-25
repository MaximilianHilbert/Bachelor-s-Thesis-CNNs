# Bachelor Thesis - Fast Scattering Data Analysis Using Convolutional Neural Networks - Maximilian Hilbert
Abstract
In recent years, the established methods of X-ray reflectometry have been systematically complemented by automated data analysis, which requires little to no prior knowledge of the nature of the film parameters. Their ability to analyze data in real-time makes machine learning methods superior to classical, recursive and thus slow methods.

In the present work, a comparison was made between the existing MLP model [1] and an alternative CNN approach. For this purpose, new network architecture and adapted data processing were established and model-specific parameters were optimized. The network was trained on X-ray reflectometry data, simulated with the mlreflect [2] package and evaluated on simulated and experimentally recorded data.
In a 'proof of concept', the two models were first compared with respect to a (pseudo) test data set enriched with statistical errors. Three material-specific parameters film thickness, surface roughness and SLD have been determined.

The comparison of both models on experimental (in situ) data shows that the parameters of film thickness and film SLD can be predicted by the new approach with comparable uncertainties in the median and standard deviation. However, with respect to surface roughness, the well-known MLP approach shows better results, both in the median and the corresponding standard deviation.

[1] Alessandro Greco, Vladimir Starostin, Evelyn Edel, Valentin Munteanu, Na-
    dine Rußegger, Ingrid Dax, Chen Shen, Florian Bertram, Alexander Hinder-
    hofer, Alexander Gerlach, and Frank Schreiber. Neural network analysis of
    neutron and x-ray reflectivity data: automated analysis using , experimental
    errors and feature engineering. J Appl Crystallogr, 55(2), 2022.

[2] https://pypi.org/project/mlreflect/
