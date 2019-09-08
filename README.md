# combat-sklearn



 in scikit-learn compatible format
 
 
 
 
 ComBat for correcting batch effects using the Scikit-learn format

Check https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/preprocessing/label.py


#' Adjust for batch effects using an empirical Bayes framework
#'
#' ComBat allows users to adjust for batch effects in datasets where the batch covariate is known, using methodology
#' described in Johnson et al. 2007. It uses either parametric or non-parametric empirical Bayes frameworks for adjusting data for
#' batch effects.  Users are returned an expression matrix that has been corrected for batch effects. The input
#' data are assumed to be cleaned and normalized before batch effect removal. 