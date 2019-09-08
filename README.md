# NeuroCombat-sklearn

Adjust for batch effects using an empirical Bayes framework

ComBat allows users to adjust for batch effects in datasets where the batch covariate is known,using methodology described in Johnson et al.  2007.  It uses either parametric or non-parametricempirical Bayes frameworks for adjusting data for batch effects.  Users are returned an expressionmatrix that has been corrected for batch effects.   The input data are assumed to be cleaned andnormalized before batch effect removal.

TheComBatfunction adjusts for known batches using an empirical Bayesianframework [1].  In order to use the function, you must have a known batchvariable in your dataset.


 in scikit-learn compatible format



he aim of the standardization procedure presented in section 3.1 is to reduce gene-to-gene variation in the data, because genes in the array are expected to have different expression profiles or distributions. However, we do expect that phenomena that cause batch effects to affect many genes in similar ways. To more clearly extract the common batch biases from the data, the standardization procedure standardizes all genes to have the similar overall mean and variance. On this scale, batch effect estimators can be compared and pooled across genes to create robust estimators for batch effects. Without standardization, the gene-specific variation increases the noise in the data and inflates the prior variance, decreasing the amount of shrinkage that occurs. Therefore standardization is crucial for EB shrinkage methods. However this feature is not present in many EB methods for Affymetrix arrays.  

 empirical Bayes (EB) method that is robust for adjusting for batch effects in data whose batch sizes are small.
 
 
 
 Location and scale (L/S) adjustments can be defined as a wide family of adjustments in which one assumes a model for the location (mean) and/or scale (variance) of the data within batches and then adjusts the batches to meet assumed model specifications. Therefore, L/S batch adjustments assume that the batch effects can be modeled out by standardizing means and variances across batches. These adjustments can range from simple gene-wise mean and variance standardization to complex linear or non-linear adjustments across the genes.
One straightforward L/S batch adjustment is to mean center and standardize the variance of each batch for each gene independently. Such a method is currently implemented in the dChip software (Li and Wong, 2003), designated as “using standardized separators” (see Figure 1(b)). In more complex situations such as unbalanced designs or when incorporating numerical covariates, a more general L/S framework must be used. For example, let Yijg
represent the expression value for gene g for sample j from batch i. Define an L/S model that assumes 
graphic
(2.1)
where αg is the overall gene expression, X is a design matrix for sample conditions, and βg is the vector of regression coefficients corresponding to X. The error terms, εijg⁠, can be assumed to follow a Normal distribution with expected value of zero and variance σ2g⁠. The γig and δig represent the additive and multiplicative batch effects of batch i for gene g, respectively. The batch-adjusted data, Y∗ijg⁠, are given by 
graphic
(2.2)
where αˆg,βˆg,γˆig,andδˆig are estimators for the parameters αg⁠, βg⁠, γig⁠, and δig based on the model.




 
 
 
 
 ComBat for correcting batch effects using the Scikit-learn format

Check https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/preprocessing/label.py


#' Adjust for batch effects using an empirical Bayes framework
#'
#' ComBat allows users to adjust for batch effects in datasets where the batch covariate is known, using methodology
#' described in Johnson et al. 2007. It uses either parametric or non-parametric empirical Bayes frameworks for adjusting data for
#' batch effects.  Users are returned an expression matrix that has been corrected for batch effects. The input
#' data are assumed to be cleaned and normalized before batch effect removal.


ust as withsva, we then need to create a model matrix for the adjustmentvariables, including the variable of interest. Note that you do not include batchin creating this model matrix - it will be included later in theComBatfunction.In this case there are no other adjustment variables so we simply fit an interceptterm.> modcombat = model.matrix(~1, data=pheno)Note that adjustment variables will be treated as given to theComBatfunction.This means if you are trying to adjust for a categorical variable with p differentlevels, you will need to giveComBatp-1 indicator variables for this covariate. Werecommend using themodel.matrixfunction to set these up. For continuousadjustment variables, just give a vector in the containing the covariate valuesin a single column of the model matrix.We now apply theComBatfunction to the data, using parametric empiricalBayesian adjustments.> combat_edata = ComBat(dat=edata, batch=batch, mod=modcombat, par.prior=TRUE, prior.plots=FALSE)Standardizing Data across genesThis returns an expression matrix, with the same dimensions as your originaldataset. This new expression matrix has been adjusted for batch. Significanceanalysis can then be performed directly on the adjusted data using the modelmatrix and null model matrix as described before:> pValuesComBat = f.pvalue(combat_edata,mod,mod0)> qValuesComBat = p.adjust(pValuesComBat,method="BH")These P-values and Q-values now account for the known batch effects includedin the batch variable.There are a few additional options for theComBatfunction.  By default, itperforms parametric empirical Bayesian adjustments. If you would like to usenonparametric empirical Bayesian adjustments, use thepar.prior=FALSEop-tion (this will take longer). Additionally, use theprior.plots=TRUEoption togive prior plots with black as a kernel estimate of the empirical batch effectdensity and red as the parametric estimate. For example, you might chose touse the parametric Bayesian adjustments for your data, but then can check theplots to ensure that the estimates were reasonable.Also, we have now added themean.only=TRUEoption, that only adjusts themean of the batch effects across batches (default adjusts the mean and vari-ance). This option is recommended for cases where milder batch effects areexpected (so no need to adjust the variance), or in cases where the variances are expected to be different across batches due to the biology. For example,suppose a researcher wanted to project a knock-down genomic signature to beprojected into the TCGA data. In this case, the knockdowns samples may bevery similar to each other (low variance) whereas the signature will be at vary-ing levels in the TCGA patient data. Thus the variances may be very differentbetween the two batches (signature perturbation samples vs TCGA), so onlyadjusting the mean of the batch effect across the samples might be desired inthis case.Finally, we have now added aref.batchparameter, which allows users to selectone batch as a reference to which other batches will be adjusted. Specifically,the means and variances of the non-reference batches will be adjusted to makethe mean/variance of the reference batch. This is a useful feature for caseswhere one batch is larger or better quality. In addition, this will be useful inbiomarker situations where the researcher wants to fix the traning set/modeland then adjust test sets to the reference/training batch. This avoids test-setbias in such studies. 




**References**: If you are using ComBat for the harmonization of multi-site imaging data, please cite the following papers:

|       | Citation     | Paper Link
| -------------  | -------------  | -------------  |
| ComBat for multi-site DTI data    | Jean-Philippe Fortin, Drew Parker, Birkan Tunc, Takanori Watanabe, Mark A Elliott, Kosha Ruparel, David R Roalf, Theodore D Satterthwaite, Ruben C Gur, Raquel E Gur, Robert T Schultz, Ragini Verma, Russell T Shinohara. **Harmonization Of Multi-Site Diffusion Tensor Imaging Data**. NeuroImage, 161, 149-170, 2017  |[Link](https://www.sciencedirect.com/science/article/pii/S1053811917306948?via%3Dihub#!)| 
| ComBat for multi-site cortical thickness measurements    | Jean-Philippe Fortin, Nicholas Cullen, Yvette I. Sheline, Warren D. Taylor, Irem Aselcioglu, Philip A. Cook, Phil Adams, Crystal Cooper, Maurizio Fava, Patrick J. McGrath, Melvin McInnis, Mary L. Phillips, Madhukar H. Trivedi, Myrna M. Weissman, Russell T. Shinohara. **Harmonization of cortical thickness measurements across scanners and sites**. NeuroImage, 167, 104-120, 2018  |[Link](https://www.sciencedirect.com/science/article/pii/S105381191730931X)| 
| Original ComBat paper for gene expression array    |  W. Evan Johnson and Cheng Li, **Adjusting batch effects in microarray expression data using empirical Bayes methods**. Biostatistics, 8(1):118-127, 2007.      | [Link](https://academic.oup.com/biostatistics/article/8/1/118/252073/Adjusting-batch-effects-in-microarray-expression) |



https://github.com/nih-fmrif/nielson_abcd_2018