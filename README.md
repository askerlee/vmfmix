# Dirichlet vMF mixture model
The Dirichlet-vMF mixture model implemented here is a simplification of the Bayesian vMF mixture model proposed in [1]. See "vMF mixture.pdf" for detailed model formulation and derivation. 

"corpusLoader.py", "wordclust.py" convert 20 Newsgroups/reuters corpus to word embeddings, and then feed them to the vmf-mixture model to get topic embeddings. "classEval.py" evaluates the quality of obtained topic proportions by using them as features for document classification. Part of the code is borrowed/refactored from my another project https://github.com/askerlee/topicvec. 

[1] Siddharth Gopal and Yiming Yang. Von mises-Fisher Clustering Models. In ICML, pages 154-162, 2014.
