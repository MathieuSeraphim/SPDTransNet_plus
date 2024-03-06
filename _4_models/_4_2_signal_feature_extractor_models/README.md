The only extractor model presented in this paper is ```MultichannelIITNetFeatureExtractorModel```.

Other attempts, including ```MultichannelIITNetFeatureExtractorModel_v2``` (uses CNNs to parallelize computations that v1 undergoes sequentially, so both slightly faster and theoretically identical than v1), underperform in comparison.

These are left here to avoid breaking something.