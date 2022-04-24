import numpy as np
import dask.array as da
from sklearn.base import BaseEstimator, TransformerMixin

# Adapted from: https://rdrr.io/bioc/edgeR/src/R/calcNormFactors.R
# Verified using the data in: https://davetang.org/muse/2011/01/24/normalisation-methods-for-dge-data/

class UpperQuartile(BaseEstimator, TransformerMixin):
    """
    This estimator learns a normalization factor from the data's upper quartile q,
    and uses it as a basis for the scaling factor.

    Note that UpperQuartile assumes all samples have nonzero transcripts.
    """

    def __init__(self, q=0.75):
        self.q = q

    def fit(self, X):
        # Remove all transcripts that are 0 across ALL samples, i.e. their per-gene mean is equal to 0
        self.X = X.loc[:, (X.mean(axis=0) > 0.0)]
        self.norm_factor = self._uq(self.X)
        # For symmetry, normalization factors are adjusted to multiply to 1 before being used as scaling factors.
        self.scaling_factor = self.norm_factor / np.exp(np.mean(np.log(self.norm_factor.replace(0, 1).values)))
        return self

    def _uq(self, X):
        return X.apply(lambda sample: sample.quantile(self.q) / sample.sum(), axis=1)

    def transform(self, X):
        return X.multiply(self.scaling_factor, axis=0)


class UpperQuartileDask(BaseEstimator, TransformerMixin):
    """
    This estimator learns a normalization factor from the data's upper quartile q,
    and uses it as a basis for the scaling factor.

    Note that UpperQuartile assumes all samples have nonzero transcripts.
    """

    def __init__(self, q=0.75):
        self.q = q

    def fit(self, X):
        # Remove all transcripts that are 0 across ALL samples, i.e. their per-gene mean is equal to 0
        self.X = X[X.columns[(X.mean(axis=0) > 0.0)]].persist()
        self.norm_factor = self._uq(self.X).persist()
        # For symmetry, normalization factors are adjusted to multiply to 1 before being used as scaling factors.
        self.scaling_factor = self.norm_factor / da.exp(da.mean(da.log(self.norm_factor.replace(0, 1).values)))
        return self

    def _uq(self, X):
        return X.apply(lambda sample: sample.quantile(self.q) / sample.sum(), axis=1,
                       meta=('UQ', 'f8'))

    def transform(self, X):
        return X.mul(self.scaling_factor, axis=0)


class TMM(BaseEstimator, TransformerMixin):
    """
    This estimator learns a  scaling factor from normalizing gene expression based on a pseudoreference sample
    computed from the median of the data's geometric mean.
    """

    def __init__(self, log_ratio_trim=0.3, abs_expr_trim=0.05):
        self.log_ratio_trim = log_ratio_trim
        self.abs_expr_trim = abs_expr_trim

    def fit(self, X):
        # Remove all transcripts that are 0 across ALL samples
        self.X = X.loc[:, (X != 0).any(axis=0)]
        self.pseudoref = self.X.apply(lambda sample: sample[sample > 0].quantile(0.75) / sample.sum(), axis=0)

        self.norm_factor = self._tmm(self.X)
        # For symmetry, normalization factors are adjusted to multiply to 1 before being used as scaling factors.
        self.scaling_factor = self.norm_factor / np.exp(np.mean(np.log(self.norm_factor.replace(0, 1).values)))
        return self

    def _tmm(self, X):
        log_ratio = X.apply(self._log_ratio, axis=1)
        abs_expr = X.apply(lambda sample: self._absolute_expression(sample, self.pseudoref), axis=1)

        log_ratio = log_ratio.replace([np.inf, -np.inf], np.nan)
        abs_expr = abs_expr.replace([np.inf, -np.inf], np.nan)

        trimmed_log_ratio = log_ratio[log_ratio.gt(np.nanquantile(log_ratio.values, self.log_ratio_trim)) &
                                      log_ratio.lt(np.nanquantile(log_ratio.values, 1-self.log_ratio_trim))]

        trimmed_abs_expr = abs_expr[abs_expr.gt(np.nanquantile(abs_expr.values, self.abs_expr_trim)) &
                                    abs_expr.lt(np.nanquantile(abs_expr.values, 1-self.abs_expr_trim))]

        return ((trimmed_log_ratio * trimmed_abs_expr) / trimmed_log_ratio).mean(axis=1)

    def _log_ratio(self, sample):
        sample = sample[sample > 0]
        sample_total_counts = sample.sum()

        return sample.apply(lambda gene: (sample_total_counts - gene)/(sample_total_counts * gene))

    def _absolute_expression(self, sample, ref_sample):
        sample = sample[(sample > 0) | (ref_sample > 0)]
        ref_sample = ref_sample[(sample > 0) | (ref_sample > 0)]

        sample_total_counts = sample.sum()
        ref_sample_total_counts = ref_sample.sum()

        numerator = sample[sample > 0] / sample_total_counts
        denominator = ref_sample[ref_sample > 0] / ref_sample_total_counts

        return (numerator.replace(0, 1).apply(np.log2) / denominator.replace(0, 1).apply(np.log2).replace(0, 1)) / 2

    def transform(self, X):
        return X.multiply(self.scaling_factor, axis=0)