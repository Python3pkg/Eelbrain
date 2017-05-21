# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Statistical Parametric Mapping"""

from operator import mul

import numpy as np

from .._colorspaces import stat_info
from .._data_obj import (Dataset, Factor, Var, NDVar, asmodel, asndvar,
                         DimensionMismatchError)
from . import opt
from .stats import lm_betas_se_1d
from .testnd import ttest_1samp
from functools import reduce


class LM(object):
    """Fixed effects model for SPM

    Parameters
    ----------
    y : NDVar
        Dependent variable.
    model : Model
        Model to fit.
    ds : Dataset
        Optional Dataset providing data for y/model.
    coding : 'dummy' | 'effect'
        Model parametrization (default is dummy coding). Vars are centered for
        effect coding (but not for dummy coding).
    subject : str
        Optional information used by RandomLM.
    """
    def __init__(self, y, model, ds=None, coding='dummy', subject=None):
        if subject is not None and not isinstance(subject, str):
            raise TypeError("subject needs to be None or string, got %s"
                            % repr(subject))

        y = asndvar(y, ds=ds)
        n_cases = len(y)
        model = asmodel(model, None, ds, n_cases)
        p = model._parametrize(coding)
        n_coeff = p.x.shape[1]
        coeffs_flat = np.empty((n_coeff, reduce(mul, y.shape[1:])))
        y_flat = y.x.reshape((n_cases, -1))
        opt.lm_betas(y_flat, p.x, p.projector, coeffs_flat)
        se_flat = lm_betas_se_1d(y_flat, coeffs_flat, p)
        self.__setstate__({
            'coding': coding, 'coeffs': coeffs_flat, 'se': se_flat,
            'model': model, 'p': p, 'dims': y.dims[1:], 'subject': subject})

    def __setstate__(self, state):
        self.coding = state['coding']
        self._coeffs_flat = state['coeffs']
        self._se_flat = state['se']
        self.model = state['model']
        self.dims = state['dims']
        self.subject = state['subject']
        if 'p' in state:
            self._p = state['p']
        else:
            self._p = self.model._parametrize(self.coding)
        # secondary attributes
        self._shape = tuple(map(len, self.dims))
        self.column_names = self._p.column_names
        self.n_cases = self.model.df_total

    def __getstate__(self):
        return {'coding': self.coding, 'coeffs': self._coeffs_flat,
                'se': self._se_flat, 'model': self.model, 'dims': self.dims,
                'subject': self.subject}

    def _coefficient(self, term):
        """Regression coefficient for a given term"""
        index = self._index(term)
        return self._coeffs_flat[index].reshape((1,) + self._shape)

    def _index(self, term):
        if term in self.column_names:
            return self.column_names.index(term)
        elif term in self._p.terms:
            index = self._p.terms[term]
            if index.stop - index.start > 1:
                raise NotImplementedError("Term has more than one column")
            return index.start
        else:
            raise KeyError("Unknown term: %s" % repr(term))

    def t(self, term):
        ":class:`NDVar` with t-values for a given term"
        index = self._index(term)
        t = self._coeffs_flat[index] / self._se_flat[index]
        info = stat_info('t', term=term)
        return NDVar(t.reshape(self._shape), self.dims, info)

    def _n_columns(self):
        return {term: s.stop - s.start for term, s in self._p.terms.items()}


class RandomLM(object):
    """Random effects model for SPM"""
    def __init__(self, lms):
        # check lms
        lm0 = lms[0]
        n_columns_by_term = lm0._n_columns()
        for lm in lms[1:]:
            if lm.dims != lm0.dims:
                raise DimensionMismatchError("LMs have incompatible dimensions")
            elif lm._n_columns() != n_columns_by_term:
                raise ValueError("Model for %s and %s don't match" %
                                 (lm0.subject, lm.subject))
            elif lm.coding != lm0.coding:
                raise ValueError("Models have incompatible coding")

        # make sure to have a unique subject label for each lm
        name_i = 0
        subjects = [lm.subject for lm in lms]
        str_names = [_f for _f in subjects if _f]
        if len(set(str_names)) < len(str_names):
            raise ValueError("Duplicate subject names in %s" % str_names)
        new_name = 'S000'
        for i in range(len(subjects)):
            if not subjects[i]:
                while new_name in subjects:
                    name_i += 1
                    new_name = 'S%3i' % name_i
                subjects[i] = new_name

        self.__setstate__({'lms': lms, 'subjects': tuple(subjects)})

    def __setstate__(self, state):
        self._lms = state['lms']
        self._subjects = state['subjects']
        self.tests = state.get('tests')
        lm = self._lms[0]
        self.dims = lm.dims
        self.coding = lm.coding
        self.column_names = lm.column_names

        if self.tests is None:
            self.samples = None
        else:
            self.samples = self.tests[self.column_names[0]].samples

    def __getstate__(self):
        return {'lms': self._lms, 'tests': self.tests,
                'subjects': self._subjects}

    def _single_column_coefficient(self, term, asds=False):
        coeff = NDVar(np.concatenate([lm._coefficient(term) for lm in self._lms]),
                      ('case',) + self.dims)
        if asds:
            return Dataset(('coeff', coeff))
        else:
            return coeff

    def column_ttest(self, term, return_data=False, popmean=0, *args, **kwargs):
        """Perform a one-sample t-test on a single model column

        Parameters
        ----------
        term : str
            Name of the term to test.
        return_data : bool
            Return the individual subjects' coefficients along with test
            results.
        popmean : scalar
            Value to compare Y against (default is 0).
        tail : 0 | 1 | -1
            Which tail of the t-distribution to consider:
            0: both (two-tailed);
            1: upper tail (one-tailed);
            -1: lower tail (one-tailed).
        samples : None | int
            Number of samples for permutation cluster test. For None, no
            clusters are formed. Use 0 to compute clusters without performing
            any permutations.
        pmin : None | scalar (0 < pmin < 1)
            Threshold for forming clusters:  use a t-value equivalent to an
            uncorrected p-value.
        tmin : None | scalar
            Threshold for forming clusters.
        tfce : bool
            Use threshold-free cluster enhancement (Smith & Nichols, 2009).
            Default is False.
        tstart, tstop : None | scalar
            Restrict time window for permutation cluster test.
        mintime : scalar
            Minimum duration for clusters (in seconds).
        minsource : int
            Minimum number of sources per cluster.

        Returns
        -------
        result : ttest_1samp
            T-test result.
        data : Dataset (only with ``return_data=True``)
            Dataset with subjects' coefficients.

        Notes
        -----
        Performs a one-sample t-test on coefficient estimates from all subjects
        to test the hypothesis that the coefficient is different from popmean
        in the population.
        """
        ds = self._single_column_coefficient(term, asds=True)
        res = ttest_1samp('coeff', popmean, None, None, ds, *args, **kwargs)
        if return_data:
            ds['subject'] = Factor(self._subjects)
            ds['n'] = Var([lm.n_cases for lm in self._lms])
            return res, ds
        else:
            return res

    def design(self, subject=None):
        if subject is None:
            lm = self._lms[0]
            subject = self._subjects[0]
        else:
            for lm, lm_subject in zip(self._lms, self._subjects):
                if lm_subject == subject:
                    break
            else:
                raise ValueError("subject=%r" % (subject,))

        table = lm.model.as_table(lm.coding)
        table.caption("Design matrix for %s" % subject)
        return table

    def _column_ttests(self, *args, **kwargs):
        "Precompute all tests"
        self.tests = {}
        for term in self.column_names:
            self.tests[term] = self.column_ttest(term, False, *args, **kwargs)
        self.samples = self.tests[self.column_names[0]].samples
