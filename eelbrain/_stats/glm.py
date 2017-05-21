# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""General linear model

References
----------
Fox, J. (2008). Applied regression analysis and generalized linear models.
    2nd ed. Sage, Los Angeles.
Hopkins, K. D. (1976). A Simplified Method for Determining Expected Mean
    Squares and Error Terms in the Analysis of Variance. Journal of
    Experimental Education, 45(2), 13--18.
"""




import logging

import numpy as np
from scipy.linalg import lstsq
import scipy.stats

from .. import fmtxt
from .._utils import LazyProperty
from .._utils.print_funcs import strdict
from .._data_obj import (
    Model, Var, asmodel, assub, asvar, assert_has_no_empty_cells, find_factors,
    hasrandom, is_higher_order_effect, isbalanced, iscategorial, isnestedin)
from .opt import anova_fmaps, anova_full_fmaps, lm_res_ss, ss
from .stats import ftest_p
from . import test


# Method to use for least squares estimation:
# (0) Use scipy.linalg.lstsq
# (1) Use lstsq after Fox (2008) with caching of the model transformation
_lm_lsq = 0  # for the LM class


class hopkins_ems(dict):
    """Find components of the F-test denominator according to Hopkins (1976)

    A dictionary supplying for each of a model's effects the components of
    the F-test denominator (error term).

    "The E(MS) for any source of variation for any ANOVA model, in addition
    to the specified effect (main effect or interaction), includes the
    interaction of this effect with any random factor or combinations of
    random factors, plus any random effect nested within the specified effect,
    plus any random effect that either crosses or is nested within each
    ingredient and the specified effect." (Hopkins, 1976: 18)

    Parameters
    ----------
    x : Model
        ANOVA model. Needs to balanced and completely specified.
    """
    def __init__(self, x):
        super(hopkins_ems, self).__init__()

        if x.df_error > 0:
            err = "Hopkins E(MS) estimate requires a fully specified model"
            raise ValueError(err)
        elif not any(f.random for f in find_factors(x)):
            err = ("Need at least one random effect in fully specified model "
                   "(got %s)" % x.name)
            raise ValueError(err)
        elif not isbalanced(x):
            logging.warn('X is not balanced')

        self.x = x
        for e in x.effects:
            self[e] = _find_hopkins_ems(e, x)

    def __repr__(self):
        items = {}
        for k, v in self.items():
            kstr = ' %s' % k.name
            vstr = '(%s)' % ''.join(e.name + ', ' for e in v)
            items[kstr] = vstr
        return strdict(items, fmt='%s')


def _hopkins_ems_array(x):
    """Construct E(MS) array"""
    n = len(x.effects)
    out = np.zeros((n, n), dtype=np.int8)
    for row, erow in enumerate(x.effects):
        for col, ecol in enumerate(x.effects):
            out[row, col] = _hopkins_test(erow, ecol)
    return out


def _hopkins_test(e, e2):
    """Test whether e2 is in the E(MS) of e.

    Parameters
    ----------
    e : effect
        effect whose E(MS) is being constructed
    e2 : effect
        Model effect which is tested for inclusion in E(MS) of e
    """
    if e is e2:
        return False
    else:
        e_factors = find_factors(e)
        e2_factors = find_factors(e2)

        a = all((f in e_factors or f.random) for f in e2_factors)
        b = all((f in e2_factors or isnestedin(e2, f)) for f in e_factors)

        return a and b


def _find_hopkins_ems(e, x):
    "Tuple with all effects included in the Hopkins E(MS)"
    return tuple(e2 for e2 in x.effects if _hopkins_test(e, e2))


class LM(object):
    """Fit a linear model to a dependent variable

    Attributes
    ----------
    F, p : scalar
        Test of the null-hypothesis that the model does not explain a
        significant amount of the variance in the dependent variable.
    """
    def __init__(self, Y, X, sub=None, ds=None):
        """Fit the model X to the dependent variable Y

        Parameters
        ----------
        Y : Var
            Dependent variable.
        X : Model
            Model.
        sub : None | index
            Only use part of the data
        """
        # prepare input
        sub = assub(sub, ds)
        Y = asvar(Y, sub, ds)
        X = asmodel(X, sub, ds)

        assert len(Y) == len(X)
        assert X.df_error > 0

        # fit
        if _lm_lsq == 0:  # use scipy (faster)
            beta, SS_res, _, _ = lstsq(X.full, Y.x)
        elif _lm_lsq == 1:  # Fox
            # estimate least squares approximation
            beta = X.fit(Y)
            # estimate
            values = beta * X.full
            Y_est = values.sum(1)
            self._residuals = residuals = Y.x - Y_est
            SS_res = np.sum(residuals ** 2)
            if not Y.mean() == Y_est.mean():
                msg = ("Y.mean() != Y_est.mean() (%s vs "
                       "%s)" % (Y.mean(), Y_est.mean()))
                logging.warning(msg)
        else:
            raise ValueError

        # SS total
        SS_total = self.SS_total = np.sum((Y.x - Y.mean()) ** 2)
        df_total = self.df_total = X.df_total
        self.MS_total = SS_total / df_total

        # SS residuals
        self.SS_res = SS_res
        df_res = self.df_res = X.df_error
        self.MS_res = SS_res / df_res

        # SS explained
        SS_model = self.SS = self.SS_model = SS_total - SS_res
        df_model = self.df = self.df_model = X.df - 1  # don't count intercept
        self.MS_model = self.MS = SS_model / df_model

        # store stuff
        self.Y = Y
        self.X = X
        self.sub = sub
        self.beta = beta

    def __repr__(self):
        # repr kwargs
        kwargs = []
        if self.sub:
            kwargs.append(('sub', getattr(self.sub, 'name', '<...>')))

        fmt = {'Y': getattr(self.Y, 'name', '<Y>'),
               'X': repr(self.X)}
        if kwargs:
            fmt['kw'] = ', '.join([''] + list(map('='.join, kwargs)))
        else:
            fmt['kw'] = ''
        return "LM({Y}, {X}{kw})".format(**fmt)

    def anova(self, title='ANOVA', empty=True, ems=False):
        """ANOVA table for the linear model"""
        X = self.X
        values = self.beta * self.X.full

        if X.df_error == 0:
            e_ms = hopkins_ems(X)
        elif hasrandom(X):
            err = ("Models containing random effects need to be fully "
                   "specified.")
            raise NotImplementedError(err)
        else:
            e_ms = False

        # table head
        table = fmtxt.Table('l' + 'r' * (5 + ems))
        if title:
            table.title(title)

        if not isbalanced(X):
            table.caption("Warning: Model is unbalanced, use anova class")

        table.cell()
        headers = ["SS", "df", "MS"]
        if ems:
            headers += ["E(MS)"]
        headers += ["F", "p"]
        for hd in headers:
            table.cell(hd, r"\textbf", just='c')
        table.midrule()

        # MS for factors (Needed for models involving random effects)
        MSs = {}
        SSs = {}
        for e in X.effects:
            idx = X.full_index[e]
            SSs[e] = SS = np.sum(values[:, idx].sum(1) ** 2)
            MSs[e] = (SS / e.df)

        # table body
        results = {}
        for e in X.effects:
            MS = MSs[e]
            if e_ms:
                e_EMS = e_ms[e]
                df_d = sum(c.df for c in e_EMS)
                MS_d = sum(MSs[c] for c in e_EMS)
                e_ms_name = ' + '.join(repr(c) for c in e_EMS)
            else:
                df_d = self.df_res
                MS_d = self.MS_res
                e_ms_name = "Res"

            # F-test
            if MS_d:
                F = MS / MS_d
                p = 1 - scipy.stats.distributions.f.cdf(F, e.df, df_d)
                F_tex = fmtxt.stat(F, stars=test.star(p))
            else:
                F_tex = None
                p = None
            # add to table
            if e_ms_name or empty:
                table.cell(e.name)
                table.cell(SSs[e])
                table.cell(e.df, fmt='%i')
                table.cell(MS)
                if ems:
                    table.cell(e_ms_name)
                table.cell(F_tex)
                table.cell(fmtxt.P(p))
            # store results
            results[e.name] = {'SS': SS, 'df': e.df, 'MS': MS,
                               'E(MS)': e_ms_name, 'F': F, 'p': p}

        # Residuals
        if self.df_res > 0:
            table.cell("Residuals")
            table.cell(self.SS_res)
            table.cell(self.df_res, fmt='%i')
            table.cell(self.MS_res)

        return table

    @LazyProperty
    def F(self):
        return self.MS_model / self.MS_res

    @LazyProperty
    def p(self):
        return scipy.stats.distributions.f.sf(self.F, self.df_model, self.df_res)

    @LazyProperty
    def regression_table(self):
        """
        Not fully implemented!

        A table containing slope coefficients for all effects.

        """
        # prepare table
        table = fmtxt.Table('l' * 4)
        df = self.X.df_error
        table.cell()
        table.cell('\\beta', mat=True)
        table.cell('T_{%i}' % df, mat=True)
        table.cell('p', mat=True)
        table.midrule()
        #
        q = 1  # track start location of effect in Model.full
        ne = len(self.X.effects)
        for ie, e in enumerate(self.X.effects):
            table.cell(e.name + ':')
            table.endline()
            for i, name in enumerate(e.beta_labels):  # Fox pp. 106 ff.
                beta = self.beta[q + i]
                T = 0
                p = 0
                # todo: T/p
                table.cell(name)
                table.cell(beta)
                table.cell(T)
                table.cell(fmtxt.p(p))
            q += e.df
            if ie < ne - 1:
                table.empty_row()
        return table

    @LazyProperty
    def residuals(self):
        values = self.beta * self.X.full
        Y_est = values.sum(1)
        return self.Y.x - Y_est


def _nd_anova(x):
    "Create an appropriate anova mapper"
    x = asmodel(x)
    assert_has_no_empty_cells(x)
    if hasrandom(x):
        if not iscategorial(x):
            raise NotImplementedError("Random effects ANOVA with continuous "
                                      "predictors")
        elif x.df_error != 0:
            raise NotImplementedError("Random effects ANOVA need to be fully "
                                      "specified")
        return _FullNDANOVA(x)
    elif isbalanced(x):
        return _BalancedFixedNDANOVA(x)
    else:
        return _IncrementalNDANOVA(x)


class _NDANOVA(object):
    """Efficiently fit a model to multiple dependent variables."""
    def __init__(self, x, effects, dfs_denom):
        self.x = x
        self._n_obs = len(x)
        self.effects = effects
        self.n_effects = len(effects)
        self.dfs_nom = [e.df for e in effects]
        self.dfs_denom = dfs_denom
        self._flat_f_map = None

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, self.x.name)

    def map(self, y, perm=None):
        """Fit the model to multiple dependent variables

        Parameters
        ----------
        y : np.array (n_cases, ...)
            Assumes that the first dimension of Y provides cases.
            Other than that, shape is free to vary and output shape will match
            input shape.
        perm : None | array (n_cases, )
            Permutation.

        Returns
        -------
        f_maps : list
            A list with maps of F values (order corresponding to self.effects).
        """
        if y.shape[0] != self._n_obs:
            msg = ("Y has wrong number of observations (%i, model has %i)" %
                   (y.shape[0], self._n_obs))
            raise ValueError(msg)

        # find result container
        if self._flat_f_map is None:
            shape = (self.n_effects,) + y.shape[1:]
            f_map = np.empty(shape)
            flat_f_map = f_map.reshape((self.n_effects, -1))
        else:
            f_map = None
            flat_f_map = self._flat_f_map

        if y.ndim > 2:
            y = y.reshape((self._n_obs, -1))

        self._map(y, flat_f_map, perm)
        return f_map

    def _map(self, y, flat_f_map, perm):
        raise NotImplementedError

    def p_maps(self, f_maps):
        """Convert F-maps for uncorrected p-maps

        Parameters
        ----------
        f_maps : numpy array (n_effects, ...)
            Maps of f-values as returned by self.map().

        Returns
        -------
        p_maps : numpy array (n_effects, ...)
            Maps of uncorrected p values (corresponding to f_maps).
        """
        p_maps = np.empty_like(f_maps)
        for i in range(len(f_maps)):
            p_maps[i] = ftest_p(f_maps[i], self.dfs_nom[i], self.dfs_denom[i])
        return p_maps

    def preallocate(self, y_shape):
        """Pre-allocate an output array container.

        Parameters
        ----------
        y_shape : tuple
            Data shape, will allow preallocation of containers for results.

        Returns
        -------
        f_maps : array
            Properly shaped output array. Every time .map() is called, the
            content of this array will change (and map() will not return
            anything)
        """
        shape = (self.n_effects,) + y_shape[1:]
        f_map = np.empty(shape)
        self._flat_f_map = f_map.reshape((self.n_effects, -1))
        return f_map


class _BalancedNDANOVA(_NDANOVA):
    "For balanced but not fully specified models"
    def __init__(self, x, effects, dfs_denom):
        _NDANOVA.__init__(self, x, effects, dfs_denom)

        self._effect_to_beta = x._effect_to_beta
        self._x_full_perm = None
        self._xsinv_perm = None

    def _map(self, y, flat_f_map, perm):
        x = self.x
        if perm is None:
            x_full = x.full
            xsinv = x.xsinv
        else:
            if self._x_full_perm is None:
                self._x_full_perm = x_full = np.empty_like(x.full)
                self._xsinv_perm = xsinv = np.empty_like(x.xsinv)
            else:
                x_full = self._x_full_perm
                xsinv = self._xsinv_perm
            x.full.take(perm, 0, x_full)
            x.xsinv.take(perm, 1, xsinv)

        self._map_balanced(y, flat_f_map, x_full, xsinv)

    def _map_balanced(self, y, flat_f_map, x_full, xsinv):
        raise NotImplementedError


class _BalancedFixedNDANOVA(_BalancedNDANOVA):
    "For balanced but not fully specified models"
    def __init__(self, x):
        effects = x.effects
        dfs_denom = (x.df_error,) * len(effects)
        _BalancedNDANOVA.__init__(self, x, effects, dfs_denom)

        self.df_error = x.df_error

    def _map_balanced(self, y, flat_f_map, x_full, xsinv):
        anova_fmaps(y, x_full, xsinv, flat_f_map, self._effect_to_beta,
                    self.df_error)


class _FullNDANOVA(_BalancedNDANOVA):
    """For balanced, fully specified models.

    Object for efficiently fitting a model to multiple dependent variables.

    Parameters
    ----------
    x : Model
        Model which will be fitted to the data.

    Notes
    -----
    E(MS) for F statistic is determined after Hopkins (1976)
    """
    def __init__(self, x):
        e_ms = hopkins_ems(x)
        df_den = {e: sum(e_.df for e_ in e_ms[e]) for e in x.effects}
        effects = tuple(e for e in x.effects if df_den[e])
        dfs_denom = [df_den[e] for e in effects]
        _BalancedNDANOVA.__init__(self, x, effects, dfs_denom)

        self.e_ms = e_ms
        self._e_ms_array = _hopkins_ems_array(x)

    def _map_balanced(self, y, flat_f_map, x_full, xsinv):
        anova_full_fmaps(y, x_full, xsinv, flat_f_map, self._effect_to_beta,
                         self._e_ms_array)


class _IncrementalNDANOVA(_NDANOVA):
    def __init__(self, x):
        if hasrandom(x):
            raise NotImplementedError("Models containing random effects")
        comparisons, models, skipped = _incremental_comparisons(x)
        effects = tuple(item[0] for item in comparisons)
        dfs_denom = (x.df_error,) * len(effects)
        _NDANOVA.__init__(self, x, effects, dfs_denom)

        self._comparisons = comparisons
        self._models = models
        self._skipped = skipped
        self._SS_diff = None
        self._MS_e = None
        self._SS_res = None

        self._x_orig = x_orig = {}
        for i, x in models.items():
            if x is None:
                x_orig[i] = None
            else:
                x_orig[i] = (x.full, x.xsinv)
        self._x_perm = None

    def preallocate(self, y_shape):
        f_map = _NDANOVA.preallocate(self, y_shape)

        shape = self._flat_f_map.shape[1]
        self._SS_diff = np.empty(shape)
        self._MS_e = np.empty(shape)
        self._SS_res = {}
        for i in self._models:
            self._SS_res[i] = np.empty(shape)
        return f_map

    def _map(self, y, flat_f_map, perm):
        if self._SS_diff is None:
            shape = y.shape[1]
            SS_diff = MS_diff = np.empty(shape)
            MS_e = np.empty(shape)
            SS_res = {}
            for i in self._models:
                SS_res[i] = np.empty(shape)
        else:
            SS_diff = MS_diff = self._SS_diff
            MS_e = self._MS_e
            SS_res = self._SS_res

        if perm is None:
            x_dict = self._x_orig
        else:
            x_orig = self._x_orig
            x_dict = self._x_perm
            if x_dict is None:
                self._x_perm = x_dict = {}
                for i, x in self._models.items():
                    if x is None:
                        x_dict[i] = None
                    else:
                        x_dict[i] = (np.empty_like(x.full), np.empty_like(x.xsinv))

            for i in x_dict:
                if x_dict[i]:
                    x_orig[i][0].take(perm, 0, x_dict[i][0])
                    x_orig[i][1].take(perm, 1, x_dict[i][1])

        # calculate SS_res and MS_res for all models
        for i, x in x_dict.items():
            ss_ = SS_res[i]
            if x is None:
                ss(y, ss_)
            else:
                x_full, xsinv = x
                lm_res_ss(y, x_full, xsinv, ss_)

        # incremental comparisons
        np.divide(SS_res[0], self.x.df_error, MS_e)
        for i in range(self.n_effects):
            e, i1, i0 = self._comparisons[i]
            np.subtract(SS_res[i0], SS_res[i1], SS_diff)
            np.divide(SS_diff, e.df, MS_diff)
            np.divide(MS_diff, MS_e, flat_f_map[i])


def _incremental_comparisons(x):
    """Determine models for incremental comparisons

    Parameters
    ----------
    x : Model
        Model for which to derive incremental comparisons.

    Returns
    -------
    comparisons : list of (effect, int model_1, int model_0) tuples
        Comparisons to test each effect in x.
    models : dict {int -> Model}
        Models as indexed from comparisons.
    skipped : list of (effect, reason) tuples
        Effects that can't be tested.
    """
    comparisons = []  # (Effect, int m1, int m0)
    model_idxs = {}  # effect tuple -> ind
    models = {}  # int -> Model
    next_idx = 1

    # add full model
    model_idxs[tuple(x.effects)] = 0
    models[0] = x

    # Find comparisons for each effect
    skipped = []
    for e_test in x.effects:
        # find effects in model 0
        effects = []
        for e in x.effects:
            # determine whether e_test
            if e is e_test:
                pass
            elif is_higher_order_effect(e, e_test):
                pass
            else:
                effects.append(e)

        # get model 0
        e_tuple = tuple(effects)
        if e_tuple in model_idxs:
            idx0 = model_idxs[e_tuple]
            model0 = models[idx0]
        else:
            idx0 = next_idx
            next_idx += 1
            model_idxs[e_tuple] = idx0
            if len(effects):
                model0 = Model(effects)
            else:
                model0 = None

        # test whether comparison is feasible
        if model0 is None:
            df_res_0 = x.df_total
        else:
            df_res_0 = model0.df_error

        if e_test.df > df_res_0:
            skipped.append((e_test, "overspecified"))
            continue
        elif idx0 not in models:
            models[idx0] = model0

        # get model 1
        effects.append(e_test)
        e_tuple = tuple(effects)
        if e_tuple in model_idxs:
            idx1 = model_idxs[e_tuple]
        else:
            idx1 = next_idx
            next_idx += 1
            model_idxs[e_tuple] = idx1
            models[idx1] = Model(effects)

        # store comparison
        comparisons.append((e_test, idx1, idx0))

    return comparisons, models, skipped


class incremental_f_test:
    """Incremental F-Test between two linear models

    Test the null hypothesis that the model of lm1 does not explain more
    variance than that of lm0 (with model_1 == model_0 + q factors,  q > 0).
    If lm1 is None it is assumed that lm1 is the full model with 0 residuals
    (Fox 2008, p. 109 f.).

    Parameters
    ----------
    lm1, lm0 : Model
        The two models to compare.
    MS_e, df_e : scalar | None
        Parameters for random effects models: the Expected value of MS;
        if None, the error MS of lm1 is used (valid for fixed effects
        models).

    Attributes
    ----------
    lm1 : Model
        The extended model.
    lm0 : Model
        The control model.
    SS : scalar
        the difference in the SS explained by the two models.
    df : int
        The difference in df between the two models.
    MS : scalar
        The MS of the difference.
    F, p : scalar
        F and p valuer of the comparison.
    """
    def __init__(self, lm1, lm0, MS_e=None, df_e=None, name=None):
        if lm1 is None:
            lm1_SS_res = 0
            lm1_MS_res = 0
            lm1_df_res = 0
        else:
            lm1_SS_res = lm1.SS_res
            lm1_MS_res = lm1.MS_res
            lm1_df_res = lm1.df_res

        if MS_e is None:
            assert df_e is None
            MS_e = lm1_MS_res
            df_e = lm1_df_res
        else:
            assert df_e is not None

        SS_diff = lm0.SS_res - lm1_SS_res
        df_diff = lm0.df_res - lm1_df_res
        MS_diff = SS_diff / df_diff

        if df_e > 0:
            F = MS_diff / MS_e
            p = ftest_p(F, df_diff, df_e)
        else:
            F = None
            p = None

        self.lm0 = lm0
        self.lm1 = lm1
        self.SS = SS_diff
        self.MS = MS_diff
        self.df = df_diff
        self.MSe = MS_e
        self.dfe = df_e
        self.F = F
        self.p = p
        self.name = name

    def __repr__(self):
        name = ' %r' % self.name if self.name else ''
        return "<incremental_f_test%s: F=%.2f, p=%.3f>" % (name, self.F, self.p)


class ANOVA(object):
    """Univariate ANOVA.

    Mixed effects models require balanced models and full model specification
    so that E(MS) can be estimated according to Hopkins (1976).

    Parameters
    ----------
    y : Var
        dependent variable
    x : Model
        Model to fit to Y
    sub : index
        Only use part of the data.
    title : str
        Title for the results table (optional).
    ds : Dataset
        Dataset to use data from.

    Examples
    --------
    The objects' string representation is the
    anova table, so the model can be created and examined inone command::

    >>> ds = datasets.get_loftus_masson_1994()
    >>> print test.ANOVA('n_recalled', 'exposure.as_factor()*subject', ds=ds)
                SS       df   MS         F         p
    ---------------------------------------------------
    exposure     52.27    2   26.13   42.51***   < .001
    ---------------------------------------------------
    Total      1005.87   29

    For other uses, properties of the fit can be accessed with methods and
    attributes.
    """
    # Unbalanced models
    # -----------------
    #  - The SS of Effects which do not include the between-subject factor are
    #    higher than in SPSS
    #  - The SS of effects which include the between-subject factor agree with
    #    SPSS
    def __init__(self, y, x, sub=None, title=None, ds=None):
        # prepare kwargs
        sub = assub(sub, ds)
        y = asvar(y, sub, ds)
        x = asmodel(x, sub, ds)

        if len(y) != len(x):
            raise ValueError("y and x must describe same number of cases")
        assert_has_no_empty_cells(x)

        # save args
        self.y = y
        self.x = x
        self.title = title
        self._log = []

        # decide which E(MS) model to use
        if x.df_error == 0:
            is_mixed = True
            fx_desc = 'Mixed'
        elif x.df_error > 0:
            if hasrandom(x):
                err = ("Models containing random effects need to be fully "
                       "specified.")
                raise NotImplementedError(err)
            is_mixed = False
            fx_desc = 'Fixed'
        else:
            raise ValueError("Model Overdetermined")
        self._log.append("Using %s effects model" % fx_desc)

        # list of (name, SS, df, MS, F, p)
        self.f_tests = []
        self.names = []

        if len(x.effects) == 1:
            self._log.append("single factor model")
            lm1 = LM(y, x)
            self.f_tests.append(lm1)
            self.names.append(x.name)
            self.residuals = lm1.SS_res, lm1.df_res, lm1.MS_res
        else:
            if is_mixed:
                pass  # <- Hopkins
            else:
                full_lm = LM(y, x)
                SS_e = full_lm.SS_res
                MS_e = full_lm.MS_res
                df_e = full_lm.df_res

            comparisons, models, skipped = _incremental_comparisons(x)

            # store info on skipped effects
            for e_test, reason in skipped:
                self._log.append("SKIPPING: %s (%s)" % (e_test.name, reason))

            # fit the models
            lms = {}
            for idx, model in models.items():
                if model.df_error > 0:
                    lm = LM(y, model)
                else:
                    lm = None
                lms[idx] = lm

            # incremental F-tests
            for e_test, i1, i0 in comparisons:
                name = e_test.name
                skip = None

                # find model 0
                lm0 = lms[i0]
                lm1 = lms[i1]

                if is_mixed:
                    # find E(MS)
                    EMS_effects = _find_hopkins_ems(e_test, x)

                    if len(EMS_effects) > 0:
                        lm_EMS = LM(y, Model(EMS_effects))
                        MS_e = lm_EMS.MS_model
                        df_e = lm_EMS.df_model
                    else:
                        if lm1 is None:
                            SS = lm0.SS_res
                            df = lm0.df_res
                        else:
                            SS = lm0.SS_res - lm1.SS_res
                            df = lm0.df_res - lm1.df_res
                        MS = SS / df
                        skip = ("no Hopkins E(MS); SS=%.2f, df=%i, "
                                "MS=%.2f" % (SS, df, MS))

                if skip:
                    self._log.append("SKIPPING: %s (%s)" % (e_test.name, skip))
                else:
                    res = incremental_f_test(lm1, lm0, MS_e=MS_e, df_e=df_e, name=name)
                    self.f_tests.append(res)
                    self.names.append(name)
            if not is_mixed:
                self.residuals = SS_e, df_e, MS_e

        self._is_mixed = is_mixed

    def __repr__(self):
        return "anova(%s, %s)" % (self.y.name, self.x.name)

    def __str__(self):
        return str(self.table())

    def print_log(self):
        out = self._log[:]
        print('\n'.join(out))

    def table(self):
        """Create an ANOVA table

        Returns
        -------
        anova_table : eelbrain.fmtxt.Table
            Anova table.
        """
        # table head
        table = fmtxt.Table('l' + 'r' * (5 + 2 * self._is_mixed))
        if self.title:
            table.title(self.title)
        table.cell()
        table.cells("SS", "df", "MS")
        if self._is_mixed:
            table.cells(fmtxt.symbol('MS', 'denom'), fmtxt.symbol('df', 'denom'))
        table.cells("F", "p")
        table.midrule()

        # table body
        for name, f_test in zip(self.names, self.f_tests):
            table.cell(name)
            table.cell(fmtxt.stat(f_test.SS))
            table.cell(f_test.df)
            table.cell(fmtxt.stat(f_test.MS))
            if self._is_mixed:
                table.cell(fmtxt.stat(f_test.MSe))
                table.cell(f_test.dfe)

            if f_test.F:
                stars = test.star(f_test.p)
                table.cell(fmtxt.stat(f_test.F, stars=stars))
                table.cell(fmtxt.p(f_test.p))
            else:
                table.endline()

        # residuals
        if self.x.df_error > 0:
            table.empty_row()
            table.cell("Residuals")
            SS, df, MS = self.residuals
            table.cell(fmtxt.stat(SS))
            table.cell(df)
            table.cell(fmtxt.stat(MS))
            table.endline()

        # total
        table.midrule()
        table.cell("Total")
        SS = np.sum((self.y.x - self.y.mean()) ** 2)
        table.cell(fmtxt.stat(SS))
        table.cell(len(self.y) - 1)
        return table


def anova(y, x, sub=None, title=None, ds=None):
    """Univariate ANOVA.

    Mixed effects models require balanced models and full model specification
    so that E(MS) can be estimated according to Hopkins (1976).

    Parameters
    ----------
    y : Var
        dependent variable
    x : Model
        Model to fit to Y
    sub : index
        Only use part of the data.
    title : str
        Title for the results table (optional).
    ds : Dataset
        Dataset to use data from.

    Returns
    -------
    table : FMText Table
        Table with results.

    Examples
    --------
    >>> ds = datasets.get_loftus_masson_1994()
    >>> test.ANOVA('n_recalled', 'exposure.as_factor()*subject', ds=ds)
                SS       df   MS         F         p
    ---------------------------------------------------
    exposure     52.27    2   26.13   42.51***   < .001
    ---------------------------------------------------
    Total      1005.87   29
    """
    anova_ = ANOVA(y, x, sub, title, ds)
    return anova_.table()
