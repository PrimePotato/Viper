from decorators import print_me, memoized, init_accepts

import functools
import pyximport;
from mpl_toolkits.mplot3d import Axes3D
from pandas.tslib import relativedelta
from sortedcontainers import SortedDict
from sortedcontainers import SortedList

pyximport.install()

from collections import OrderedDict, Hashable
from scipy.interpolate import griddata, RectBivariateSpline

import pandas as pd
import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
import timeit as ti

from pandas import Timestamp as pts
from scipy.interpolate import PchipInterpolator as Pchip
from scipy.interpolate import interp1d
from numba import jit
from scipy.stats import norm
from holidays import Event


DAYS_IN_YEAR = 365

def my_square(ary):
    return np.array([np.sqrt(i) for i in ary])


@print_me()
class QuoteNode(object):
    @init_accepts(float, pd.Timestamp, pd.Timestamp, float, float)
    def __init__(self, vol, start_date, quote_date, strike, fwd):
        self.vol = vol
        self.strike = strike
        self.start_date = start_date
        self.fwd = fwd
        self.quote_date = quote_date
        self.time_delta = quote_date - start_date
        self.days = float(self.time_delta.days)
        self.var = vol * vol * self.days
        self.cdelta = self._calculate_delta()

    def _calculate_delta(self):
        return np.log(self.fwd/self.strike)*np.sqrt(self.var)


class DateFunctions(object):
    @staticmethod
    def resolve_tenor(tenor, start):
        un = tenor[:-1]
        fq = tenor[-1:]
        if fq in 'mM':
            fq_delta = relativedelta(months=1)
        elif fq in 'dD':
            fq_delta = relativedelta(days=1)
        elif fq in 'wW':
            fq_delta = relativedelta(weeks=1)
        else:
            raise Exception('Unresolvable tenor {}'.format(tenor))
        return start + un*fq_delta


class TimeSpan(object):
    def __init__(self, **kwargs):
        if 'start' in kwargs:
            self.start = kwargs['start']
        else:
            raise Exception('start argument missing')
        if 'expiry' in kwargs:
            self.expiry = kwargs['expiry']
            self._delta = self.expiry - self.start
            self.tenor = str(self._delta.days) + 'D'
        elif 'tenor' in kwargs:
            self.tenor = kwargs['tenor']
            self.expiry = DateFunctions.resolve_tenor(self.tenor, self.start)
        else:
            raise Exception('Ill-defined missing arguments')


class QuoteRequest(object):
    #TODO: Combine and tidy up with node
    def __init__(self, time_span, **kwargs):
        self.time_span = time_span
        if 'delta' in kwargs:
            pass
        elif 'strike' in kwargs:
            self.strike = kwargs['strike']
        else:
            pass


# class RiskLayer


class Skeleton(object):
    def __init__(self, kind, bounds):
        self.kind = kind
        self.bounds = bounds


class QuoteList(object):

    def __init__(self, quote_nodes, events):
        self.qns = quote_nodes
        self.events = events
        self.var_dict = {(qn.days, qn.strike): qn.vol * qn.vol * qn.days for qn in self.qns}
        self._ustrikes = np.array(SortedList(set(qn.strike for qn in self.qns)))
        self._udays = np.array(SortedList(set(qn.days for qn in self.qns)))
        self._x, self._y, (self._mesh_x, self._mesh_y) = self._build_tight_grid()
        # self._x, self._y, (self._mesh_x, self._mesh_y) = self._build_mesh_grid(0.25, 200, 200, False)

        px = np.array([qn.days for qn in self.qns])
        py = np.array([qn.strike for qn in self.qns])
        t = np.array([qn.vol for qn in self.qns])

        self._n_grid = griddata((px, py), t, (self._mesh_x, self._mesh_y), method='nearest')
        self._c_grid = griddata((px, py), t, (self._mesh_x, self._mesh_y), method='cubic')

        self._ext_mask = np.isnan(self._c_grid)
        self._rel_mask = np.logical_not(self._ext_mask)

        self._xr = self._mesh_x[self._rel_mask]
        self._yr = self._mesh_y[self._rel_mask]
        self._cr = self._c_grid[self._rel_mask]

        #TODO: can calc moneyness instead of delta perhaps
        self._cn_grid = griddata((self._xr, self._yr), self._cr, (self._mesh_x, self._mesh_y), method='nearest')
        self._spline = RectBivariateSpline(self._x, self._y, self._cn_grid.transpose())

    @staticmethod
    def _nan_remover(x):
        return x[np.logical_not(np.isnan(x))]

    def plot(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        xx, yy = np.meshgrid()
        ax.plot_surface(xx, yy, self._cn_grid, rstride=1, cstride=1,
                        linewidth=0, antialiased=False)
        plt.show()

    def quote_days(self, days):
        uds = self._udays
        eds = [e.days for e in self.events]

    @memoized
    def interp(self, kind):
        return interp1d(self.x, self.y, kind=kind)

    def _populate_mesh(self):
        pass

    def vol_quote(self, days, strike):
        iv = self._spline.ev(days, strike)
        return iv

    def _mean_vol(self):
        return np.mean([qn.vol for qn in self.qns])

    def _mean_fwd(self):
        return np.mean([qn.fwd for qn in self.qns])

    def _max_days(self):
        return np.max([qn.days for qn in self.qns])

    def _build_mesh_grid(self, p, xn, yn, sqr):
        mv = self._mean_vol()
        mx = self._max_days()
        mf = self._mean_fwd()
        sd = mv*mf*mx / DAYS_IN_YEAR
        up = norm.ppf(1-p*.5)*sd + mf
        dn = norm.ppf(p*.5)*sd + mf
        x = np.sqrt(np.linspace(0, mx, yn)) if sqr else np.linspace(0, mx, yn)
        y = np.linspace(dn, up, xn)
        return x, y, np.meshgrid(x, y)

    def _build_tight_grid(self):
        x = self._udays
        y = self._ustrikes
        return x, y, np.meshgrid(x, y)



qqs = []
qqs.append(QuoteNode(0.11, pts('01/Aug/2009'), pts('02/Aug/2009'), 100., 100.))

qqs.append(QuoteNode(0.12, pts('01/Aug/2009'), pts('01/Oct/2009'), 90., 100.))
qqs.append(QuoteNode(0.11, pts('01/Aug/2009'), pts('01/Oct/2009'), 100., 100.))
qqs.append(QuoteNode(0.14, pts('01/Aug/2009'), pts('01/Oct/2009'), 110., 100.))
qqs.append(QuoteNode(0.12, pts('01/Aug/2009'), pts('01/Oct/2009'), 120., 100.))
qqs.append(QuoteNode(0.11, pts('01/Aug/2009'), pts('01/Oct/2009'), 130., 100.))
qqs.append(QuoteNode(0.15, pts('01/Aug/2009'), pts('01/Oct/2009'), 140., 100.))

qqs.append(QuoteNode(0.11, pts('01/Aug/2009'), pts('01/Oct/2010'), 90., 100.))
qqs.append(QuoteNode(0.13, pts('01/Aug/2009'), pts('01/Oct/2010'), 100., 100.))
qqs.append(QuoteNode(0.15, pts('01/Aug/2009'), pts('01/Oct/2010'), 110., 100.))
qqs.append(QuoteNode(0.12, pts('01/Aug/2009'), pts('01/Oct/2010'), 120., 100.))
qqs.append(QuoteNode(0.14, pts('01/Aug/2009'), pts('01/Oct/2010'), 130., 100.))

qqs.append(QuoteNode(0.17, pts('01/Aug/2009'), pts('01/Oct/2011'), 90., 100.))
qqs.append(QuoteNode(0.18, pts('01/Aug/2009'), pts('01/Oct/2011'), 100., 100.))
qqs.append(QuoteNode(0.13, pts('01/Aug/2009'), pts('01/Oct/2011'), 110., 100.))
#
# ts = TimeSpan(start=pts('1/1/2010'), expiry=pts('1,1,2011'))
# print(ts.tenor)
# ts = TimeSpan(start=pts('1/1/2010'), tenor='1W')
# print(ts.expiry)

# print(q1)

# print(q1.vol)

evs = [Event(20,0.1,3)]
ql = QuoteList(qqs, evs)
ql.quote_days(24)

fig = plt.figure()
ax = fig.gca(projection='3d')

x, y, (xx, yy) = ql._build_mesh_grid(0.25, 50, 50, False)
ax.plot_surface(xx, yy, ql.vol_quote(xx, yy), rstride=1, cstride=1,
                linewidth=0, antialiased=False)
plt.show()

sk = Skeleton('a','b')

print(ql.vol_quote(1, 100))

# xi = np.arange(1, 2000, 1)
# yi = np.sqrt(ql.pchip(xi) / xi)
# print(yi)

# plt.plot(xi, yi)
# plt.show()

# print(ql.interp('linear'))
#
# print(ti.timeit(lambda: ql.vol_quote(60, 'linear'), number=10000))
