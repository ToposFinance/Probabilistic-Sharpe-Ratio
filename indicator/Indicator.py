
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import math

def ClassByName(classname):
    cls = globals()[classname]

    return cls()

def IndicatorFactory(config):
    ind=ClassByName(config['type'])
    ind.set_parms(**config['parms'])
    return ind

class Indicator(ABC):

    @abstractmethod
    def set_parms(self, **parms):
        pass

    @abstractmethod
    def predict(self,data):
        pass






class BreakOut(Indicator):

    def __init__(self,window=1,direction=1):
        self._window=window
        self._direction=direction

    def set_parms(self, **parms):
        self._window = parms['window']
        self._direction = parms['direction']


    def predict(self,data):
        data1=data-data.rolling(window=self._window).mean()
        data1=np.sign(data1)
        pred = self._direction*data1.rolling(window=self._window).mean()
        return pred




class Digital(Indicator):

    def __init__(self,window=1,direction=1):
        self._window=window
        self._direction=direction

    def set_parms(self, **parms):
        self._window = parms['window']
        self._direction = parms['direction']


    def predict(self,data):
        data1=np.sign(data.pct_change())
        print(data1)
        pred = self._direction*data1.rolling(window=self._window).mean()
        return pred

class MaxMinusMin(Indicator):

    def __init__(self,window=1,direction=1):
        self._window=window
        self._direction=direction

    def set_parms(self, **parms):
        self._window = parms['window']
        self._direction = parms['direction']


    def predict(self,data):
        data1=data.pct_change()
        pred = self._direction*(data1.rolling(window=self._window).max()-data1.rolling(window=self._window).min())

        return pred

class DigitalDistanceToIndex(Indicator):

    def __init__(self,window=1,direction=1):
        self._window=window
        self._direction=direction

    def set_parms(self, **parms):
        self._window = parms['window']
        self._direction = parms['direction']


    def predict(self,data):
        data1=np.sign(data.pct_change())
        print(data1)
        pred = self._direction*data1.rolling(window=self._window).mean()
        pred=pred.sub(pred.mean(axis=1),axis=0)
        return pred
class Sharpe(Indicator):

    def __init__(self,window=1,direction=1):
        self._window=window
        self._direction=direction

    def set_parms(self, **parms):
        self._window = parms['window']
        self._direction = parms['direction']


    def predict(self,data):
        pred = self._direction*data.rolling(window=self._window).mean() / data.rolling(window=self._window).std() * math.sqrt(24 * 365)
        return pred



class ZScore(Indicator):

    def __init__(self,**parms):
        self._window=parms['window']

    def predict(self,data):
        pred = (data-data.rolling(window=self._window).mean()) / data.rolling(window=self._window).std()
        return pred


class MACD(Indicator):

    def __init__(self,**parms):
        self._short_window=parms['short_window']
        self._long_window = parms['long_window']

    def predict(self,data):
        pred = data.rolling(window=self._short_window).mean()-data.rolling(window=self._long_window).mean()
        return pred

class OCHL(Indicator):
    def __init__(self,**parms):
        self._window=parms['window']


    def predict(self,data):
        ochl=(data['Close']-data['Open'])/(data['High']-data['Low']+0.0001)
        ochlma=ochl.rolling(window=self._window).mean()
        return ochlma

class DistanceToIndex(Indicator):
    def __init__(self,window=1,direction=1):

        self._window=window
        self._direction=direction

    def set_parms(self, **parms):

        self._window = parms['window']
        self._direction = parms['direction']

    def predict(self,data):
        rets=data.pct_change()
        distance=rets.sub(rets.mean(axis=1),axis=0)
        pred=self._direction*distance.rolling(window=self._window).mean()
        return pred


class Return(Indicator):
    def __init__(self,window=1,direction=1):

        self._window=window
        self._direction=direction

    def set_parms(self, **parms):

        self._window = parms['window']
        self._direction = parms['direction']

    def predict(self,data):
        rets=data.pct_change(self._window)


        return rets

class AutoCorr(Indicator):
    def __init__(self, window=1, direction=1):
        self._window = window
        self._direction = direction

    def set_parms(self, **parms):
        self._window = parms['window']
        self._direction = parms['direction']

    def predict(self, data):
        rets = data.pct_change()
        autocorr=rets*rets.shift(1)
        mean=autocorr.rolling(window=self._window).mean()

        return rets

class AbsoluteReturn(Indicator):
    def __init__(self, window=1, direction=1):
        self._window = window
        self._direction = direction

    def set_parms(self, **parms):
        self._window = parms['window']
        self._direction = parms['direction']

    def predict(self, data):
        rets = data.diff(self._window)

        return rets

class DistanceToIndexZ(Indicator):

    def __init__(self,window=1,direction=1):

        self._window=window
        self._direction=direction

    def set_parms(self, **parms):

        self._window = parms['window']
        self._direction = parms['direction']
    def predict(self,data):
        rets=data.pct_change()
        distance=rets.sub(rets.mean(axis=1),axis=0)
        pred=self._direction*(distance-distance.rolling(window=self._window).mean())/distance.rolling(window=self._window).std()
        return pred
if __name__ == '__main__':

    '''
    prefix = "s3://cl-m-mn-private-us-east-1/backtests/"

    sharpe=Sharpe(window=24*4)

    data = pd.read_parquet(prefix + "prices/{}.{}.90day.prices.parquet".format(2022, 1))
    close = data.pivot(columns=['symbol'])['Close']
    returnc = close.pct_change()
    ret2023 = returnc.sum(axis=1)

    sharpe_out=sharpe.predict(ret2023)

    print(sharpe_out)
    '''