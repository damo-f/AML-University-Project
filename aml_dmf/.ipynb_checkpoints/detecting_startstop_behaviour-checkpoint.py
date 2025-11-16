# This py module does an statisitcal analysis of the provided dataset. The beginning and end gets cut away, if the duration of the time between ticks is outside of the [0.2, 0.8] quantile. For more details please refer to the readme file.

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class timecut:
    def __init__(self, dfin):
        self.dfin = dfin
        self.tickstat = None

    def statcut(self):
        tick = pd.DataFrame()
        tickct = pd.Series()
        tick['Timestamp'] = self.dfin['Timestamp']
        tick['Tickval'] = self.dfin['PT Revolution']
        tick['Time'] = self.dfin['Timestamp'].diff()
        tick.dropna(inplace=True)  # Dropping first line due to NaN from .diff()
        # Variables for the loop
        tickstate = 1
        lotickamount = 0
        # Detection of the ticks
        for i, row in tick.iterrows():
            if row['Tickval'] > -1:
                if tickstate != 0:
                    tickstate = 0
                    lotickamount += 1
                tickct[i] = lotickamount
            else:
                if tickstate == 0:
                    tickstate = 1
                tickct[i] = 1
        # The Ticks in a Series
        tick['Tickct'] = tickct
        pivtick = pd.pivot_table(tick, columns=['Tickct'], values=['Time'],
                                 aggfunc={'Time': 'sum'})
        # Enabling other classes to access pivtick
        statistic = pivtick
        self.tickstat = statistic
        # Statistical analysis and determining cuttingpoint regarding the
        # first/last value in quantile. Time value
        quant = pivtick.quantile([0.2, 0.80], axis=1)
        # Two lows between ticks due to magnet positioning
        av = pivtick.iloc[0, :].mean()*2
        lo_met = False
        hi_met = False
        for i, col in pivtick.items():
            if  quant.at[0.2, 'Time'] > col['Time'] or col['Time'] > quant.at[0.8, 'Time']:
                lowcut = i
                lo_met = True
                continue
            else:
                i = i-1
                break
        for i, col in pivtick.iloc[0, ::-1].items():
            if quant.at[0.2, 'Time'] > col or col > quant.at[0.8, 'Time']:
                highcut = i
                hi_met = True
                continue
            else:
                i = i-1
                break
        # No cutting needed?
        if hi_met is False:
            highcut = lotickamount
        if lo_met is False:
            lowcut = 0
        # cutting
        first_index = tick[tick['Tickct'] >= lowcut].index[0]
        last_index = tick[tick['Tickct'] == highcut].index[-1]
        tick.drop(index=tick.index[last_index + 1:], inplace=True)
        tick.drop(index=tick.index[:first_index], inplace=True)
        tick.drop(['Tickval', 'Time', 'Tickct'], axis=1, inplace=True)
        tick.reset_index(drop=True, inplace=True)
        print('Initial length: ', len(pivtick.columns), '\n',
              'Cut ticks at beginning: ',
              lowcut, '\n', 'Cut away ticks at ending: ',
              (len(pivtick.columns) - highcut), '\n Resumes in ',
              len(pivtick.columns) - lowcut - (len(pivtick.columns) - highcut),
              'Ticks', '\n The average Revolution duration[ms] : ', av)
        return tick

    def get_tickstat(self):
             # Zugriff auf die pivot-Tabelle (self.tickstat)
            if self.tickstat is not None:
                return self.tickstat
            else:
                raise ValueError("tickstat ist noch nicht berechnet.")
