import pandas as pd
import numpy as np
import warnings
from detecting_startstop_behaviour import timecut
warnings.filterwarnings("ignore", category=FutureWarning)

def datacut(name):
    
    
    inp = pd.read_csv(name, sep=';',
                      header=None,
                      converters={col: (lambda x: x.replace(',', '.')) for col in
                                  pd.read_csv(name, sep=';', header=None, nrows=0).columns})
    #inp.to_csv('AML Roboproject/Data/data.csv', index=False)
    firstrows = inp.head(2)
    # filling up empty "" cells with NaN
    inp.replace("", np.nan, inplace=True)
    #inp.replace(',', '.', inplace=True)
    # monitored sensors
    sensors = ['Force imposed', 'Laser', 'US horizontal',
               'US vertical',
               'Beltrevolution', 'PT Revolution']
    # choose every file in folder
    checkfile = pd.read_csv('Data/Original/001_l_000N_fa_r_f.csv', sep=';',
                      header=None, 
                      converters={col: (lambda x: x.replace(',', '.')) for col in pd.read_csv('Data/Original/001_l_000N_fa_r_f.csv', sep=';', header=None, nrows=0).columns})
    # check header strucutre
    for i in [0, 2, 4, 6, 8, 10]:
        if checkfile.iloc[0, i] != firstrows.iloc[0, i]:
            print('The row 0 "Labels" is not structured equally')
        if checkfile.iloc[1, i] != firstrows.iloc[1, i]:
            print('The row 1 "Times" is not structured equally')
        else:
            print('Column ', i, ' is ok')


    # setting up the main data csv df
    # copying the first time column + Sensor values as main time axis
    ts = pd.DataFrame(data=inp.iloc[3:, 0]).astype(float)
    ts.columns = ['Timestamp']
    merg = pd.DataFrame(data=inp.iloc[3:, [10, 11]]).astype(float)
    merg.columns = ['Timestamp', sensors[5]]
    ts = ts.merge(merg, how='left', on='Timestamp')
    ts.iloc[:, 1] = ts.iloc[:, 1].interpolate(method='linear')
    ts.reset_index(drop=True, inplace=True)
    ts.columns = ['Timestamp', sensors[5]]
    # Insert statcut here!
    tc = timecut(ts)
    df = tc.statcut()
    # Merging the sensors according the Timestamp set in ts
    # adding all the other non-NaN-lines
    for i in range(0, len(sensors), 1):
        movcol = pd.DataFrame(data=inp.iloc[3:, [i*2, i*2+1]])
        movcol.reset_index(drop=True, inplace=True)
        movcol.columns = ['Timestamp', sensors[i]]
        movcol.dropna(inplace=True)
        movcol = movcol.astype(float)  # converting all values to float
        df = df.merge(movcol, how='left', on='Timestamp')
    
    # Interpolating all NaN Values due to uneven timestamp
    df.iloc[:, 6] = df.iloc[:, 6].interpolate(method='linear')
    #df.head()

    #df.to_csv('Data/data.csv', index=False)

    return df