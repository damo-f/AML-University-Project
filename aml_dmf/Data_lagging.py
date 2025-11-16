import numpy as np
import warnings
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore", category=FutureWarning)


class lagging:
    @staticmethod
    def create_lag_Gauss(data, n_lags):
        scalerh = MinMaxScaler()
        scalerv = MinMaxScaler()
        val_h = data['US horizontal'].values.reshape(-1, 1)
        val_v = data['US vertical'].values.reshape(-1, 1)
        sc_val_h = scalerh.fit_transform(val_h).reshape(-1, 1)
        sc_val_v = scalerv.fit_transform(val_v).reshape(-1, 1)
        X, Y = [], []
        for i in range(n_lags, len(data)):
            lagged_val_h = sc_val_h[i - n_lags:i]
            lagged_val_v = sc_val_v[i - n_lags:i]
            lagged_dataX = np.append(lagged_val_h, lagged_val_v).reshape(1, -1)
            lagged_dataY = np.append(sc_val_h[i], sc_val_v[i]).reshape(1, -1)
            X = np.append(X, lagged_dataX)
            Y = np.append(Y, lagged_dataY)
        print('Lag-Lines created: ', i)
        return np.array(X).reshape(-1, n_lags*2), np.array(Y).reshape(-1, 2) , scalerh, scalerv
    
    def create_lag_DesTree_flex(data, n_lags, position, idler):
        n_lags = n_lags-1  # Due to issue with position and index
        X, Y = [], []
        for i in range(n_lags, len(data)):
            lag_data_X1 = data.loc[i - n_lags:i, ['US horizontal']].values
            lag_data_X2 = data.loc[i - n_lags:i, ['US vertical']].values
            lagged_dataX = np.append(lag_data_X1, lag_data_X2).reshape(1, -1)
            X.append(lagged_dataX.reshape(1, -1))
            lagged_data_Y = np.append(data.loc[i, 'Force imposed'], position)
            lagged_data_Y = np.append(lagged_data_Y, idler)
            Y.append(lagged_data_Y.reshape(1, -1))
        X = np.vstack(X)
        Y = np.vstack(Y)
        print('Lag-Lines created: ', len(X))
        return X, Y