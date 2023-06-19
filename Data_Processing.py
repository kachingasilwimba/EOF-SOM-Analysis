#========================================================================================================
#                        The Code Preprocess the Data                                                   #
#========================================================================================================
class Data:
    def __init__(self, file_path, with_time=False):
        ds = xr.open_mfdataset(file_path).sel(lat=slice(24.0, 50.0), lon=slice(360 - 126.0, 360 - 66.0))
        self.lat = ds.lat.values
        self.lon = ds.lon.values
        self.time = ds["time"].values if with_time else None
        self.with_time = with_time
        
        #=================== Extract variables
        ET = np.nan_to_num(ds.tsl.sel(levgrnd=slice(0, 2.1)).mean('levgrnd').values)
        SNM = np.nan_to_num(ds.snm.values)
        SM = np.nan_to_num(ds.mrlsl.sel(levsoi=slice(0, 12)).mean('levsoi').values)
        MRRO = np.nan_to_num(ds.mrro.values)
        
        #=================== Stack features and targets
        X = np.stack((ET, SNM, MRRO), axis=-1)
        y = SM
        
        #=================== Standardize features and targets
        mX, sX = X.mean(axis=(0, 1, 2)), X.std(axis=(0, 1, 2))
        my, sy = y.mean(axis=(0, 1, 2)), y.std(axis=(0, 1, 2))
        self.mX, self.sX, self.my, self.sy = mX, sX, my, sy
        X = (X - mX) / sX
        y = (y - my) / sy
        
        #=================== Split data into train, test, and validation sets
        index = np.arange(len(X))
        rng = np.random.default_rng(42)
        train_test_index = rng.permutation(index[:7920])
        train_index = train_test_index[:7920]
        test_index = np.append(train_test_index[7920:], index[7920:8430])
        val_index = index[8430:]
        batch_size = 10
        
        X_train, X_test, X_val = X[train_index], X[test_index], X[val_index]
        y_train, y_test, y_val = y[train_index], y[test_index], y[val_index]
        t_train, t_test, t_val = time[train_index], time[test_index], time[val_index]
#         t_train, t_test, t_val = self.time[train_index], self.time[test_index], self.time[val_index] if self.with_time else (None, None, None)
        
        #=================== Create TensorFlow datasets
        self.d_train = tf.data.Dataset.from_tensor_slices((X_train, y_train, train_index) if self.with_time else (X_train, y_train))
        self.d_test = tf.data.Dataset.from_tensor_slices((X_test, y_test, test_index) if self.with_time else (X_test, y_test))
        self.d_val = tf.data.Dataset.from_tensor_slices((X_val, y_val, val_index) if self.with_time else (X_val, y_val))
        self.d_train = self.d_train.shuffle(len(train_index)).batch(batch_size)
        self.d_test = self.d_test.shuffle(len(test_index)).batch(batch_size)
        self.d_val = self.d_val.shuffle(len(val_index)).batch(batch_size)
    
    def get_data(self, batch_size=10):
        if self.with_time:
            return self.d_train, self.d_test, self.d_val, self.time
        else:
            return self.d_train, self.d_test, self.d_val