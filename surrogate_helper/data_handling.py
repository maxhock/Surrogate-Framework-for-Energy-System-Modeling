"""
Functions related to handling the data
"""
import pandas as pd
import numpy as np
import keras

def loadData(data_file):
    """Load data from a given path string.
    This function can be modified/overloaded to add features.

    Args:
        data_file -- Path string to file

    Returns:
        Dataframe of loaded file
    """
    data = pd.read_csv(data_file, dtype=np.float32)
    return data

def augmentData(data, n_scenarios, augment=()):
    """Augment given data with:
        timestamp
        variational mode decomposition
        seasonal/trend decomposition (loess)
    VMD and STL should not be applied together as they iterate 
    over all columns blindly and would create a lot of columns.

    Args:
        augment -- list of augmentations to apply to the data
            "vmd" -- variational mode decomposition
            "stl" -- seasonal/trend decomposition
            "mstl" -- multi-seasonal trend decomposition
            "ts" -- timestamps
            int -- number of modes for vmd
    Returns:
        Augmented dataframe
    """

    if isinstance(augment, (str, int)):
        augment = (augment,)
    
    if "vmd" in augment:
        # Variational Mode Decomposition of the non capacity inputs

#         from vmdpy import VMD
        from sktime.transformations.series.vmd import VmdTransformer
        new_df = pd.DataFrame()

        K = 5 # Number or modes to split into

        for i in (3,5,7,9):
            if i in augment or str(i) in augment:
                K = i
        
        print(f"Augmenting data with variational mode decomposition with {K} modes")
        
        for column in data.columns:
            if "capacity" not in column and "net" not in column:
                modes = pd.DataFrame()
                #import pdb; pdb.set_trace()
                for chunk in np.array_split(data[column], n_scenarios): # iterrate over each scenario
                    #import pdb; pdb.set_trace()
                    u = VmdTransformer(K).fit_transform(chunk)
#                     u, _, _ = VMD(
#                         chunk,
#                         2000,  # moderate bandwidth constraint
#                         0.0,  # noise-tolerance (no strict fidelity enforcement)
#                         K,  # 3 modes
#                         0,  # no DC part imposed
#                         1,  # initialize omegas uniformly
#                         1e-7, # tolerance for noise
#                     )
                    modes = pd.concat([modes, pd.DataFrame(u)])
                modes.columns = [f"{column}_mode{i}" for i in range(K)]
                modes.reset_index(drop=True, inplace=True)
                new_df = pd.concat(
                    [new_df, data[column], modes], axis=1, ignore_index=True
                )
            else:
                new_df = pd.concat([new_df, data[column]], axis=1, ignore_index=True)
        data = new_df

    if "stl" in augment:
        # Add Seasonal Trend Loess Decomposition
        print("Augmenting data with Season Trend Decomposition")

        from statsmodels.tsa.seasonal import STL
        new_df = pd.DataFrame()
        for column in data.columns:
            if "capacity" not in column and "net" not in column:
                modes = pd.DataFrame()

                for chunk in np.array_split(data[column], n_scenarios): # iterrate over each scenario
                    u = STL(chunk, period=24, robust=True).fit()
                    
                    modes = pd.concat([modes, pd.DataFrame((u.trend, u.seasonal, u.resid)).T])

                modes.columns = [f"{column}_{x}" for x in modes.columns]
                modes.reset_index(drop=True, inplace=True)
                new_df = pd.concat(
                    [new_df, data[column], modes], axis=1
                )
            else:
                new_df = pd.concat([new_df, data[column]], axis=1)
        data = new_df
    
    
    if "mstl" in augment:
        # Add Multi-Seasonal Trend Loess Decomposition
        print("Augmenting data with Multi-Season Trend Decomposition")
        from statsmodels.tsa.seasonal import MSTL
        new_df = pd.DataFrame()
        for column in data.columns:
            if "capacity" not in column and "net" not in column:
                modes = pd.DataFrame()

                for chunk in np.array_split(data[column], n_scenarios): # iterrate over each scenario
                    u = MSTL(chunk, periods=(24,24*7), stl_kwargs={"robust":True}).fit()
                    #import pdb; pdb.set_trace()
                    if len(u.seasonal.shape) > 1:
                        modes = pd.concat([modes, pd.DataFrame((u.trend, u.seasonal.iloc[:,0],u.seasonal.iloc[:,1], u.resid)).T])
                    else:
                        raise ValueError("Scenario too short, cannot apply all periods. please only use STL")

                modes.columns = [f"{column}_{x}" for x in modes.columns]
                modes.reset_index(drop=True, inplace=True)
                new_df = pd.concat(
                    [new_df, data[column], modes], axis=1
                )
            else:
                new_df = pd.concat([new_df, data[column]], axis=1)
        data = new_df
        

    if "ts" in augment:
        data = data.copy()
        # Add timestamp columns
        print("Augmenting data with Timestamps")
        data.insert(0, "hour", pd.Series(data.index, dtype=int).mod(int(len(data.index) / n_scenarios)))
        #data = data.rename_axis("hour").reset_index(drop=False).mod(int(len(data.index)/n_scenarios))
        #data = data.rename_axis("hour").reset_index(drop=False).mod(8760)
        data.insert(1, "day", data["hour"].floordiv(24) + 1)
        data.insert(1, "week", data["hour"].floordiv(24 * 7) + 1)
        data.insert(1, "weekday", data["hour"].floordiv(24).mod(7) + 1)
        data.insert(1, "dayhour", data["hour"].mod(24) + 1)
        #import pdb; pdb.set_trace()
    return data

def splitData(data, train=0.8, test=1/27):
    """Split dataframe into multiple dataframes according to the given proportions.

    Args:
        data: Dataframe
        train: ratio of data not used for testing to use for training
        test: ratio of all data used for testing

    Returns:
        Tuple of dataframes ordered training, validation, testing
    """

    data_len = len(data)

    train_cut = int(data_len * (1 - test) * train)
    test_cut = int(data_len * (1 - test))

    data_train = data.iloc[:train_cut]
    data_val = data.iloc[train_cut + 1 : test_cut - 1]
    data_test = data.iloc[test_cut:]
    return data_train, data_val, data_test


def makeDataset(df, length: int, stride: int, batch_size=128):
    """Turn dataframe into keras timeseries dataset.

    The function calls timeseries_dataset_from_array on a given dataframe.
    It uses the last column as the target and all others as features.
    It slices the array into stacked arrays with length `length`.
    The distance between the first element of two stacked arrays is defined by `stride`.

    Args:
        df: Dataframe to be converted
        length: Size of each stacked array
        stride: Distance between first elements
        shuffle: Deprecated! shuffles elements before conversion

    Returns:
        Dataset
    """

    # if shuffle:
    #     df = df.sample(frac=1, random_state=200).reset_index(drop=True)

    dataset = keras.utils.timeseries_dataset_from_array(
        df.iloc[:, 0:-1],  # features
        df.iloc[length-1:, -1],  # target
        sequence_length=length,
        sequence_stride=stride,
        batch_size=batch_size,
    )
    return dataset


def processData(data, data_settings: dict, ram=8e6):
    """Prepare data for training and inference.

    Args:
        data: Dataframe to be prepared
        data_settings: Config parameters regarding the preparation of data
        ram: gpu memory in kilobytes to calculate appropriate batch size

    Returns:
        Tuple of keras datasets in order training, validation, testing.
    """

    if data_settings["normalize"] is True:
        data = (data - data.mean()) / data.std().replace(0,1)

    data_train, data_val, data_test = splitData(
        data, data_settings["training_size"], data_settings["testing_size"]
    )

    n = int(ram / (data_settings["timeseries_window"] * data.shape[-1] * 64 / 8) / 2)
    # align batch size to closest byte
    batch_size = n & (1 << n.bit_length() - 1)
    
    #import pdb; pdb.set_trace()
    # apply makeDataset to train, val and test
    data_train, data_val, data_test = (
        makeDataset(
            x,
            data_settings["timeseries_window"],
            data_settings["timeseries_stride"],
            batch_size,
        )
        for x in (data_train, data_val, data_test)
    )

    for batch in data_train.take(1):
        inputs, targets = batch
        input_shape = inputs.numpy().shape[1:3]
        print("Input shape:", input_shape)
        print("Target shape:", targets.numpy().shape)

    return data_train, data_val, data_test, input_shape


