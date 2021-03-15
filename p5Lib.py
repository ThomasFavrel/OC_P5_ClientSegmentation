
#============================================================================#

def show_values_on_bars(axs, vertical=True, space=0.4):
    '''
    Show Values on a bar chart.

    Parameters
    ----------
    axs : plt.axes
        Axes matplotlib.
    vertical : BOOL, optional
        Show values on a vertical barplot. The default is True.
    space : FLOAT, optional
        Space between the end og the bar and the value. The default is 0.4.

    Returns
    -------
    None.

    '''
    
    import numpy as np
    
    def _show_on_single_plot(ax):
        if vertical == True:
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + space
                value = int(p.get_height())
                ax.text(_x, _y, value, ha="center") 
        elif vertical == False:
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + space
                _y = p.get_y() + p.get_height() / 2
                value = int(p.get_width())
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)
      
#============================================================================#
        
def display_filling_of_col(df, line=0, color='#3556C8', figsize=(8, 5), show_values=False):
    '''
    Display the filling of columns in a Dataframe.

    Parameters
    ----------
    df : Dataframe
        Dataframe.
    line : INT, optional
        Number of line to display. The default is 0 to display all lines.
    color : COLOR, optional
        Color of the plot. The default is '#3556C8'.
    figsize : TUPLE, optional
        Size of the plot. The default is (8, 5).
    show_values : BOOL, optional
        Show values. The default is False.

    Returns
    -------
    None.

    '''
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    df_tmp = pd.DataFrame()
    for col in df.columns:
        df_tmp[col] = pd.Series(df[col].count())
    x = list(df_tmp.T.sort_values(by=0, ascending=False)[0] / df.shape[0] * 100)
    y = list(df_tmp.T.sort_values(by=0, ascending=False).index)
    fig, ax = plt.subplots(figsize=figsize)
    if line == 0:
        sns.barplot(x=x, 
                    y=y,
                   orient='h', color=color)
    else:
        sns.barplot(x=x[:line], 
                    y=y[:line],
                   orient='h', color=color)
    if show_values == True:
        show_values_on_bars(ax, vertical=False)

#============================================================================#
        
def display_cate_bar(data, var, show_values=True, figsize=(5,5), color='b'):
    '''
    Display the distribution of a categorical variable.

    Parameters
    ----------
    data : Dataframe
        Dataframe.
    var : STRING
        Name of the variable to display.
    show_values : BOOL, optional
        Show values. The default is True.
    figsize : TUPLE, optional
        Size of the plot. The default is (5,5).
    color : COLOR, optional
        Color of the plot. The default is 'b'.

    Returns
    -------
    None.

    '''
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    
    value_cont = pd.DataFrame.from_dict(dict(data[var].value_counts())
                                    ,orient='index')
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=value_cont[0],
       y=value_cont.index,
       color=color,
       orient='h')
    if show_values:
        show_values_on_bars(ax, vertical=False)

#============================================================================#

def downcast(df):
    '''
    This function tries to downcast integer and floating dtypes columns
    to the smallest numerical corresponding dtype.
    It returns a dictionnary of the actually downcasted dtypes.    

    Parameters
    ----------
    df : Dataframe
        Dataframe to downcast.

    Returns
    -------
    dict_dtypes : DICT
        DESCRIPTION.

    '''
    
    import pandas as pd
    
    # initialise the dict of downcasted dtypes for features
    dict_dtypes = {}
    
    # getting list of integer columns
    columns_int = df.select_dtypes(include=['integer']).columns
    
    for column in columns_int:
        old_dtype = str(df[column].dtypes)
        # trying to downcast integer columns (np.int8 to np.int64)
        df[column] = pd.to_numeric(df[column], downcast='integer')
        new_dtype = str(df[column].dtypes)
        
        # if dtype was downcasted
        if new_dtype != old_dtype:
            print(f"Column {column} downcasted from {old_dtype} to {new_dtype}.")
            # add new key in dictionnary
            dict_dtypes[column] = str(df[column].dtypes)

    # getting list of floatting columns
    columns_float = df.select_dtypes(include=['floating']).columns
    
    for column in columns_float:
        old_dtype = str(df[column].dtypes)
        # trying to downcast float columns (np.float32 to np.float64)
        df[column] = pd.to_numeric(df[column], downcast='float')
        new_dtype = str(df[column].dtypes)
        
        # if dtype was downcasted
        if new_dtype != old_dtype:
            print(f"Column {column} downcasted from {old_dtype} to {new_dtype}.")
            # add new key in dictionnary
            dict_dtypes[column] = str(df[column].dtypes)
        
    # return dict of downcasted dtypes
    return dict_dtypes

#============================================================================#

def fillingrate_filter_rows(df, limit_rate):
    '''
    This function drop the rows where the filling rate is less than a defined 
    limit rate.

    Parameters
    ----------
    df : Dataframe
        Dataframe to filter.
    limit_rate : FLOAT
        Completeness threshold of rows to filter.

    Returns
    -------
    Dataframe
        Dataframe filtered.

    '''
    

    # Count of the values on each row
    rows_count = df.count(axis=1)

    # Number of columns in the dataframe
    nb_columns = df.shape[1]
    
    # Calculating filling rates
    filling_rates = rows_count / nb_columns

    # Define a mask of features with a filling_rate bigger than the limit rate
    mask = filling_rates > limit_rate
       
    # Get the number of rows under threshold
    number_rows_under_limit_rate = len(filling_rates[~mask])
    print("Number of rows with a filling rate below {:.2%}: {} rows.".format(limit_rate, number_rows_under_limit_rate))

    # Return a projection on the selection of features
    return df[mask]

#============================================================================#

def smart_imputation(dataframe):

    '''
    Do column-wise imputation based on the dtypes.

    Parameters
    ----------
    dataframe : Dataframe
        Dataframe to filter.

    Returns
    -------
    None.

    '''
    
    # Load libraries
    import pandas as pd
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler
    
    # Getting names of columns based on data types
    categorical_features = list(dataframe.select_dtypes(include=['category']).columns)
    boolean_features = list(dataframe.select_dtypes(include=['bool']).columns)
    numerical_features = list(dataframe.select_dtypes(include='number').columns)
    datetime_features = list(dataframe.select_dtypes(include='datetime').columns)
    timedelta_features = list(dataframe.select_dtypes(include='timedelta').columns)
    object_features = list(dataframe.select_dtypes(include='object').columns)
    
    # Make a copy for proceeding imputation on
    df = dataframe.copy()
    
    #-----------------------------------------------------------------------
    # Imputation of numerical features
        
    # Proceed to imputation for ALL quantitative features
    # Do not work well on timeseries… (impute negative values)
    numerical_imputer = IterativeImputer()
    df[numerical_features] = numerical_imputer.fit_transform(df[numerical_features])
    
    #-----------------------------------------------------------------------
    # Imputation of timeseries features with a kNN regressor
    
    # Conversion of timeseries features to integers
    timeseries_features = datetime_features + timedelta_features
    df[timeseries_features] = df[datetime_features].values.astype('int64')
    
    # Standardization of the numerical features
    standardizer = StandardScaler()
    df_std = standardizer.fit_transform(df[numerical_features])
    
    # Train a kNN regressor model for each categorical feature
    for feature in timeseries_features:
        # filter the non-missing data
        mask = df[feature].notnull()
        # proceed imputation only if there is missing-data
        if not mask.all():
            # filter the data for training
            X_std = df_std[mask]
            y = df.loc[mask, feature]
            # train the model
            knn = KNeighborsRegressor(n_neighbors=5, n_jobs=-1).fit(X_std, y)
            # predict the missing data for missing data
            X_mis = df_std[~mask]
            df.loc[~mask, feature] = knn.predict(X_mis)
            
#============================================================================#

def global_filling_rate(dataframe):
    """Compute and displays global filling rate of a DataFrame"""

    # get the numbers of rows and columns in the dataframe
    nb_rows, nb_columns = dataframe.shape
    print("DataFrame has {} rows and {} columns.".format(nb_rows, nb_columns))

    # get the number of non-Nan data in the dataframe
    nb_data = dataframe.count().sum()

    # computing the filling rate
    filling_rate = nb_data / (nb_rows * nb_columns)
    missing_rate = 1 - filling_rate

    # computing the total missing values
    missing_values = (nb_rows * nb_columns) - nb_data

    # display global results
    print("")
    print("Global filling rate of the DataFrame: {:.2%}".format(filling_rate))
    print("Missing values in the DataFrame: {} ({:.2%})"
          .format(missing_values, missing_rate))

    # compute number of rows with missing values
    mask = dataframe.isnull().any(axis=1)
    rows_w_missing_values = len(dataframe[mask])
    rows_w_missing_values_percentage = rows_w_missing_values / nb_rows

    # display results
    print("")
    print("Number of rows with missing values: {} ({:.2%})"
          .format(rows_w_missing_values, rows_w_missing_values_percentage))

    # compute number of columns with missing values
    mask = dataframe.isnull().any(axis=0)
    cols_w_missing_values = len(dataframe[dataframe.columns[mask]].columns)
    cols_w_missing_values_percentage = cols_w_missing_values / nb_columns

    # display results
    print("Number of columns with missing values: {} ({:.2%})"
          .format(cols_w_missing_values, cols_w_missing_values_percentage))

#============================================================================#

def columns_filling_rate(dataframe, columns='all', missing_only=False):
    """Calculate and displays the filling rate for
    a particular column in a pd.DataFrame."""
    
    # Importations
    import pandas as pd
    import numpy as np
    
    # If 'feature' is not specified
    if columns == 'all':
        columns = dataframe.columns
        
    
    # initialization of the results DataFrame
    results = pd.DataFrame(columns=['nb_values', 'missing_values', 'filling_rate'])
        
    # for each feature
    for column in columns:

        # Count of the values on each column
        values_count = dataframe[column].count()
        
        # Computing missing values
        nb_rows = dataframe.shape[0]
        missing_values = nb_rows - values_count

        # Computing filling rates
        filling_rate = values_count / nb_rows
        if missing_only and missing_values == 0:
            filling_rate = np.nan
        
        # Adding a row in the results' dataframe
        results.loc[column] = [values_count, missing_values, filling_rate]

    # Sorting the features by number of missing_values
    results = results.dropna(subset=['filling_rate'])
    results = results.sort_values('filling_rate')
    
    if results.empty == False:
        return results
    else:
        print("No missing value.")

#============================================================================#



#============================================================================#