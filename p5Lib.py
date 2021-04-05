
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

def single_countplot(df, ax, x=None, y=None, top=None, order=True, hue=False, palette='plasma',
                     width=0.75, sub_width=0.3, sub_size=12, annot=False):
    
    import seaborn as sns
    import numpy as np

    ncount = len(df)
    if x:
        col = x
    else:
        col = y

    if top is not None:
        cat_count = df[col].value_counts()
        top_categories = cat_count[:top].index
        df = df[df[col].isin(top_categories)]

    if hue != False:
        if order:
            sns.countplot(x=x, y=y, data=df, palette=palette, ax=ax, order=df[col].value_counts().index, hue=hue)
        else:
            sns.countplot(x=x, y=y, data=df, palette=palette, ax=ax, hue=hue)
    else:
        if order:
            sns.countplot(x=x, y=y, data=df, palette=palette, ax=ax, order=df[col].value_counts().index)
        else:
            sns.countplot(x=x, y=y, data=df, palette=palette, ax=ax)

    format_spines(ax, right_border=False)

    if annot == True:
        if x:
            for p in ax.patches:
                x = p.get_bbox().get_points()[:, 0]
                y = p.get_bbox().get_points()[1, 1]
                try:
                    ax.annotate('{}\n{:.1f}%'.format(int(y), 100. * y / ncount), (x.mean(), y), ha='center', va='bottom')
                except:
                    pass
        else:
            for p in ax.patches:
                x = p.get_bbox().get_points()[1, 0]
                y = p.get_bbox().get_points()[:, 1]
                try:
                    ax.annotate('{} ({:.1f}%)'.format(int(x), 100. * x / ncount), (x, y.mean()), va='center')
                except:
                    pass

#============================================================================#

def format_spines(ax, right_border=True):

    import matplotlib
    
    # Setting up colors
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['top'].set_visible(False)
    if right_border:
        ax.spines['right'].set_color('#CCCCCC')
    else:
        ax.spines['right'].set_color('#FFFFFF')
    ax.patch.set_facecolor('#FFFFFF')

#============================================================================#

def gini(clusters_labels):
    """Compute the Gini coefficient for a clustering.
    Parameters:
        - clusters_labels: pd.Series of labels of clusters for each point.
    """

    # Get frequencies from clusters_labels
    clusters_labels = pd.Series(clusters_labels)
    frequencies = clusters_labels.value_counts()

    # Mean absolute difference
    mad = frequencies.mad()

    # Mean frequency of clusters
    mean = frequencies.mean()

    # Gini coefficient
    gini_coeff = 0.5 * mad / mean

    return gini_coeff

#============================================================================#

def metrics_clustering(algorithm_name,
                       standardized_features,
                       clusters_labels,
                       results_df):

    from sklearn import metrics

    # Number of clusters
    # -------------------------------------------
    nb_clusters = len(set(clusters_labels)) \
        - (1 if -1 in clusters_labels else 0)
    nb_clusters = int(nb_clusters)
    print('Estimated number of clusters (excluding noise): %d' % nb_clusters)
    results_df.loc[algorithm_name, 'Nb of clusters'] = nb_clusters

    # Silhouette score
    # -------------------------------------------
    silhouette = metrics.silhouette_score(
        standardized_features,
        clusters_labels
    )
    print("\nSilhouette coefficient: s = {:.3f}".format(silhouette))
    print("  Notice: values closer to 1 indicate a better partition")
    results_df.loc[algorithm_name, 'Silhouette'] = silhouette

    # Gini coefficient
    # --------------------------------------------
    gini_coeff = gini(clusters_labels)
    print("\nGini coefficient: G = {:.3f}".format(gini_coeff))
    print("  Notice: values closer to 0 indicate \
    homogenic frequencies for clusters.")
    results_df.loc[algorithm_name, 'Gini'] = gini_coeff

    # Sorting the pd.DataFrame of results
    results_df = results_df.sort_values('Silhouette', ascending=False)

    print("")
    return results_df

#============================================================================#

from sklearn.base import BaseEstimator


class GridSearch(BaseEstimator):
    """Classe permettant d'implémenter une recherche exhaustive sur grille
    (sans validation croisée) pour les algorithmes de clustering."""

    # Method: init
    # ------------------------------------------------------------------------------
    def __init__(
            self,
            estimator,  # clustering algorithm to test
            param_grid,  # research space for hyperparameters
            scoring=None):
        """Méthode d'initialisation prenant en entrée le modèle
        à tester et la grille de paramètres."""

        # getting parameters
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring

    # Method: fit
    # ------------------------------------------------------------------------------
    def fit(self, X):
        """Méthode permettant de réaliser la recherche sur grille,
        et renvoyant le meilleur modèle trouvé, ré-entraîné sur les données."""

        # initialization of the dict of results
        self.results_ = {"scores": [],
                         "params": [],
                         "models": [],
                         "fit_times": [],
                         "nb_clusters": [],
                         "gini_coef": []}

        # Loading modules
        from sklearn.model_selection import ParameterGrid

        # iterating upon all combinations of parameters
        for param_combination in ParameterGrid(param_grid):

            # instanciation of the model with selected parameters
            model = self.estimator(**param_combination)

            # Measuring training time while fitting the model on the data
            time_train = %timeit -n1 -r1 -o -q model.fit(X)
            time_train = time_train.average

            # Scoring the model
            if not self.scoring:  # if scoring parameter not defined
                model_score = model.score(X)
            else:  # if scoring parameter is defined
                try:
                    labels = model.labels_
                    model_score = self.scoring(X, labels)
                except:
                    model_score = np.nan

            # Computing number of clusters, excluding noise (#-1)
            nb_clusters = \
                len(set(model.labels_)) - (1 if -1 in clusters_labels else 0)
            nb_clusters = int(nb_clusters)

            # Computing Gini coefficient
            gini_coeff = gini(model.labels_)

            # saving results, parameters and models in a dict
            self.results_["scores"].append(model_score)  # scores
            self.results_["params"].append(param_combination)  # parameters
            self.results_["models"].append(model)  # trained models
            self.results_["fit_times"].append(time_train)  # training time
            self.results_["gini_coef"].append(gini_coeff)  # Gini coefficient
            self.results_["nb_clusters"].append(nb_clusters)  # nb of clusters

        # Selecting best model (assumes that 'greater is better')
        # -----------------------------------
        best_model_index, best_score = None, None  # initialisation
        # iterating over scores
        for index, score in enumerate(self.results_["scores"]):

            # initialisation
            if not best_score:
                best_score = score
                best_model_index = index

            # if score is better than current best_score
            if score > best_score:
                # update the current best_score and current best_model_index
                best_score = score
                best_model_index = index

        # Update attributes of the instance
        self.best_score_ = self.results_["scores"][best_model_index]
        self.best_params_ = self.results_["params"][best_model_index]
        self.best_estimator_ = self.results_["models"][best_model_index]
        self.best_index_ = best_model_index
        self.refit_time_ = self.results_["fit_times"][best_model_index]

        return self

    # Method: predict
    # ------------------------------------------------------------------------------
    def predict(self, X_test):
        """Méthode permettant de réaliser les prédictions sur le jeu de test,
        en utilisant le meilleur modèle trouvé avec la méthode .fit
        entraîné sur le jeu d'entraînement complet."""

        # use the .predict method of the estimator on the best model
        return self.best_model.predict(X_test)
    
#============================================================================#

def plot_clusters(
        standardized_features,
        clusters_labels,
        embedding_algo='tSNE',
        ax=None):
    """
    Arguments:
    ---------
    embedding_algo: 'tSNE' or 'PCA' or 'Isomap'
    """

    # Applying the embedding
    # -----------------
    # Import libraries
    from sklearn import manifold
    from sklearn import decomposition

    # Instanciation of the embedding
    if embedding_algo == 'tSNE':
        X_projected = X_tsne

    elif embedding_algo == 'Isomap':
        X_projected = X_isomap

    elif embedding_algo == 'PCA':
        X_projected = X_pca

    # Plotting the Isomap embedding
    # -----------------------------

    # If no axes is passed…
    if not ax:
        # Set the axes to the current one
        ax = plt.gca()  # ax = ax or plt.gca()
        # Set dimensions of the figure (if no axes is passed)
        plt.gcf().set_figwidth(12)
        plt.gcf().set_figheight(7)

    # Definitions of axis boundaries
    ax.set_xlim(X_projected[:, 0].min()*1.1, X_projected[:, 0].max()*1.1)
    ax.set_ylim(X_projected[:, 1].min()*1.1, X_projected[:, 1].max()*1.1)

    # Properties of the axes
    ax.set_title(embedding_algo, fontsize=20)

    if embedding_algo == 'PCA':
        # Names of x- and y- axis, with percentage of explained variance
        ax.set_xlabel('First component ({}%)'
                      .format(round(100*pca.explained_variance_ratio_[0], 1)))
        ax.set_ylabel('Second component ({}%)'
                      .format(round(100*pca.explained_variance_ratio_[1], 1)))
    else:
        ax.set_xlabel('First component')
        ax.set_ylabel('Second component')

    # Setting color
    NB_CLUSTERS = \
        len(set(clusters_labels)) - (1 if -1 in clusters_labels else 0)
    color = clusters_labels / NB_CLUSTERS

    # Setting color to black for noise points
    # color = pd.Series(clusters_labels / NB_CLUSTERS).astype('object')
    # color = color.map(lambda x: x if x != (-1 / NB_CLUSTERS) else 'black')

    # Plotting the scatter plot
    ax.scatter(
            X_projected[:, 0],  # x-coordinate
            X_projected[:, 1],  # y-coordinate
            c=color,  # base for coloration of points
            cmap=plt.cm.get_cmap('Set1'),  # colormap
            )
    
#============================================================================#

def clustering_plots(algorithm_name, standardized_features, clusters_labels):

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure()

    # Set properties of the figure
    fig.set_figheight(6)
    fig.set_figwidth(23)
    fig.suptitle('Visualisation of clustering with {}'
                 .format(algorithm_name), fontsize=20)

    # Set the geometry of the grid of subplots
    gs = gridspec.GridSpec(nrows=1, ncols=3,)

    # Initialize axes and set position (left to right, top to bottom)
    # Use sharex or sharey parameter for sharing axis
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # Filling the axes
    plot_clusters(
        standardized_features,
        clusters_labels,
        embedding_algo='tSNE',
        ax=ax1
    )
    plot_clusters(
        standardized_features,
        clusters_labels,
        embedding_algo='PCA',
        ax=ax2
    )
    plot_clusters(
        standardized_features,
        clusters_labels,
        embedding_algo='Isomap',
        ax=ax3
    )

    # Automatically adjusts subplots params to fit the figure
    gs.tight_layout(fig, rect=[0, 0, 1, 0.96])
    
#============================================================================#