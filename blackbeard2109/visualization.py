### LIBRARIES ###
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, r2_score, mean_absolute_error, mean_squared_error, accuracy_score, f1_score, precision_score, recall_score,\
                            roc_auc_score, roc_curve
import plotly.express as px
import folium
from IPython.display import display



### FUNCTIONS ###

def draw_boxplot(df, column, color = "b", figsize=(8,8), title = None, label_column_name = None):
    '''
    This function makes a box plot for a specific column.
    ------------------
    Args:
        df: dataframe
        column: column to be represented.
        color: color to be used.
        figsize = here we define the graph size, it has to be a tuple with 2 values
        title: graph name.
        label_column_name: column to be graphed.
    '''
    
    plt.figure(figsize=figsize)
    sns.boxplot(x=df[column], color=color)
    plt.title(title)
    plt.xlabel(label_column_name)
    plt.show()





def sort(df, column):
    '''
    This function sorts a DataFrame according to the selected column. 
    ------------------
    Args:
        df: Pandas DataFrame
        column: here we write the column we want to sort the dataframe by
    ------------------
    Return:
        the same dataframe, only ordered by the selected column
    '''
    df = df.sort_values(by=column,ascending=False)
    return df




def pairplot_heatmap(df):
    
    '''
    Display function that plots a pairplot and a heatmap at the same time showing a pearson correlation at the top.
    ------------------
    Args:
        df = dataframe with numeric variables
    '''

    def codot(*args, **kwargs):
        corr_r = args[0].corr(args[1], 'pearson')
        corr_text = f"{corr_r:2.2f}".replace("0.", ".")
        ax = plt.gca()
        ax.set_axis_off()
        marker_size = abs(corr_r) * 10000
        ax.scatter([.5], [.5], marker_size, [corr_r], alpha=0.6, cmap="coolwarm",
                   vmin=-1, vmax=1, transform=ax.transAxes)
        font_size = abs(corr_r) * 40 + 5
        ax.annotate(corr_text, [.5, .5,],  xycoords="axes fraction",
                    ha='center', va='center', fontsize=font_size)

    g = sns.PairGrid(df, aspect=1.4, diag_sharey=False)
    g.map_lower(sns.regplot, lowess=True, ci=False, line_kws={'color': 'black'})
    g.map_diag(sns.distplot, kde_kws={'color': 'black'})
    g.map_upper(codot);

    return g




def draw_missing_ratio(df, style = 'classic', figsize=(10,5), cmap='inferno', color = "lightgrey", lw=2,  title = None ,fontsize=18):
    ''' 
    This function shows a heatmap with missing values ratio of a dataframe
    ----------------
    Args:
        df = here we define the dataframe we want to analize
        style = here we define the graph style we want to use. 
                To see all the available styles, please type this:  print(plt.style.available)
        figsize = here we define the graph size, it has to be a tuple with 2 values
        cmap = here we define the graph's color palette. To change it, please see the seaborn library
        color = here we define the color of the vertical separation bars
        lw = here we define the thickness of the vertical separation bars
        title = here we define the graph's title, by default is set to None
        fontsize = here we define the title size, if there is a title
    '''
    plt.style.use(style)

    plt.figure(figsize=figsize)

    ax = sns.heatmap(df.isna(),yticklabels=False,xticklabels=df.columns,cbar=False,cmap=cmap)

    ## vertical separation bars between the dataframe columns
    for i in range(df.isna().shape[1]+1):
        ax.axvline(i,color = color, lw=lw);
        
    plt.title(title, fontsize=fontsize);
    plt.show()
    



def draw_statistic_values(df, figsize=(10,8), palette="crest", s= 500, alpha=0.8,  title = None ,fontsize=18, loc_legend= "upper left"):
    ''' 
    This function shows in a scatterplot the 5 most common statistic measures of each numeric column of the dataframe:
        mean: the average
        min= the minimum value
        max= the maximum value
        50%= the median
        std= the standard deviation
    ------------------------
    Args:
        df= the dataframe
        figsize = here we define the graph size, it has to be a tuple with 2 values
        palette= here we define the graph's color palette. To change it, please see the seaborn library
        s= here we define the size of each symbol
        alpha= here we define the transparency of the symbols
        title= here we define th etitle of the graph. It's set to None by default
        fontsize= here we define the title size, if there is a title
        loc_legend= here we define the position of the legend
    ------------------------
    Result:
        a scatterplot with these 5 statistic measures for each column of the dataframe
    '''
    
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()

    ax.set_title(title);
    
    df_stats = df.describe().T
    df_stats = df_stats[["mean","min","max", "50%", "std"]]

    sns.scatterplot(data=df_stats, palette=palette, s= s, alpha=alpha );

    plt.legend(loc=loc_legend);





def draw_target_transformation(column, figsize=(15,5), color = "b"):
            
    ''' 
    This function shows the distribution of a dataframe specific column. 
    Usually, this column is the target (for example in a machine learning problem)
    but, actually, we could apply it to any column with NUMERICAL values.
    
    IMPORTANT: box cox transformation raise an error with 0 values or negative ones.
    
    This function is useful when, in a machine learning problem, we want to see if the distribution of the target column
    is a normal one, or if we have to apply some others transformation to obtain best results.
    ----------------
    Args:
        column = here we define the column we want to see the transformations of. Value must be inserted as Pandas Series
        figsize = here we define the graph size, it has to be a tuple with 2 values
        color = here we define the bars color. Default value is blue ("b")
    ----------------
    Results:
        4 graphs: in the first one we'll see the original distribution, in the second one we'll see the logaritmic distribution,
        in the third one we'll see the boxcox distribution and in the last graph we'll see the power distribution.
    '''
    
    fig,axes = plt.subplots(1, 4, figsize=figsize, sharey=True)

    # Original target
    sns.histplot(column, kde=False, color= color, ax=axes[0])
    axes[0].set_title("Original target")

    # Logaritmic
    sns.histplot(np.log(column),kde=False, color= color, ax=axes[1])
    axes[1].set_title("Log")

    # Box-cox
    #here we define a try/except to manage some critical values
    try:
            sns.histplot(stats.boxcox(column)[0],kde=False,color= color,  ax=axes[2])
            axes[2].set_title("Box-Cox");
    except:
            print("To visualize the boxcox graphs, values must be positive and different from zero.")

    # Power 2
    sns.histplot(np.power(column, 2),kde=False, color= color, ax=axes[3])
    axes[3].set_title("Power 2");





def draw_feat_importance(importance,columns,model_type, figsize=(10,8)):
    '''   
    This function shows a graph with feature importance values of a trained model.
    ------------------
    Args:
        importance: here we have to put the trained model with the feature importance function, we have to write the model, followed by this sentence:  .feature_importances_
            (for example, if we have a trained random forest model, called "rf", the name will be this:  rf.feature_importances_ ).
        columns: here we put the name of the columns we want to show the feature importances of
                (usually, all the dataframe columns, except the target one).
        model_type: here we put the name of model we used to train our dataframe
                (for example random forest, xgboost, etc...)
        figsize: here we define the graph size, it has to be a tuple with 2 values.
    ------------------
    Result:
        the function shows a horizontal bar plot with all the feature importances, sorted by descending order
    
    '''
    
    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(columns)

    #Create a DataFrame using a Dictionary, to store the feature names and their respective feature importances
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    feat_imp_df = pd.DataFrame(data)

    #Sort the DataFrame in order decreasing feature importance
    feat_imp_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

   
    plt.figure(figsize=figsize)
    sns.barplot(x=feat_imp_df['feature_importance'], y=feat_imp_df['feature_names'])
    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')






def draw_maps(latitude = 40.4167 , longitude = -3.70325, zoom = 6):
    '''
    This function displays a map based on the latitude, longitude, and zoom.
    By default it will show Spain, with a zoom of 6.
    ------------------
    Args: 
        latitude: here we enter the latitude value, in decimal format
        longitude: here we enter the longitude value, in decimal format
        zoom: here we enter the zoom value of the map
    '''
    
    center = [latitude, longitude]
    my_map = folium.Map(location=center, zoom_start=zoom)
    display(my_map)



  

def draw_sunburst(data_frame, path, color):
    '''
    This function performs a sunburst graph. 
    ------------------
        Args: 
        -"data_frame": the dataframe we want to use
        -"path": we must introduce (as a list) the two columns or variables we want to represent
        -"color": it refers to the column or main variable, that will determine the color (the hue) of the graph
    '''

    fig = px.sunburst(
    data_frame = data_frame,
    path = path,
    color = color,
    color_discrete_sequence = ["red","green","blue","orange"],
    maxdepth = -1,
    )
    fig.update_traces(textinfo='label+percent entry')
    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
    fig.show()
    
    



def draw_donut_chart(df, column, title = None):
    ''' This function performs a donut chart. 
    ------------------
        Args: 
            -df = the dataframe 
            -column = enter the column whose two values you want to represent in the chart 
            -title = string format. Here we have to define a title for the graph. By default it's set to None
        '''

    total = df[column].value_counts()
    my_circle=plt.Circle( (0,0), 0.7, color='white') 
    plt.figure(figsize=(10,10))
    plt.pie(total.values,
            labels = total.index,
            autopct='%1.2f%%')
    p = plt.gcf()
    p.gca().add_artist(my_circle)
    plt.title(title)
    plt.show()
    
    



def draw_three_countplot(df, column_ax0, column_ax1, column_ax2, column_hue0, column_hue1, column_hue2, title0, title1,
                     title2, palette1, palette2, palette3):
    '''
    This function performs three vertical countplot graphs with included legend, titles
    and labels with 45 degree rotation and color palette to choose from. 
    ------------------
    Args: 
        -data: the dataframe 
        -column_ax0, column_ax1, column_ax2: here we must enter the columns or variables we want to represent
        -column_hue0, column_hue1, column_hue2: here we must enter the value we want to represent within the coutplot graphs, 
        -title0, title1, title2 : in string format, here we write the titles of each subplot
        -palette1, palette2 and palette3:  these are the color palettes for each subplot
    '''

    fig, axes = plt.subplots(1, 3,  figsize=(20, 8))
    a = sns.countplot(df[column_ax0], hue=df[column_hue0], ax=axes[0], palette= palette1)
    axes[0].set_title(title0)
    a.set_xticklabels(a.get_xticklabels(), rotation=45)
    b = sns.countplot(df[column_ax1], hue=df[column_hue1], palette=palette2, ax=axes[1])
    axes[1].set_title(title1)
    b.set_xticklabels(b.get_xticklabels(), rotation=45)
    c = sns.countplot(df[column_ax2], hue=df[column_hue2], palette=palette3, ax=axes[2])
    axes[2].set_title(title2)
    c.set_xticklabels(c.get_xticklabels(), rotation=45)
    plt.show()




def show_roc_curve(y, y_pred, style = 'seaborn', figsize=(10,5), extra_title = ''):
    ''' 
    This function plots the ROC curve for a classification model predicts 
    ------------------ 
    Args:
        y (pandas.Series): The real target values.
        y_pred (pandas.Series): The target values predicted by the model.
        style (str): Here we define the graph style we want to use. 
            To see all the available styles, please type this: print(plt.style.available)
        figsize (tuple): Here we define the graph size, it has to be a tuple with 2 values
        extra_title (str): An extra text added to the title
    ------------------
    Returns:
        None
    '''

    fpr, tpr, thresholds = roc_curve(y, y_pred, )

    plt.style.use(style)
    plt.figure(figsize=figsize)

    ax = sns.lineplot(fpr, tpr)
    ax.set(xlim = [0.0, 1.0],
           ylim = [0.0, 1.0],
           title = 'ROC curve ' + extra_title,
           xlabel = 'False Positive Rate (1 - Specificity)',
           ylabel = 'True Positive Rate (Sensitivity)',
    )
    plt.show()

