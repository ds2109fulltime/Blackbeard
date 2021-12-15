#Libraries:
import pandas as pd
import numpy as np
import datetime
import calendar
import requests
import os, sys 
import cv2 as cv
from sklearn import preprocessing
from itertools import combinations
from collections import Counter
from itertools import combinations
from collections import Counter

"""
Get_root_path
"""
#
#Body of the function:
#
def get_root_path(n):
    '''
    This function allows us to iterate over folders to add the path of our root folder
    Params:
    - n (int): the number of times we will iterate to reach the desired folder
    '''
    path = os.path.dirname(os.path.abspath(__file__)) 
    for i in range(n):
        path = os.path.dirname(path)
    print(path)
    sys.path.append(path)


#
#Body of the function:
#
def delete_duplicates(df, column_names = None):
    """
    Delete the rows with duplicated values in a dataframe. 
    Arguments: 
    - df : dataframe
    - column_names : names of the columns to take into account
    Returns the dataframe without the duplicated values and prints the number of deleted rows.
    """   
    quant_dup_rows = np.size(df[df.duplicated(subset= column_names)].index)  # Get the number of duplicated rows
    print(f"{quant_dup_rows} rows have been deleted")   # Print the number of duplicated rows

    return df.drop_duplicates(subset= column_names)


"""
Function to deal with missing values in dataframes
"""
#
#Body of the function:
#
def missing_values(df, method = 'drop', column_names = None):
    """
    Function to deal with missing values in dataframes. Three different options: 1 - Drop the rows with missing values, 
    2 - Fill the missing values with zeros, 3 - Fill the missing values with the average of the column values.
    Arguments: 
        - df : the given dataframe
        - method = 'drop' : drop the missing values, 'zero' : fill with zeros, 'avg' : fill with the average
            Default: 'drop' // 
        - column_names --> the columns to take into account
    Return: The resulting dataframe.
    """   
    if method == 'drop':
        df_res = df.dropna(subset = column_names)   # Drop the rows with missing values in the given columns
    elif method == 'zero':
        for name in column_names:
            df[name] = df[name].fillna(0)    # Fill missing values with zeros
        df_res = df
    else:
        for name in column_names:
            df[name] = df[name].fillna(df[name].mean())   # Fill missing values with the average
        df_res = df  

    return df_res

"""
Save the names of columns in a dataframe and export it to csv.
"""
#
#Body of the function:
#
def save_col_names(data):
    """
    Save the names of the columns of a given dataset in a dataframe and export it to csv. 
    Arguments: data --> the given dataframe
    Return: The new dataframe containing the column names.
    """   
    column_names = pd.DataFrame({'Names':data.columns})   # Creates a dataframe with the column names
    column_names.to_csv('column_names.csv', sep = ';')   # Exports the column names to csv

    return column_names


"""
Unix_to_UTC
"""
#
#Body of the function:
#
def Unix_to_UTC(unixdate):
    '''
    This Function transforms data in UNIX time format to UTC time format.
    Arguments: 
    - unixdate : Date in Unix format in interger type
    Unix time is a system for describing a point in time, 
    it is the number of seconds that have elapsed since the Unix epoch,
    The Unix epoch is 00:00:00 UTC on 1 January 1970. 
    IMPORTANT --> UNIX input has to have less than 11 digits
    '''
    
    int(unixdate)
    if unixdate == unixdate>10000000000:
        print('UNIX input has to have less than 11 digits')
    else:
        return datetime.datetime.utcfromtimestamp(unixdate).strftime('%Y-%m-%d:%H:%M')

"""
Split_datetime
"""
#
#Body of the function:
#
def download_csv(url):
    '''
    This Functions takes a URL with a CSV file and downloads it in the same folder the function is being run.
    The only required parameter is the URL itself.
    '''
    req = requests.get(url)
    url_content = req.content

"""
Split_datetime
"""
#
#Body of the function:
#
def split_datetime(df, column, only_date = False):
    
    '''
    This function receives a dataframe with a column with values 'datetime' look like but in 'object' format and transforms it
    into 'datetime' format and divide every component in different columns to operate with the values. In that way, we will
    have days, months, years, hours and minutes in different columns.
    Parameters:
    - df: dataframe
    - column: dataframe's column with the values we want to transform. Tendrá la forma 'df[datetime]'
    - only_date: 'True' if there isn't time and 'False' if there is date and time in the column.
    '''
    column = pd.to_datetime(column)
    time = pd.DataFrame(column)

    time['day'] = column.dt.day
    time['month'] = column.dt.month
    time['year'] = column.dt.year
    time['weekday'] = column.dt.weekday

    if only_date == False:
        time['hour'] = column.dt.hour
        time['minutes'] = column.dt.minute
    
    df = pd.concat([df, time], axis = 1)
    df = df.loc[:,~df.columns.duplicated(keep='last')]
    
    return df

"""
Max_min 
"""
#
#Body of the function:
#
def  max_min (df,column):
    '''
    This Function receives a dataframe and a specific column of said dataframe, 
    it prints the minimun and maximun value in said column.
    Params: 
    - df : dataframe name
    - column : column that you want to iterate on.
    '''
    a = df[column].min()
    print("The minimun value of this column is:", a)
    b = df[column].max()
    print("The maximun value of this column is:", b)

"""
Nulos_to_0
"""
#
#Body of the function:
#
def null_to_0 (df):
    '''
    This Function receives a dataframe and returns the same dataframe with 
    all the NaN values modified to the interger 0.
    Params: 
    - df : dataframe name
    '''
    df = df.fillna(0)
    return df

"""
Nulos_media
"""
#
#Body of the function:
#
def null_to_mean (df,column):
    '''
    This Function receives a dataframe and a specific column of said dataframe,
    and returns the same dataframe with all the NaN values modified 
    to the mean value of that column.
    Params: 
    - df : dataframe name
    - column : column that you want to iterate on.
    '''
    mean_df = df[column].mean()
    df = df.fillna(mean_df)
    return df

"""
Numeric
"""
#
#Body of the function:
#

def categorical_numeric(df, column):
    '''This function is to change the categorical values ​​to numeric values ​​of a concrete column.
    Params:
    - df = DataFrame
    - columns = column to modify
    '''
    le = preprocessing.LabelEncoder()
    for i in column:
        le.fit(df[i])
        df[i] = le.transform(df[i])
    return column

"""
Df_information
"""
#
#Body of the function:
#
def df_information(df):
    '''
    Function to see all the relevant information of the DataFrame
    '''
    print(df.info())
    print(df.describe())
    print(df.corr())
    print(df.head())

"""
Merge_df
"""
#
#Body of the function:
#
def merge_df(df_1, df_2, column):
    '''
    Function to unify 2 DataFrames
    Params:
    - df_n : Each of the Data Frames
    - column : Same column in both dataframes
    '''
    df3 = pd.merge(df_1, df_2, column)
    return df3

"""
Drop_NaN
"""
#
#Body of the function:
#
def drop_NaN(df, column):
    '''This function eliminates the NaN values of a specific column
    Params:
    - df = DataFrame
    - column = columna de la que queremos eliminar los valorers NaN
    '''
    column.replace(0,np.nan)
    df.dropna(axis=0, inplace=True)

"""
Deleate_column
"""
#
#Body of the function:
#
def deleate_column(df,column):
    '''
    Deleting columns and automatically saving to the dataframe
    Params:
    - df:pandas Dataframe
    - col: column to delete
    - axis:1
    - inplace:True
    '''
    df.drop(column,axis=1,inplace=True)

"""
Replace_by_numpy
"""
#
#Body of the function:
#
def replace_by_numpy(df,column,string, value):
    '''
    Replacing string or number with a new value
    Params:
    - df:pandas Dataframe
    - col:column where is the value to replace
    - string: value to replace
    - value:new value
    '''
    df[column] = np.where(df[column] == string, value,df[column])
    return 

"""
Replace_string
"""
#
#Body of the function:
#
def replace_string(df, column, string_list, replacement):
    '''
    Run through list of strings to replace for a value.
    Params:
    - df: pandas Dataframe
    - col: replaced column
    - string_list: list of strings to be runned
    - replacement: new value
    '''
    for pos, val in enumerate(string_list):
        df[column] = df[column].str.replace(string_list[pos], replacement)

"""
Percentage to decimal column
"""
#
#Body of the function:
#
def percentage(df,column):   
    '''
    This functions become your column with % to decimal column of your data.
    Params:
    - df: dataframe
    - column: column to be represented
    '''
    x = df[column].str.replace('%', '').astype(float)
    df[column] = x/100
    return df[column]

"""
Sort_column
"""
#
#Body of the function:
#
def sort_column(df, column):
    '''
    This function sorts a DataFrame according to the selected column. 
    You only have to indicate the columns that you want to make up the dataframe.
    Params:
    - df : Dataframe 
    - column : column you want to modify
    '''
    df = df.sort_values(by=column,ascending=False)
    return df

"""
Outliers_in_column
"""
#
#Body of the function:
#
def outliers (df, column):
    '''
    Function that removes outliers from the given column. 
    Params: 
        - df : Dataframe 
        - column : column you want to modify
    '''
    mean = df[column].mean() 
    std = df[column].std() 
    values = []
    for i in df[column]:
        if i >= (mean + (2*std)):
            i == mean
        values.append(i)
    return values

"""
Get the number of Sales by Category 
"""
#
#Body of the function:
#
def sales_by_category(df, column_1, column_2):
    ''' 
    It will sort values by descending order, from most to least, and top 10. 
    It will create another dataframe that will show column_1, column you want to groupby, 
    column_2 is the number of sales based on column_1 groupby.
    Params: 
    - df : dataframe name
    - column_1 : column that you want to groupby, could be country, state, product 
    - column_2 : column which will indicate a number of sales
    '''
    sales_by_category = df[[column_1,column_2]].groupby([column_1]).count().sort_values(column_2, ascending=False).head(10)
    sales_by_category = sales_by_category.rename(columns={column_2: 'Number of Sales'})
    sales_by_category = sales_by_category.reset_index()
    sales_by_category

    return sales_by_category

"""
From_month_into_to_name
"""
#
#Body of the function:
#
def month_to_name(df, column): 
    '''
    It will return a values of Month in name format for example, 03 -> March 
    Params: 
    - df : dataframe
    - column: should be month, in datatime format
    '''
    df[column] = df[column].apply(lambda x: calendar.month_abbr[x])
    return df

"""
Columns_dtype
"""
#
#Body of the function:
#
def columns_dtype(df, include='all', cardinality_threshold=None, return_df=False):
    '''
    Returns categorical, numerical or both kind of variables based on their 
    dtype, list-like or DataFrame, based on the user specified parameter.
    Params:
        - df: Pandas DataFrame
            Pandas DataFrame where we want to analyze the variables.
        - include: str, default 'all'
            String that indicates which dtype the user wants to return.
            'all' returns both numerical and categorical variables.
            'categorical' returns only categorical variables.
            'numerical' returns only numerical variables.
        - cardinality_threshold: int, default None
            Number of maximum unique values that the categorical variable 
            has to be. The variables with higher cardinality that the 
            specified by this parameter will be ignored.
        - return_df: bool, default False
            Whether the return will be a DataFrame or not. If True, returns 
            the Pandas DataFrame with only the specified variables. If False, 
            returns a list for each of the specified variables.
    Returns:
        - List of variables or DataFrame, depending of the return_df parameter.
    '''

    # Capturing the variables depending of the user's selection

    # Include All
    if include == 'all':
        if cardinality_threshold:
            categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and df[col].nunique() <= cardinality_threshold]
        else:
            categorical_cols = [col for col in df.columns if df[col].dtype == 'object']

        numerical_cols = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
        data_columns = categorical_cols + numerical_cols

        if return_df:
            return df[data_columns]

        return categorical_cols, numerical_cols

    # Include categorical only
    elif include == 'categorical':
        if cardinality_threshold:
            categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and df[col].nunique() <= cardinality_threshold]
        else:
            categorical_cols = [col for col in df.columns if df[col].dtype == 'object']

        if return_df:
            return df[categorical_cols]

        return categorical_cols

    # Include numerical only
    elif include == 'numerical':
        numerical_cols = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]

        if return_df:
            return df[numerical_cols]

        return numerical_cols

"""
Get_products_that_are_sold_together
"""
#
#Body of the function:
#
def get_products_that_are_sold_together(df, order_id_column, product_column, products_bought_together_column):
    '''
    This function will return a dataframe along with number of transactions of products
    that were purchased together. 
    Params: 
        df : dataframe that will be used 
        order_id_column : is the transaction id, could be order ID. 
        product_column : is the column for products 
        products_bought_together_column : the column that will two products purchased together
    '''
    # creating a new dataframe to separate duplicates from Order ID
    df_updated = df[df[order_id_column].duplicated(keep=False)]
    #Joining the products with the same Order ID group to be on the same line 
    df_updated[products_bought_together_column] = df_updated.groupby(order_id_column)[product_column].transform(lambda x: ','.join(x))
    #Getting rid of the duplicate values 
    df_updated = df_updated[[order_id_column, products_bought_together_column]].drop_duplicates()

    count = Counter()

    for row in df_updated[products_bought_together_column]:
        row_list = row.split(',')
        count.update(Counter(combinations(row_list,2)))

    # Below Will return products bought together along with the number of transactions
    products_bought_together = count.most_common(20)
    products_bought_together_df = pd.DataFrame(products_bought_together, columns=['Product_bundle', 'Number of transactions'])
    return products_bought_together_df


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Images

"""
Frames_from_video
"""
#
#Body of the function:
#
def frames_from_video(path_in, path_out, ms_extract = 1000):
    '''
    This function extracts frames from videos in an explicit path, modifies the resolution of the frames and, finally, 
    it saves them in another path.
    This function works well on windows, 
    in MAC it is recommended that you delete the file that is generated automatically (.DS_Store).
    Parameters:
    - path_in: Path of the folder where videos are.
    - path_out: Path of the folder where frames will be saved.
    - ms_extract: miliseconds between frames.
    '''
    filenames = os.listdir(path_in)
    count = 0
    for filename in filenames:
        vidcap = cv.VideoCapture(path_in + '\\' + filename)
        print(path_in + '\\' + filename)
        counter = 0
        while True:
            vidcap.set(cv.CAP_PROP_POS_MSEC,(counter*ms_extract))
            success,image = vidcap.read()
            if success:
                print ('Read a new frame: ', success, count)
                imagesmall = cv.resize(image, (int(image.shape[1]*0.5), int(image.shape[0]*0.5)))                                         
                cv.imwrite( path_out + "\\frame_{}.jpg".format(count), imagesmall)
                count = count + 1
                counter = counter + 1
            else:
                print('***********************')
                print('All frames catched.')
                print('***********************')
                break

    return 'All videos have been catched'



"""
Images_dataset_properties
"""
#
#Body of the function:
#
def images_dataset_properties(images):
    '''
    This function receives an array of images and returns its .ndim, .shape y .size
    This function works well on windows, 
    in MAC it is recommended that you delete the file that is generated automatically (.DS_Store).
    Parameters:
     - images: array with the images. It is an iterable.
    '''
    print("Dimensions:",images.ndim)
    print("Shape:",images.shape)
    print("Size:",images.size)


"""
Read_images
"""
#
#Body of the function:
#
def read_images(path):
    '''
    This function reads an image from a path.
    Parameters:
    - path: Path where the image is.
    '''
    image = cv.imread(path)
    return image

"""
Flip_images
"""
#
#Body of the function:
#
def flip_images(paths_in, paths_out):
    '''
    This function receives the paths where the images are and flip them (mirror effect).
    This function works well on windows, 
    in MAC it is recommended that you delete the file that is generated automatically (.DS_Store).
    Parameters:
    - paths_in: Paths of the folders where the images are. It is an iterable.
    - paths_out: Paths of the folders where the images will be saved. It is an iterable.
    '''

    for i in range(len(paths_in)):
        filenames = os.listdir(paths_in[i])
        count = 0
        for file in filenames:
            if file not in os.listdir(paths_out[i]):
                try: 
                    print('Flipping: ', count+1)
                    path_img = paths_in[i] + '\\' + file
                    image = read_images(path_img)
                    flip = cv.flip(image, 1)
                    cv.imwrite(paths_out[i] + '\\' + file, flip)
                    count = count + 1
                except:
                    break
            else:
                pass

        print('***********************')
        print('Images in folder {} fipped.'.format(paths_in[i]))
        print('***********************')


"""
Resize_images
"""
#
#Body of the function:
#
def resize_images(images, height, width):
    '''
    This function receives images and returns them resized with the specified dimensions.
    This function works well on windows, 
    in MAC it is recommended that you delete the file that is generated automatically (.DS_Store).
    Observacion
    Parameters:
- images: array of images. It is an iterable. If we have just an image, it must be written as 'np.array([image])'
    - height: height in ppp
    - width: width in ppp
    '''
    resized_images = []

    for image in images:
        resized = cv.resize(image, (height, width))
        resized_images.append(resized)
    
    return np.array(resized_images)


"""
Edit_images
"""
#
#Body of the function:
#
def edit_images(images, alpha = 1, beta = 0):
    '''
    This function modifies contrast and bright of an array of images.
    This function works well on windows, 
    in MAC it is recommended that you delete the file that is generated automatically (.DS_Store).
    Parámetros:
    - images: array of images. It is an iterable. If we have just an image, it must be written as 'np.array([image])'
    - alpha: (1.0-3.0). Contrast controller. The higher it is, the higher is the contrast.
    - beta: (0-100). Bright controller. The higher it is, the higher is the bright.
    '''
    edited_images = []

    for image in images:

        adjusted = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
            
        edited_images.append(adjusted)
    
    return np.array(edited_images)

"""
Color2gray
"""
#
#Body of the function:
#
def color2gray(images):
    '''
    Esta función recibe las imágenes en un iterable y las devuelve en escala de grises.
    This function works well on windows, 
    in MAC it is recommended that you delete the file that is generated automatically (.DS_Store).
    Parámetros:
    - images: array of images. It is an iterable. If we have just an image, it must be written as 'np.array([image])'
    '''
    grey_images = []

    for image in images:
        grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        grey_images.append(grey)
    
    return np.array(grey_images)


"""
Negative_colors_images
"""
#
#Body of the function:
#
def negative_colors_images(images): 
    '''
    This function receives an array of images and returns it with the negative colors.
    This function works well on windows, 
    in MAC it is recommended that you delete the file that is generated automatically (.DS_Store).
    Parameters:
    - images: array of images. It is an iterable. If we have just an image, it must be written as 'np.array([image])'
    '''
    negative_images = []

    for image in images:
        negative = 255 - image
        negative_images.append(negative)
    
    return np.array(negative_images)


"""
Monocolor_images
"""
#
#Body of the function:
#
def monocolor_images(images, color = 'blue'):
    '''
    This function receives an array of images and the color channel (red, green o blue) and returns them into this channel.
    Cambia a azul ('blue') si no se especifica.
    This function works well on windows, 
    in MAC it is recommended that you delete the file that is generated automatically (.DS_Store).
    Parameters:
    - images: array of images. It is an iterable. If we have just an image, it must be written as 'np.array([image])'
    - color: color channel we want to get (red, green, blue).
    '''
    monocolor_images = []

    for image in images:

        if color == 'blue':
            b = image.copy()
            b[:,:,0] = b[:,:,1] = 0
            channel = b
        
        elif color == 'green':
            g = image.copy()
            g[:,:,0] = g[:,:,2] = 0
            channel = g
        
        elif color == 'red':
            r = image.copy()
            r[:,:,1] = r[:,:,2] = 0
            channel = r
        
        else:
            print('Error: Chose the correct color.')
            break
        
        monocolor_images.append(channel)

    return np.array(monocolor_images)


"""
Images_load_data
"""
#
#Body of the function:
#
def images_load_data(path_in):
    '''
    This function loads the images from a list of specific locations and assigns them to the variables 'images' and 'labels', 
    so that it returns the 'train' and the 'tests' of each of them in different variables.
    The return would be (X_train, y_train), (X_test, y_test).
    For this to work, the folders must be organized as follows:
        for the train: '../ubication/train_images/' and inside will be the images of each class divided into folders, for example: buildings, forest, mountain, beach.
        for the test: '../ubication/test_images/' and inside will be the images of each class divided into folders, for example: buildings, forest, mountain, beach.
    Params:
    - path_In: List with all locations, It is an iterable.
    '''
    class_names = os.listdir(path_in[0])
    class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

    output = []
    
    # Iterate through training and test sets
    for dataset in path_in:
        
        images = []
        labels = []
        
        print("Loading {}".format(dataset))
        
        # Iterate through each folder corresponding to a category
        for folder in os.listdir(dataset):
            label = class_names_label[folder]
            
            # Iterate through each image in our folder.
            for file in os.listdir(os.path.join(dataset, folder)):
                
                # Get the path name of the image
                img_path = os.path.join(os.path.join(dataset, folder), file)
                
                # Open and resize the img
                image = cv.imread(img_path)
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                image = cv.resize(image, (int(image.shape[1]*0.6), int(image.shape[0]*0.6))) 
                
                # Append the image and its corresponding label to the output
                images.append(image)
                labels.append(label)
                
        images = np.array(images, dtype = 'float32')
        labels = np.array(labels, dtype = 'int32')   
        
        output.append((images, labels))

    return output

"""
Images_load_data
"""
#
#Body of the function:
#
def save_images(image, path_out, image_name):
    '''
    This function saves specified image
    Params:
    - image: image to save
    - path_out: Path of the folder where the image will be saved.
    - image_name: name with the image will be saved
    '''
    cv.imwrite(path_out + "\\" + image_name + ".jpg", image)

    