# Libraries:
from tensorflow import keras
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,precision_score,recall_score,f1_score,r2_score,mean_squared_error,mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from supervised.automl import AutoML
import pandas as pd
import numpy as np
import cv2 as cv
import imageio
import urllib.request

# Functions
def grid_search(X_train, y_train):
    
    '''
    This function performs a grid search with RandomForestClassifier, LogisticRegression and SVC
    to find the best parameters after making train_test_split.
    '''

    pipe = Pipeline(steps=[
        ('classifier', RandomForestClassifier())
    ])

    logistic_params = {
        'classifier': [LogisticRegression()],
        'classifier__penalty': ['l1', 'l2']
    }

    random_forest_params = {
        'classifier': [RandomForestClassifier()],
        'classifier__max_features': [1,2,3]
    }

    svm_param = {
        'classifier': [SVC()],
        'classifier__C': [0.001, 0.1, 0.5, 1, 5, 10, 100],
    }

    search_space = [
        logistic_params,
        random_forest_params,
        svm_param
    ]

    clf = GridSearchCV(estimator = pipe,
                    param_grid = search_space,
                    cv = 10)

    clf.fit(X_train, y_train)
    print(clf.best_estimator_)
    print(clf.best_score_)
    print(clf.best_params_)

#-----------------------------------

def models(X_train, X_test, y_train, y_test):

    '''This function compares and prints results of various ML models after doing train_test_split.'''

    models = []
    models.append(['XGBClassifier',XGBClassifier(learning_rate=0.1,objective='binary:logistic',random_state=0,eval_metric='mlogloss')])
    models.append(['Logistic Regression',LogisticRegression(random_state=0)])
    models.append(['SVC',SVC(random_state=0)])
    models.append(['KNeigbors',KNeighborsClassifier()])
    models.append(['GaussianNB',GaussianNB()])
    models.append(['DecisionTree',DecisionTreeClassifier(random_state=0)])
    models.append(['RandomForest',RandomForestClassifier(random_state=0)])
    models.append(['AdaBoostClassifier',AdaBoostClassifier()])
    lst_1 = []
    for m in range(len(models)):
        lst_2 = []
        model = models[m][1]
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test,y_pred)
        accuracies = cross_val_score(estimator= model, X = X_train, y = y_train, cv=10)

    # k-fOLD Validation
        roc = roc_auc_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        f1 = f1_score(y_test,y_pred)
        print(models[m][0],':')
        print(cm)
        print('Accuracy Score: ',accuracy_score(y_test,y_pred))
        print('')
        print('K-Fold Validation Mean Accuracy: {:.2f} %'.format(accuracies.mean()*100))
        print('')
        print('Standard Deviation: {:.2f} %'.format(accuracies.std()*100))
        print('')
        print('ROC AUC Score: {:.2f} %'.format(roc))
        print('')
        print('Precision: {:.2f} %'.format(precision))
        print('')
        print('Recall: {:.2f} %'.format(recall))
        print('')
        print('F1 Score: {:.2f} %'.format(f1))
        print('-'*40)
        print('')
        lst_2.append(models[m][0])
        lst_2.append(accuracy_score(y_test,y_pred)*100)
        lst_2.append(accuracies.mean()*100)
        lst_2.append(accuracies.std()*100)
        lst_2.append(roc)
        lst_2.append(precision)
        lst_2.append(recall)
        lst_2.append(f1)
        lst_1.append(lst_2)

#-----------------------------------

def NN_conv_model(layers, optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy']):
    
    '''
    This function receives the 'layers', 'optimizer', 'loss' y 'metrics' which are necessaries to create a NN or a CNN.
    Parameters:
    - layers: of the NN or CNN
    - optimizer: optimizer of the model. 'adam' by default
    - loss: loss function. 'sparse_categorical_crossentropy' by default.
    - metrics: metrics for the evaluation. It is an iterable. ['accuracy'] by default.
    
    Example of layers for a CNN:
    
    layers = [keras.layers.Conv2D(first_layer_conv, (3,3), activation=activation, input_shape=image_size),
            keras.layers.MaxPooling2D(pool_size=(2,2)),
            keras.layers.Conv2D(second_layer_conv, (3,3), activation=activation),
            keras.layers.MaxPooling2D(pool_size=(2,2)),
            keras.layers.Flatten(),
            keras.layers.Dense(first_layer_NN, activation=activation),
            keras.layers.Dense(second_layer_NN, activation=activation),
            keras.layers.Dense(len(class_names_label), activation=final_activation)
        ]
    '''

    model = keras.Sequential(layers)

    model.compile(optimizer = optimizer,
                loss = loss,
                metrics = metrics
                )
    
    return model


def model_train(model, X, y, batch_size= 64, epochs= 15, validation_split= 0.1, shuffle_ = False, conv = False):

    '''
    This function receives a model and trains it. Then, it returns the history of the trained model.
    Variables:
    - model: prediction model. It must be a NN or a CNN.
    - X: training values. It is an iterable.
    - y: training labels. It is an iterable.
    - batch_size: batch_size parameter of the NN. 64 by default.
    - epoch: epochs for the training model. 15 by default.
    - validation_split: portion for the validation. 0.1 by default.
    - shuffle_: True for making a 'shuffle' of the values. False by default.
    - conv: False if it is a NN and True if it is a CNN (images). False by default.
    '''

    if shuffle_ == True:
        X, y = shuffle(X, y, random_state=42)
    
    if conv == True:
        X = X/255

    history = model.fit(X,
                    y,
                    batch_size = batch_size,
                    epochs = epochs,
                    validation_split = validation_split
                    )
    return history


def load_model(path):
    
    '''
    The function loads a keras model of the path provided.
    Parameters:
    - path: path of the model.
    '''

    model = keras.models.load_model(path)

    return model


def save_model(model, path, model_name):

    '''
    The function saved the keras model received in the path provided.
    Parameters:
    - model: the trained model
    - path: saving path
    - model_name: name of the model
    '''

    model.save(path + '\\' + model_name + '.h5')

#-----------------------------------

def basic_regression_models(model, X_train, y_train, degree = None, alpha = None, l1_ratio = None):
    
    '''This function receives the type of model and 'X','y' train values to train the basic regression models.
    Parameters:
     - model: the type of model the user wants to train. Values: LinearRegression, PolynomialRegression, Ridge, Lasso, ElasticNet.
     - X_train: X values for training. It has to be an iterable.
     - y_train: yvalues for training. It has to be an iterable.
     - degree: just in case of model = PolynomialRegression. It must be an Integer.
     - alpha: just in case of model = Ridge, Lasso or ElasticNet.
     - l1_ratio: just in case of model = Elasticnet.
     '''
    
    if model == 'LinearRegression':
        lm = LinearRegression()
        trained_model = lm.fit(X_train, y_train)
    
    elif model == 'PolynomialRegression':
        # Preprocessing
        poly_feats = PolynomialFeatures(degree = degree)
        poly_feats.fit(X_train)
        X_poly = poly_feats.transform(X_train)

        # Train
        pol_reg = LinearRegression()
        trained_model = pol_reg.fit(X_poly, y_train)
    
    elif model == 'Ridge':
        ridge = Ridge(alpha = alpha)
        trained_model = ridge.fit(X_train, y_train)
    
    elif model == 'Lasso':
        lasso = Lasso(alpha = alpha)
        trained_model = lasso.fit(X_train, y_train)
    
    elif model == 'ElasticNet':
        elastic = ElasticNet(alpha = alpha, l1_ratio = l1_ratio)
        trained_model = elastic.fit(X_train, y_train)
    
    else:
        print('Error. Chose one of these models: LinearRegression or PolynomialRegression')
    
    return trained_model

#-----------------------------------

def regression_errors(y, y_pred):

    '''
    This function receives the parameters 'y', 'y_pred' and prints the MAE, MSE, RMSE and R2 score.
    Parameters:
    - y: y values for the test
    - y_pred: predicted values from X values for the test
    '''

    print('MAE:', mean_absolute_error(y, y_pred))
    print("MSE:", mean_squared_error(y, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y, y_pred)))
    print('R2 score', r2_score(y, y_pred))


def print_regress_metrics(y, y_pred):
    ''' 
    This function print the plot the R^2, MAE, MSE, RMSE and MAPE score of a regression model.
    Args:
        y (pandas.Series): The real target values.
        y_pred (pandas.Series): The target values predicted by the model.
    
    Returns:
        None
    '''

    print("R^2 score:", round(r2_score(y_pred, y), 4))
    print("MAE score:", round(mean_absolute_error(y_pred, y), 4))
    print("MSE score:", round(mean_squared_error(y_pred, y), 4))
    print("RMSE score:", round(np.sqrt(mean_squared_error(y_pred, y)), 4))
    y_array, y_pred_array = np.array(y), np.array(y_pred)
    mape = np.mean(np.abs((y_array - y_pred_array) / y_array)) * 100
    print(f'MAPE score: {round(mape, 4)} %')


def print_classif_metrics(y, y_pred):
    ''' 
    This function print the plot the accuracy, recall, precision, F1 score and AUC
        of a classification model.
    Args:
        y (pandas.Series): The real target values.
        y_pred (pandas.Series): The target values predicted by the model.
    
    Returns:
        None
    '''

    print(f'Accuracy score: {round(accuracy_score(y_pred, y), 3)} %')
    print(f'Recall score: {round(recall_score(y_pred, y), 3)} %')
    print(f'Precision score: {round(precision_score(y_pred, y), 3)} %')
    print(f'F1 score: {round(f1_score(y_pred, y), 3)} %')
    print(f'AUC: {round(roc_auc_score(y_pred, y), 3)} %')

#-----------------------------------

def split_and_scale(data, target_col_name=None, scaling_method='standard', time_series=False, test_size=0.2):

    '''
    Split the data into train and validation data. If the data 
    will train a model for time series, the split will not have 
    shuffling. Once the data is splitted, scale the data (only if 
    the data is not time series data). The user must choose by 
    parameter the scaling method (standard or min-max scaling).
    Params:
        - data: expected Pandas DataFrame.
        - target_col_name: str, default None
            name of the target variable.
        -  scaling_method: str, default 'standard'
            allowed 'minmax' or 'standard'.
        - time_series: bool, default False
            if True, shuffle and scaling will be False.
        - test_size: float, default 0.2
            percentage size of the validation data (0 to 1).
    Returns:
        - Splitted —and scaled, if selected— data. 
    
    '''

    # Getting Features and Target
    X = data.drop(target_col_name, axis=1)
    y = data[target_col_name]

    if time_series:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False)
        
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=True)

        # Scaling the data
        if scaling_method == 'standard':
            scaler = StandardScaler()

            # Scaling the data (Standard Scaler)
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()

            # Scaling the data (MinMax Scaler)
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

#-----------------------------------

def categorical_encoding(data, kind='label_encoding'):

    '''
    Function to encode categorical variables.
    Encodes categorical data from a given DataFrame with one of the 
    following methods: Label Encoding or One Hot Encoding. This function 
    makes use of the scikit-learn library.
    Params:
        - data: expected Pandas DataFrame
        - kind: str, default 'label_encoding'
            one of both 'label_encoding' or 'one_hot_encoding' must be passed
    Returns:
        - Pandas DataFrame with categorical data encoded
    '''

    # Checking if kind variable is correct:
    if kind not in ['label_encoding', 'one_hot_encoding']:
        raise ValueError ('"kind" parameter must be "label_encoding" or "one_hot_encoding"')

    # Checking the encoding
    if kind == 'label_encoding':
        le = LabelEncoder()
        categorical_columns = [col for col in data.columns if data[col].dtype == 'object']
        for column in categorical_columns:
            data[column] = le.fit_transform(data[column])
        
        # Returning label-encoded data
        return data

    elif kind == 'one_hot_encoding':

        # Getting categorical columns
        categorical_columns = [col for col in data.columns if data[col].dtype == 'object']

        oh = OneHotEncoder(handle_unknown='ignore', sparse=False)

        oh_encoded_dataframe = pd.DataFrame(oh.fit_transform(data[categorical_columns]))
        # Putting back the indexes
        oh_encoded_dataframe.index = data.index
        # Removing categorical columns to replace with onehot-encoded columns
        numeric_data = data.drop(categorical_columns, axis=1)
        # Adding categorical columns to numerical features
        onehot_encoded_data = pd.concat([numeric_data, oh_encoded_dataframe], axis=1)

        # Returning onehot-encoded data
        return onehot_encoded_data

#-----------------------------------

def automl(X_train,y_train,X_test,y_test):

    '''
    This function receives the train and test data and makes an AutoML to compare different prediction model
    with some default values
    '''

    my_automl = AutoML(eval_metric='accuracy')
    my_automl.fit(X_train,y_train)
    preds = my_automl.predict(X_test)
    accuracy_score(preds, y_test)
    return accuracy_score

#-----------------------------------

def captain():
    
    '''
    For calling best function ever you must have installed the libraries 'imageio' and 'urllib'.
    Press 'q' to exit from our Captain.
    '''

    print("Fuck Blackbeard... I'm the captain.")

    url = "https://64.media.tumblr.com/bba725c6dd9e1fdcacc651925c44d0d5/b6dc49731ce9499b-60/s640x960/1192587ef5ad3412f4020077a0c5ae51a387392d.gifv"
    fname = "CAPTAIN Jack Sparrow, please.gif"

    ## Read the gif from the web, save to the disk
    imdata = urllib.request.urlopen(url).read()
    imbytes = bytearray(imdata)
    open(fname,"wb+").write(imdata)

    ## Read the gif from disk to `RGB`s using `imageio.miread` 
    gif = imageio.mimread(fname)
    nums = len(gif)
    print("Total {} frames in the gif!".format(nums))

    # convert form RGB to BGR 
    imgs = [cv.cvtColor(img, cv.COLOR_RGB2BGR) for img in gif]

    ## Display the gif
    i = 0

    while True:
        cv.imshow("gif", imgs[i])
        if cv.waitKey(100)&0xFF==ord('q'):
            break
        i = (i+1)%nums
    cv.destroyAllWindows()