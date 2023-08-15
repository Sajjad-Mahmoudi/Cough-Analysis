import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
import pandas as pd
import matplotlib
from esperity.cough_analysis.utils import avg_2d_features, multi_plot_simple
from esperity.cough_analysis.results_for_all import log_spectrograms, log_melspectrograms, mfccs, spi_readings
import tensorflow_addons as tfa

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class NeuralNetworks:
    def __init__(self, data: list, labels: dict):
        """
        Initializes the NeuralNetworks class.
        :param data: A list of dictionaries that contain the features, to be used later as the input of the neural
        network.
        :param labels: A dictionary of dataframes containing spirometry readings (use the multi_load_spirometry
        function in the utils module).
        """

        self.data = data
        self.labels = labels

    @staticmethod
    def cnn_model(input_shape: tuple):
        """
        Generates CNN model
        :param input_shape: Shape of input set (height, width, num_channel)
        :return model: CNN model
        """

        model = keras.Sequential()

        # 1st conv layer
        model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))  # (50, 315, 32)
        model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))  # (25, 157, 32)
        model.add(keras.layers.BatchNormalization())  # (25, 157, 32)

        # 2nd conv layer
        model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))  # (25, 157, 32)
        model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))  # (13, 78, 32)
        model.add(keras.layers.BatchNormalization())  # (13, 78, 32)

        # 3rd conv layer
        model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))  # (13, 78, 32)
        model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))  # (7, 39, 32)
        model.add(keras.layers.BatchNormalization())  # (7, 39, 32)

        # flatten output and feed it into dense layer
        model.add(keras.layers.Flatten())  # (16368)
        model.add(keras.layers.Dense(64, activation='relu'))  # (64)
        model.add(keras.layers.Dropout(0.3))  # (64)

        # output layer (2 neurons for R5Hz and Frez)
        model.add(keras.layers.Dense(1, activation=None))  # (1)
        # model.add(keras.layers.Dense(1, activation=None))  # (1)

        return model

    def fuse_features(self, two_dim_features_dicts=None):
        """
        Fuse the different 2D features with common entity into a new dictionary with the same entity keys and
        averaged features (for more info, see the functions average_arrays and avg_2d_features).

        Parameters:
        two_dim_features_dicts (list): A list of dictionaries containing a 2D feature (e.g. spectrogram) where the keys
        are strings of the form "XX_YY_ZZ", where "XX" is the patient number, "YY" is the intervention, and "ZZ" is the
        feature number. The values are 2D numpy arrays.

        Returns:
        dict: A dictionary where the keys are strings of the form "XX_YY", and the values are an array from the values
        of all the input dictionaries of all the entities.
        """

        if two_dim_features_dicts is None:
            two_dim_features_dicts = self.data

        # Initialize an empty dictionary to store the concatenated arrays
        concatenated_dict = {}

        # Iterate over the list of input dictionaries
        for d in two_dim_features_dicts:
            # Calculate the average features of each entity of each dictionary using the avg_2d_features function
            avg_dict = avg_2d_features(d)
            # Iterate over the keys in the average dictionary
            for key, value in avg_dict.items():
                # Check if the key is already in the concatenated dictionary
                if key in concatenated_dict:
                    # If the key is already in the dictionary, concatenate the arrays along the first axis
                    concatenated_dict[key] = np.concatenate([concatenated_dict[key], value])
                else:
                    # If the key is not in the dictionary, add it and its value
                    concatenated_dict[key] = value

        return concatenated_dict

    def concatenate_arrays(self, dict_of_arrays=None):
        """
        Concatenate 2D arrays in a dictionary along the first axis

        Parameters:
        -----------
        arrays_dict : dict
            A dictionary where the values are 2D arrays
        Returns:
        --------
        np.ndarray : 3D array
            A 3D array that is the concatenation of all 2D arrays in the input dictionary
        """

        if dict_of_arrays is None:
            dict_of_arrays = self.fuse_features()

        # convert the dictionary values to a list of numpy arrays
        arrays_list = [np.array(array) for array in dict_of_arrays.values()]
        # concatenate the arrays along the first axis
        concatenated_array = np.stack(arrays_list, axis=0)
        return concatenated_array

    def spi_labels_adaptation(self, column_list=None, row_list=None, spi_readings=None):
        """
        Adapts the dataframes of spirometry readings to be used as the labels (removes the redundant columns and
        rearrange the dataframe to have the values of "NC" row in the last row of new dataframes)

        Parameters:
        column_list (list): List of the names of the columns of the new dataframes, defaulted in
        (['intervention', 'R5Hz', 'Frez'])
        row_list (list): List of the names of the interventions to be considered, defaulted in
        (['BA', 'NC', 'C1', 'C2', 'C3', 'C4', 'C5'])
        spi_readings (dict): A dictionary of which the values are dataframes containing spirometry readings

        Returns:
        dict: A new dictionary of which the values are the new dataframes containing only the columns "intervention",
        "R5Hz", and "Frez". The order of intervention column is "BA", "C1", "C2", "C3", "C4", "C5", "NC".
        """

        if row_list is None:
            row_list = ['BA', 'NC', 'C1', 'C2', 'C3', 'C4', 'C5']
        if column_list is None:
            column_list = ['intervention', 'R5Hz', 'Frez']
        if spi_readings is None:
            spi_readings = self.labels

        new_spi_readings = {}
        for key in spi_readings:
            df = spi_readings[key]
            new_df = pd.DataFrame(columns=column_list)
            for row_name in row_list:
                if row_name != 'NC':
                    if row_name in df["intervention"].tolist():
                        new_df = pd.concat([new_df, df[df['intervention'] == row_name][column_list]], ignore_index=True)
            if 'NC' in row_list:
                if 'NC' in df["intervention"].tolist():
                    nc_rows = df[df['intervention'] == 'NC'][column_list]
            new_df = pd.concat([new_df, nc_rows], ignore_index=True)
            new_spi_readings[key] = new_df
        return new_spi_readings

    def spi_labels_array(self, dict_of_dfs=None):
        """
        This function takes a dictionary of spirometry DataFrames as input and returns a vertically stacked array of
        the 'R5Hz' and 'Frez' columns of each DataFrame.
        :param dict_of_dfs: A dictionary of spirometry DataFrames where each key is a string (e.g. "spi_01") and each
        value is a DataFrame containing 'R5Hz' and 'Frez' columns.
        :return: A vertically stacked array (np.float32) of the 'R5Hz' and 'Frez' columns of each DataFrame in
        the input dictionary.
        """

        if dict_of_dfs is None:
            dict_of_dfs = self.spi_labels_adaptation()

        # extract the 'R5Hz' and 'Frez' columns values from all DataFrames in the dictionary
        list_labels = [df[['R5Hz', 'Frez']].values.astype(np.float32) for df in dict_of_dfs.values()]
        return np.concatenate(list_labels)

    def split_dataset(self, test_size, validation_size, X=None, y=None):
        """
        Splits the data and labels into train, validation, and test sets.
        The data is first split into train and test sets using the test_size parameter.
        Then, the train set is split into a train and validation set using the validation_size parameter.
        If data or labels are not provided, they default to self.data and self.labels, respectively.
        :param test_size: (float) Value between 0 and 1 indicating percentage of data set to allocate to test split.
        :param validation_size: (float) Value between 0 and 1 indicating percentage of train set to allocate to
        validation split.
        :param X: (ndarray, optional) The input data to split. Defaults to self.data.
        :param y: (ndarray, optional) The target labels to split. Defaults to self.labels.
        :return X_train: (ndarray) Input training set.
        :return X_validation: (ndarray) Input validation set.
        :return X_test: (ndarray) Input test set.
        :return y_train: (ndarray) Target training set.
        :return y_validation: (ndarray) Target validation set.
        :return y_test: (ndarray) Target test set.
        """

        if X is None:
            X = self.concatenate_arrays()
        if y is None:
            y = self.spi_labels_adaptation()

        # create train, validation and test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

        # add an axis to input sets
        X_train = X_train[..., np.newaxis]
        X_validation = X_validation[..., np.newaxis]
        X_test = X_test[..., np.newaxis]

        return X_train, X_validation, X_test, y_train, y_validation, y_test

    @staticmethod
    def predict(model, X, y):
        """
        Predict a single sample using the trained model
        :param model: Trained estimator
        :param X: Input data
        :param y: Target
        :return: Spirometry reading predicted by the model
        """

        X = X[np.newaxis, ...]  # array shape (1, 50, 315, 1)
        print("\nTrue Value: {}, Predicted Value: {}".format(y, model.predict(X)[0, 0]))


'''    def plot_history(history):
        """
        Plots accuracy/loss for training/validation set as a function of the epochs
        :param history: Training history of the model
        :return:
        """

        fig, axs = plt.subplots(2)

        # create accuracy sublpot
        axs[0].plot(history.history["accuracy"], label="train accuracy")
        axs[0].plot(history.history["val_accuracy"], label="test accuracy")
        axs[0].set_ylabel("Accuracy")
        axs[0].legend(loc="lower right")
        axs[0].set_title("Accuracy eval")

        # create error sublpot
        axs[1].plot(history.history["loss"], label="train error")
        axs[1].plot(history.history["val_loss"], label="test error")
        axs[1].set_ylabel("Error")
        axs[1].set_xlabel("Epoch")
        axs[1].legend(loc="upper right")
        axs[1].set_title("Error eval")

        plt.show()'''

if __name__ == "__main__":
    # absolute_dirpath = os.path.abspath(os.path.dirname(__file__))

    data = [log_spectrograms, log_melspectrograms, mfccs]
    labels = spi_readings

    cnn = NeuralNetworks(data, labels)
    # Fuse the features into a new dictionary using the concatenate_features method
    data = cnn.fuse_features()
    data = cnn.concatenate_arrays(data)
    # Adapt the dataframes of spirometry readings using the spi_labels_adaptation method
    labels = cnn.spi_labels_adaptation(['intervention', 'R5Hz', 'Frez'], ['BA', 'NC', 'C1', 'C2', 'C3', 'C4', 'C5'],
                                       labels)
    # remove the rows C4 and C5 from spi_11 (why? see the add_spirometry method of the unify_features module)
    labels['spi_11'].drop(labels['spi_11'].loc[(labels['spi_11']['intervention'] == 'C4') |
                                               (labels['spi_11']['intervention'] == 'C5')].index, inplace=True)
    # stack all spirometry reading into an array using spi_labels_array method
    labels = cnn.spi_labels_array(labels)

    '''
                                    ****** Training only for Frez ******
    '''

    # get train, validation, and test splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = cnn.split_dataset(0.25, 0.2, data, labels[:, 1])

    # create network
    model_cnn = cnn.cnn_model((X_train.shape[1], X_train.shape[2], X_train.shape[3]))

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model_cnn.compile(optimizer=optimiser,
                      loss='mse',
                      metrics=tfa.metrics.r_square.RSquare(y_shape=(1,))
                      # tfa.metrics.RSquare(dtype=tf.float32, y_shape=(1,))
                      )
    # model_cnn.summary()

    # train model
    history = model_cnn.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=35, epochs=50)

    # plot loss and accuracy for training and validation data
    multi_plot_simple(x=[np.arange(1, 1 + len(np.array(history.history['loss']))),
                         np.arange(1, 1 + len(np.array(history.history['val_loss'])))],
                      y=[np.array(history.history['loss']), np.array(history.history['val_loss'])],
                      titles=['Loss Plot'],
                      labels=['training', 'validation'],
                      label_loc='lower left', num_plot=1, num_row=1, num_col=1, fig_size=(10, 7), x_label='#Epochs',
                      y_label='Mean Squared Error')

    multi_plot_simple(x=[np.arange(1, 1 + len(np.array(history.history['r_square']))),
                         np.arange(1, 1 + len(np.array(history.history['val_r_square'])))],
                      y=[np.array(history.history['r_square']), np.array(history.history['val_r_square'])],
                      titles=['R-Square Plot'],
                      labels=['training', 'validation'],
                      label_loc='lower right', num_plot=1, num_row=1, num_col=1, fig_size=(10, 7), x_label='#Epochs',
                      y_label='R-Square')

    # evaluate model on test set
    test_loss, test_rsquare = model_cnn.evaluate(X_test, y_test, verbose=2)
    print('\nTest Loss:', test_loss)
    print('Test R-Square:', test_rsquare)

    # take a random sample from the testing set for prediction using the trained model, "model_cnn"
    n = np.random.randint(0, len(X_test))
    X_to_predict = X_test[n]
    y_to_predict = y_test[n]
    cnn.predict(model_cnn, X_to_predict, y_to_predict)
