import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from itertools import compress
from esperity.cough_analysis.utils import plot_hist_get_mean
import numpy as np


class EDA:
    """
    EDA class gets a Pandas dataframe to perform EDA on using its methods.
    """

    def __init__(self, df=None):
        """

        :param df: Pandas dataframe on which EDA will be performed
        """
        self.df = df

    def is_null(self, df=None):
        """

        :param df: The dataframe on which the method will be applied. If None (default), the self.df of init method will
         be used
        :return: A list of the column names that contains at least 1 missing values (NaN or Null)
        """

        if df is None:
            df = self.df
        return df.columns[df.isnull().any()].tolist()

    def drop_for_missing_values(self, threshold=0.4, df=None, in_place=True):
        """

        :param threshold: If the mean of the number of missing values of a column exceeds the threshold, the column
         will be removed from the dataframe (default is 0.4).
        :param df: The dataframe on which the method will be applied. If None (default), the self.df of init method will
         be used
        :param in_place: If True (default), the column(s) removal will be done in place, and if False, the method will
        return a new filtered dataframe.
        :return: The dataframe from which the columns with high missing values are removed.
        """

        if df is None:
            df = self.df

        # Calculate the ratio of missing values for each column
        null_ratio = df.isnull().mean()

        # Select the columns with a null ratio greater than the threshold
        to_drop = null_ratio[null_ratio > threshold].index

        # Drop the columns
        if in_place:
            return df.drop(columns=to_drop, in_place=in_place)
        else:
            return df.drop(columns=to_drop)

    def get_low_var_features(self, threshold, df=None):
        """

        :param threshold: The features with variance lower than this threshold will be considered as low variance
         features.
        :param df: The dataframe on which the method will be applied. If None (default), the self.df of init method will
         be used
        :return: The method will return a list of the names of the low variance features as the 1st return and a list of
         the names of the high variance features as the 2nd return (can be used as the input of the method
         drop_feature_by_name).
        """

        if df is None:
            df = self.df

        feature_names = list(df.columns)
        vt = VarianceThreshold(threshold=threshold)
        _ = vt.fit(df)

        features_low_var = list(compress(feature_names, ~vt.get_support()))
        features_high_var = list(compress(feature_names, vt.get_support()))

        return features_low_var, features_high_var

    def drop_feature_by_name(self, feature_names, df=None, in_place=True):
        """

        :param feature_names: A list of the names of the features to be removed from the dataframe.
        :param df: The dataframe on which the method will be applied. If None (default), the self.df of init method will
         be used.
        :param in_place: If True (default), the column(s) removal will be done in place, and if False, the method will
         return a new dataframe.
        :return: The dataframe from which the features with the same name as the input feature_names are removed.
        """

        if df is None:
            df = self.df

        if in_place:
            return df.drop(columns=feature_names, in_place=in_place)
        else:
            return df.drop(columns=feature_names)

    def get_similar_value_features(self, threshold=90, df=None, binary_encoded=False):
        """

        :param threshold: The percentage (default 90%) that is used to determine the features with high number of
         similar values.
        :param df: The dataframe on which the method will be applied. If None (default), the self.df of init method will
         be used.
        :param binary_encoded: If there are binary encoded features in the given dataframe, this parameters should be
         True (Note the default is False)
        :return: A list of names of features where a singular value occurs more than the threshold.
        """

        if df is None:
            df = self.df

        sim_val_cols = []
        for col in df.columns:
            percent_vals = (df[col].value_counts() / len(df) * 100).values
            if binary_encoded:
                # filter columns where more than 90% values are same and leave out binary encoded columns
                if percent_vals[0] > threshold and len(percent_vals) > 2:
                    sim_val_cols.append(col)
            else:
                if percent_vals[0] > threshold:
                    sim_val_cols.append(col)
        return sim_val_cols

    def get_correlated_features(self, threshold=0.9, df=None):
        """

        :param threshold: Each pair of features with correlation higher than this threshold (default 0.9) will be
         considered as highly correlated pairwise features.
        :param df: The dataframe on which the method will be applied. If None (default), the self.df of init method will
         be used.
        :return: A list of names of features which are highly correlated.
        """

        if df is None:
            df = self.df

        # Correlation matrix
        corr_matrix = df.corr().abs()
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        # Find columns with correlation greater than threshold
        col_correlated = [column for column in upper.columns if any(upper[column] > threshold)]
        return col_correlated


if __name__ == '__main__':
    import os

    absolute_dirpath = os.path.abspath(os.path.dirname(__file__))
    all_dataset = pd.read_csv(os.path.join(absolute_dirpath, 'all\\all_dataset.csv'))

    # change the name of the first column to "coughs"
    all_dataset.rename(columns={list(all_dataset.columns)[0]: 'coughs'}, inplace=True)

    eda = EDA()
    features_with_null = eda.is_null(all_dataset)
    features_low_var, features_high_var = eda.get_low_var_features(0.001, all_dataset.iloc[:, 1:])
    q25, q75, q90, mean, max, min = plot_hist_get_mean(all_dataset.iloc[:, 1:-2].var().to_numpy(), bins=None)
    # DF = eda.drop_feature_by_name(features_low_var, all_dataset, in_place=False)

