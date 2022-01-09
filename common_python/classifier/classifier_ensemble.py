"""
Ensemble of classifiers formed by training with different data.
The underlying classifier is called the base classifier.
1. Requirements
  a. The base classifier must expose methods for fit,
     predict, score.
  b. Sub-class ClassifierDescriptor and implement
     getImportance
2. The EnsembleClassifer is trained using randomly
   selecting data subsets using a specified number of 
   holdouts.
3. The ClassifierEnsemble exposes fit, predict, score.
4. In addition, features can be exampled and plotted
  a. Importance is float for a feature that indicates 
     its contribution
     to classifications
  b. Rank is an ordering of features based on their 
     importance
"""

import common_python.constants as cn
from common_python.statistics import util_statistics
from common_python.util import util
from common_python.classifier.classifier_collection  \
    import ClassifierCollection
from common_python.classifier import util_classifier
from common_python.util import persister

import collections
import copy
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
import pandas as pd
from sklearn import svm
import string

#
MULTI_COLUMN_SEP = "--"
LETTERS = string.ascii_lowercase

######### Functions
def _selStrFromList(item_str, lst_str):
  """
  Selects a string from a list if it is a substring of the item.

  Parameters
  ----------
  item_str: str
      string for which the selection is done
  list_str: list-str
      list from which selection is done
  
  Returns
  -------
  str
  """
  strs = [s for s in lst_str if item_str == s]
  if len(strs) != 1:
    strs = [s for s in lst_str if s in item_str]
  if len(strs) != 1:
    raise ValueError("Cannot find %s uniquely in %s"
        % (item_str, str(lst_str)))
  return strs[0]


###############################################
class ClassifierDescriptor(object):
  # Describes a classifier used to create an ensemble

  def __init__(self, df_X=None):
    self.df_X = df_X
    self.clf = None # Must assign to a classifier Type

  def getImportance(self, clf, **kwargs):
    """
    Default handling of importance is that the classifier
    has an appropriate method.
    :param Classifier clf:
    :param dict kwargs: optional parameters for getImportance
    :return list-float:
    """
    return clf.getImportance(**kwargs)


class ClassifierDescriptorSVM(ClassifierDescriptor):
  """
  Descriptor information needed for SVM classifiers
  Descriptor is for one-vs-rest. So, there is a separate
  classifier for each class.
  """
  
  def __init__(self, clf=svm.LinearSVC(), **kwargs):
    super().__init__(**kwargs)
    self.clf = clf

  def getImportance(self, clf, class_selection=None):
    """
    Calculates the importances of features. Takes into account
    the replications, if desired.
    :param Classifier clf:
    :param DataFrame df_X:
    :param int class_selection: class for which importance is computed
    :return list-float:
    """
    if class_selection is None:
      # If none specified, choose the largest magnitude coefficient across the
      # 1-vs-rest classifiers.
      coefs = [max([np.abs(x) for x in xv]) for xv in zip(*clf.coef_)]
    else:
      coefs = [np.abs(x) for x in clf.coef_[class_selection]]
    return coefs

  def getFeatureContributions(self, clf, features, ser_X):
    """
    Constructs the contributions of features to the binary classification.
    These are the individual values of each do product.

    Parameters
    ----------
    :param SVMClassifier clf:
    :param list-str features: features in order for classifier
    :param Series ser_X: column - feature, row - instance
    
    Returns
    -------
    DataFrame: column - feature, row - class, value - contribution
    """
    labels = set(features).intersection(ser_X.index)
    ser_X_sub = ser_X.loc[labels]
    df_coef = pd.DataFrame(clf.coef_, columns=features)
    df_X = pd.concat([ser_X_sub for _ in range(len(clf.coef_))], axis=1)
    df_X = df_X.T
    df_X.index = df_coef.index
    df_result = df_X.multiply(df_coef)
    return df_result


###################################################
class ClassifierEnsemble(ClassifierCollection):
 
  def __init__(self, clf_desc=ClassifierDescriptorSVM(),
      holdouts=1, size=1, filter_high_rank=None,
      is_display_errors=True,
      **kwargs):
    """
    :param ClassifierDescriptor clf_desc:
    :param int holdouts: number of holdouts when fitting
    :param int size: size of the ensemble
    :param int/None filter_high_rank: maximum feature rank considered
    :param bool is_display_errors: Show errors found in data
    :param dict kwargs: keyword arguments used by parent classes
    """
    self.clf_desc = clf_desc
    self.filter_high_rank = filter_high_rank
    self.holdouts = holdouts
    self.size = size
    self._is_display_errors = is_display_errors
    self.columns = None # Columns used in prediction
    self.classes = None
    self._df_X = None
    self._ser_y = None
    self._class_names = None
    super().__init__(**kwargs)

  def fit(self, df_X, ser_y, class_names=None, collectionMaker=None):
    """
    Fits the number of classifiers desired to the data with
    holdouts. Selects holdouts independent of class.
    :param pd.DataFrame df_X: feature vectors; indexed by instance
    :param pd.Series ser_y: classes; indexed by instance
    :param Function collectionMaker: function of df_X, ser_y
        that makes a collection
    :param class_names list-str: list of names of classes for values in ser_y
    """
    # Initializations
    self._df_X = df_X
    self._ser_y = ser_y
    self._class_names = class_names
    def defaultCollectionMaker(df_X, ser_y):
      return ClassifierCollection.makeByRandomHoldout(
          self.clf_desc.clf, df_X, ser_y, self.size, 
          holdouts=self.holdouts)
    #
    self.clf_desc.df_X = self._df_X
    self.classes = list(set(self._ser_y.values))
    if collectionMaker is None:
      collectionMaker = defaultCollectionMaker
    collection = collectionMaker(self._df_X, self._ser_y)
    self.update(collection)
    if self.filter_high_rank is None:
      self.columns = self._df_X.columns.tolist()
      return
    # Select the features
    df_rank = self.makeRankDF()
    df_rank_sub = df_rank.loc[
        df_rank.index[0:self.filter_high_rank], :]
    self.columns = df_rank_sub.index.tolist()
    df_X_sub = self._df_X[self.columns]
    collection = collectionMaker(df_X_sub, self._ser_y)
    self.update(collection)

  def predict(self, df_X):
    """
    Default prediction algorithm. Reports probability of each class.
    The probability of a class is the fraction of classifiers that
    predict that class.
    :param pd.DataFrame: features, indexed by instance.
    :return pd.DataFrame:  probability by class;
        columns are classes; index by instance
    Notes:
      1. Assumes that self.clfs has been previously populated 
         with fitted classifiers.
      2. Considers all classes encountered during fit
    """
    # Change to an array of array of features
    DUMMY_COLUMN = "dummy_column"
    if not isinstance(df_X, pd.DataFrame):
      raise ValueError("Must pass dataframe indexed by instance")
    #
    indices = df_X.index
    missing_columns = list(set(self.columns).difference(df_X.columns))
    new_df_X = df_X.copy()
    rename_dct = {}
    if len(missing_columns) > 0:
      if self._is_display_errors:
        msg = "***Warning: missing columns in prediction vector. \n %s"  \
            % str(missing_columns)
        print (msg)
      # Handle correlated columns by using the first one as surrogate
      # This should work if the prediction is done on data
      # with all of the genes.
      zero_columns = []
      for column in missing_columns:
        if MULTI_COLUMN_SEP in column:
          pos = column.index(MULTI_COLUMN_SEP)
          rename_dct[missing_column] = column[0:pos]
        else:
          # Column is not present in the prediction features
          zero_columns.append(column)
      zeroes = np.repeat(0, len(new_df_X))
      for column in zero_columns:
        new_df_X[column] = zeroes
    # Update the prediction features to accommodate the training data
    new_df_X = new_df_X.rename(rename_dct, axis='columns')
    # Do the prediction
    df_X_sub = new_df_X[self.columns]
    array = df_X_sub.values
    array = array.reshape(len(new_df_X.index), len(df_X_sub.columns)) 
    # Create a dataframe of class predictions
    clf_predictions = [clf.predict(df_X_sub)
        for clf in self.clfs]
    instance_predictions = [dict(collections.Counter(x))
        for x in zip(*clf_predictions)]
    df = pd.DataFrame()
    df[DUMMY_COLUMN] = np.repeat(-1, len(self.classes))
    for idx, instance in enumerate(instance_predictions):
        ser = pd.Series(
            [x for x in instance.values()], index=instance.keys())
        df[idx] = ser
    del df[DUMMY_COLUMN]
    df = df.fillna(0)
    df = df / len(self.clfs)
    df_result = df.T
    df_result.index = indices
    return df_result

  def score(self, df_X, ser_y):
    """
    Evaluates the accuracy of the ensemble classifier for
    the instances pvodied.
    :param pd.DataFrame df_X: columns are features, rows are instances
    :param pd.Series ser_y: rows are instances, values are classes
    Returns a float in [0, 1] which is the accuracy of the
    ensemble classifier.
    Assumes that self.clfs has been previously populated with fitted
    classifiers.
    """
    df_predict = self.predict(df_X)
    missing_columns = set(ser_y).difference(
        df_predict.columns)
    for column in missing_columns:
      df_predict[column] = np.repeat(0,
          len(df_predict))
    accuracies = []
    for instance in ser_y.index:
      # Accuracy is the probability of selecting the correct class
      try:
        accuracy = df_predict.loc[instance, ser_y.loc[instance]]
      except:
        import pdb; pdb.set_trace()
      accuracies.append(accuracy)
    return np.mean(accuracies)

  def _orderFeatures(self, clf, class_selection=None):
    """
    Orders features by descending value of importance.
    :param int class_selection: restrict analysis to a single class
    :return list-int:
    """
    values = self.clf_desc.getImportance(clf,
        class_selection=class_selection)
    length = len(values)
    sorted_tuples = np.argsort(values).tolist()
    # Calculate rank in descending order
    result = [length - sorted_tuples.index(v) for v in range(length)]
    return result

  def makeRankDF(self, class_selection=None):
    """
    Constructs a dataframe of feature ranks for importance,
    where the rank is the feature order of importance.
    A more important feature has a lower rank (closer to 1)
    :param int class_selection: restrict analysis to a single class
    :return pd.DataFrame: columns are cn.MEAN, cn.STD, cn.STERR
    """
    df_values = pd.DataFrame()
    for idx, clf in enumerate(self.clfs):
      df_values[idx] = pd.Series(self._orderFeatures(clf,
          class_selection=class_selection),
          index=self.features)
    df_result = self._makeFeatureDF(df_values)
    df_result = df_result.fillna(0)
    return df_result.sort_values(cn.MEAN)

  def makeImportanceDF(self, class_selection=None):
    """
    Constructs a dataframe of feature importances.
    :param int class_selection: restrict analysis to a single class
    :return pd.DataFrame: columns are cn.MEAN, cn.STD, cn.STERR
    The importance of a feature is the maximum absolute value of its coefficient
    in the collection of classifiers for the multi-class vector (one vs. rest, ovr).
    """
    ABS_MEAN = "abs_mean"
    df_values = pd.DataFrame()
    for idx, clf in enumerate(self.clfs):
      values = self.clf_desc.getImportance(clf,
          class_selection=class_selection)
      df_values[idx] = pd.Series(values, index=self.features)
    df_result = self._makeFeatureDF(df_values)
    df_result[ABS_MEAN] = [np.abs(x) for x in df_result[cn.MEAN]]
    df_result = df_result.sort_values(ABS_MEAN, ascending=False)
    del df_result[ABS_MEAN]
    df_result = df_result.fillna(0)
    return df_result

  def _plot(self, df, top, fig, ax, is_plot, **kwargs):
    """
    Common plotting codes
    :param pd.DataFrame: cn.MEAN, cn.STD, indexed by feature
    :param str ylabel:
    :param int top:
    :param bool is_plot: produce the plot
    :param ax, fig: matplotlib
    :param dict kwargs: keyword arguments for plot
    """
    # Data preparation
    if top == None:
      top = len(df)
    indices = df.index.tolist()
    indices = indices[0:top]
    df = df.loc[indices, :]
    # Plot
    if ax is None:
      fig, ax = plt.subplots()
    ax.bar(indices, df[cn.MEAN], yerr=df[cn.STD],
        align='center', 
        alpha=0.5, ecolor='black', capsize=10)
    bottom = util.getValue(kwargs, "bottom", 0.25)
    plt.gcf().subplots_adjust(bottom=bottom)
    ax.set_xticklabels(indices, rotation=90, fontsize=10)
    ax.set_ylabel(kwargs[cn.PLT_YLABEL])
    ax.set_xlabel(util.getValue(kwargs, cn.PLT_XLABEL, "Gene Group"))
    this_max = max(df[cn.MEAN] + df[cn.STD])*1.1
    this_min = min(df[cn.MEAN] - df[cn.STD])*1.1
    this_min = min(this_min, 0)
    ylim = util.getValue(kwargs, cn.PLT_YLIM,
        [this_min, this_max])
    ax.set_ylim(ylim)
    if cn.PLT_TITLE in kwargs:
      ax.set_title(kwargs[cn.PLT_TITLE])
    if is_plot:
      plt.show()
    return fig, ax

  def makeInstancePredictionDF(self, df_X, ser_y):
    """
    Constructs predictions for each instance for
    using a classifier trained without that instance.
    :param pd.DataFrame df_X:
    :param pd.Series ser_y:
    :return pd.DataFrame:
      columns: state indices
      index: instance
      value: probability
    """
    indices = ser_y.index
    dfs = []
    for test_index in indices:
        train_indices = list(set(indices).difference(
           [test_index])) 
        new_clf = copy.deepcopy(self)
        new_clf.fit(df_X.loc[train_indices, :],
            ser_y.loc[train_indices])
        df_X_test = pd.DataFrame(df_X.loc[
            test_index, :]) 
        dfs.append(new_clf.predict(df_X_test.T))
    df_pred = pd.concat(dfs)
    return util_classifier.aggregatePredictions(df_pred)

  def calcAdjStateProbTail(self, df_X, ser_y):
    """
    Calculates the probability of having at least the
    number of misclassifications be in an adjacent state
    as the number of predictions for the current classifier.
    :param pd.DataFrame df_X:
    :param pd.Series ser_y:
    :return float:
    """
    ser_pred = self.makeInstancePredictionDF(df_X, ser_y)
    # Find the indices where the predicted is not
    # the same as the actual
    sel = [v1 != v2 for v1, v2 in 
        zip(ser_y.values, ser_pred.values)]
    indices = ser_y[sel].index
    ser_state = util_classifier.calcStateProbs(ser_y)
    # Calculate the probability of an adjacent state
    # for each misclassified index
    probs = []
    num_adjacent = 0  # predicteds that are adjacents
    for index in indices:
      adj_states = util_classifier.findAdjacentStates(
          ser_y, index)
      states = [adj_states.prv, adj_states.nxt]
      if np.nan in states:
        states.remove(np.nan)
      states = list(set(states))
      if ser_pred[index] in states:
        num_adjacent += 1
      probs.append(np.sum([
          ser_state[s]/(1 - ser_state[adj_states.cur])
          for s in states]))
    # Calculate the tail probability for the number
    # of adjacents.
    prob = util_statistics.generalizedBinomialTail(
        probs, num_adjacent)
    #
    return prob
          
  def plotRank(self, top=None, fig=None, ax=None, 
      is_plot=True, **kwargs):
    """
    Plots the rank of features for the top valued features.
    :param int top:
    :param bool is_plot: produce the plot
    :param ax, fig: matplotlib
    :param dict kwargs: keyword arguments for plot
    """
    # Data preparation
    df = self.makeRankDF()
    kwargs = util.setValue(kwargs, cn.PLT_YLABEL, "Rank")
    self._plot(df, top, fig, ax, is_plot, **kwargs)

  def plotImportance(self, top=None, fig=None, ax=None, 
      is_plot=True, **kwargs):
    """
    Plots the rank of features for the top valued features.
    :param int top:
    :param bool is_plot: produce the plot
    :param ax, fig: matplotlib
    :param dict kwargs: keyword arguments for plot
    """
    df = self.makeImportanceDF()
    kwargs = util.setValue(kwargs, cn.PLT_YLABEL, "Importance")
    self._plot(df, top, fig, ax, is_plot, **kwargs)

  def plotSVMCoefficients(self, **kwargs):
    """
    Bar plot of the SVM coefficients.

    Parameters
    ----------
    kwargs: dict
        optional arguments for plotting
    """
    ser_X = pd.Series(np.repeat(1, len(self.features)))
    ser_X.index = self.features
    new_kwargs = dict(kwargs)
    new_kwargs["is_plot"] = False
    ax = self._plotFeatureBars(ser_X, **new_kwargs)
    ax.set_ylabel("Coefficient")
    self._showPlot(kwargs)

  @staticmethod
  def _showPlot(kwargs):
    if "is_plot" in kwargs:
      if kwargs["is_plot"]:
        plt.show()
    else:
      plt.show()

  def plotFeatureContributions(self, ser_X, **kwargs):
    """
    Plots the contribution of each feature to the final score by class
    averaged across the ensemble.  This is presented as a bar plot.
    :param Series ser_X:
    :param dict kwargs: options when plotting feature bars
    :return pyplot.Axes:
    """
    def getKwarg(key, default=None):
      if key in kwargs.keys():
        if kwargs[key] is not None:
          return kwargs[key]
      return default
    #
    true_class = getKwarg("true_class", None)
    is_plot = getKwarg("is_plot", True)
    class_names = getKwarg("class_names", default=self.classes)
    is_xlabel = getKwarg("is_xlabel", True)
    is_ylabel = getKwarg("is_ylabel", True)
    title = getKwarg("title", None)
    if not "getFeatureContributions" in dir(self.clf_desc):
      raise RuntimeError(
          "Classifier description must have the method getFeatureContributions")
    if self.classes is None:
      raise RuntimeError("Must fit before doing plotFeatureContributions")
    # Plot the contributions of features
    new_kwargs = dict(kwargs)
    new_kwargs["is_plot"] = False
    ax = self._plotFeatureBars(ser_X, **new_kwargs)
    # Calculate the mean and standard deviations of feature contributions
    ser_tot_mean = pd.Series(np.repeat(0, len(self.classes)),
        index=list(self.classes))
    ser_tot_std = pd.Series(np.repeat(0, len(self.classes)),
        index=list(self.classes))
    dfs = [self.clf_desc.getFeatureContributions(c, self.columns, ser_X)
        for c in self.clfs]
    for idx in self.classes:
      ser_tot_mean.loc[idx] = np.mean([df.loc[idx, :].sum() for df in dfs])
      ser_tot_std.loc[idx] = np.std([df.loc[idx, :].sum() for df in dfs])
    # Plot the contributions for each instance
    xvalues = [c + .25 for c in self.classes]
    ax.bar(xvalues, ser_tot_mean, yerr=ser_tot_std.values, fill=False,
        width=0.25)
    ax.plot([0, len(self.classes)+1], [0, 0], linestyle="dashed", color="black")
    is_misclassification = False
    if true_class is not None:
      ax.axvspan(true_class-0.25, true_class+0.4, facecolor='grey', alpha=0.2)
      tuples  = list(enumerate(ser_tot_mean.values))
      idx, _ = max(tuples, key=itemgetter(1))
      predicted_class = self.classes[idx]
      if predicted_class != true_class:
        is_misclassification = True
        # Indicate an inaccurate prediction
        # max_value = ser_tot_mean.max()*1.2
        # ax.text(xvalues[0], max_value, "x", color="red", fontsize=18, weight="bold")
    # xaxis label
    if isinstance(class_names[0], int):
      rotation = 0
    else:
      rotation = 30
    ax.set_xticklabels(class_names, rotation=rotation)
    if is_xlabel:
      ax.set_xlabel("class")
    else:
      ax.set_xticklabels([])
    # yaxis label
    if is_ylabel:
      ax.set_ylabel("contribution to class selection measure")
    else:
      ax.set_yticklabels([])
    # Title
    if is_misclassification:
      new_title = "**%s**" % title
    else:
      new_title = title
    ax.set_title(new_title)
    if "is_plot" in kwargs:
      if kwargs["is_plot"]:
        plt.show()
    else:
      plt.show()
    return ax

  def _plotFeatureBars(self, ser_X, ax=None, title="",
     true_class=None,  is_plot=True, is_legend=True, is_bar_label=True,
     is_xlabel=True, is_ylabel=True, class_names=None, is_legend_on_plot=False):
    """
    Plots the contribution of each feature to the final score by class
    averaged across the ensemble.  This is presented as a bar plot.
    :param Series ser_X:
    :param Matplotlib.Axes ax:
    :param str title:
    :param int true_class: true class for ser_X
    :param bool is_plot:
    :param bool is_legend: plot the legend
    :param list-str class_names:
    :param bool is_plot:
    :param bool is_legend_on_plot: leave space for legend to appear 
        on the plot
    :return pyplot.Axes:
    """
    # Calculate the mean and standard deviations of feature contributions
    dfs = [self.clf_desc.getFeatureContributions(c, self.columns, ser_X)
        for c in self.clfs]
    df_mean = dfs[0] - dfs[0]
    df_std = dfs[0] - dfs[0]
    for idx in df_mean.index:
      for col in df_mean.columns:
        values = [df.loc[idx, col] for df in dfs]
        df_mean.loc[idx, col] = np.mean(values)
        df_std.loc[idx, col] = np.std(values)
    mean_dct = {i: df_mean.loc[i,:].mean() for i in self.classes}
    std_dct = {i: df_std.loc[i,:].std() for i in self.classes}
    # Plot the contributions for each instance
    if ax is None:
      _, ax = plt.subplots(1)
    values = df_std.values
    (nrow, ncol) = np.shape(values)
    values = np.reshape(values, (ncol, nrow))
    if class_names is None:
      class_names = self.classes
    # Write text on bar components
    excluded_colors = ["grey", "gray", "white", "light", "snow", "linen",
        "oldlace", "cornsilk", "ivory", "beige", "honeydew", 
        "mintcream", "azure", "aliceblue", "lavender", "pink"]
    colors = util.getColors(len(df_mean.columns), excludes=excluded_colors)
    # Ensure bars are in sorted order
    columns = list(df_mean.columns)
    columns.sort()
    df_mean = df_mean[columns]
    if is_legend_on_plot:
      # Add two blank rows for the legend so legend appears on plot
      df_mean_T = df_mean.transpose()
      df_mean_T["blank1"] = 0
      df_mean_T["blank2"] = 0
      df_mean = df_mean_T.transpose()
      df_values = pd.DataFrame(values)
      df_values["blank1"] = 0.0
      df_values["blank2"] = 0.0
      values = df_values.values
      loc = "upper right"
    else:
      loc = "upper left"
    # Construct plot
    bar_ax = df_mean.plot(kind="bar", stacked=True, yerr=values, color=colors,
        ax=ax, width=0.25)
    letter_dct = {}
    cur_letter = 0
    for rect in bar_ax.patches:
      # Find where everything is located
      height = rect.get_height()
      width = rect.get_width()
      x = rect.get_x()
      y = rect.get_y()
      # The height of the bar is the data value and can be used as the label
      key = rect.get_facecolor()
      if key in letter_dct.keys():
        label = letter_dct[key]
      else:
        label = LETTERS[cur_letter]
        cur_letter += 1
        letter_dct[key] = label
      #label_text = "%2.2f" % height
      # ax.text(x, y, text)
      label_x = x + width / 2
      label_y = y + height / 2
      # plot only when height is greater than specified value
      if is_bar_label and (np.abs(height) > 0.1):
        ax.text(label_x, label_y, label, color="white", ha='center',
            va='center', fontsize=8)
    # Construct the legend
    letters = list(letter_dct.values())
    if is_bar_label:
      legends = ["%s: %s" % (letters[i], c) for i, c in enumerate(columns)]
    else:
      legends = ["%s" % c for i, c in enumerate(columns)]
    ax.legend(legends, bbox_to_anchor=(1.0, 1), loc=loc)
    if not is_legend:
      legend = ax.get_legend()
      legend.remove()
    # Leave one blank class for legend
    ax.plot([0, len(df_mean)+1], [0, 0], linestyle="dashed", color="black")
    if isinstance(class_names[0], int):
      rotation = 0
    else:
      rotation = 30
    ax.set_xticklabels(class_names, rotation=rotation)
    if is_xlabel:
      ax.set_xlabel("class")
    else:
      ax.set_xticklabels([])
    if is_ylabel:
      ax.set_ylabel("contribution to class selection measure")
    else:
      ax.set_yticklabels([])
    ax.set_title(title)
    if is_plot:
      plt.show()
    return ax

  def _makeFeatureDF(self, df_values):
    """
    Constructs a dataframe summarizing values by feature.
    :param pd.DataFrame df_values: indexed by feature, columns are instances
    :return pd.DataFrame: columns are cn.MEAN, cn.STD, cn.STERR
        indexed by feature
    """
    df_result = pd.DataFrame({
        cn.MEAN: df_values.mean(axis=1),
        cn.STD: df_values.std(axis=1),
        })
    df_result.index = df_values.index
    df_result[cn.STERR] = df_result[cn.STD] / np.sqrt(len(self.clfs))
    return df_result

  def serialize(self, file_path):
    """
    Exports the classifiers
    :param str file_path:
    """
    exporter = persister.Persister(file_path)
    exporter.set(self)

  @classmethod
  def deserialize(cls, file_path):
    """
    Imports the classifiers
    :param str file_path:
    :return ClassifierEnsemble:
    :exceptions ValueError: no persister file
    """
    exporter = persister.Persister(file_path)
    if not exporter.isExist():
      raise ValueError
    return exporter.get()

  @classmethod
  def crossValidate(cls, trinary_data, num_holdout=5, num_iter=10,
      clf_desc=ClassifierDescriptorSVM(), **kwargs):
    """
    Cross validates with the specified number of holdouts. Returns
    an overall accuracy calculation based on exact matches with
    classes.
    
    Parameters
    ----------
    trinary_data: TrinaryData
        df_X, ser_y
    num_holdout: int
        number of holdouts for each class
    num_iter: int
        number of cross validation iterations (folds)
    clf_desc: ClassifierDescriptor
    kwargs: dict
        arguments for constructor classifier constructor
        
    Returns
    -------
    float: accuracy
    """
    def dropIndices(df, indices):
        """
        Drops the indices from the dataframe or series.
        """
        df_result = df.copy()
        sorted_indices = list(indices)
        sorted_indices.sort()
        sorted_indices.reverse()
        for idx in sorted_indices:
            df_result = df_result.drop(idx, axis=0)
        return df_result
    #
    def getClasses(indices=None):
      """
      Returns the list of classes for the indices.
      """
      if indices is None:
        indices = list(trinary_data.ser_y.index)
      return list(set(trinary_data.ser_y.loc[indices]))
    #
    svm_ensemble = cls(clf_desc=clf_desc, **kwargs)
    all_classes = getClasses()
    total_correct = 0
    for _ in range(num_iter):
      # Select holdouts for each class
      holdout_idxs = []
      for cls in all_classes:
        cls_ser = trinary_data.ser_y[trinary_data.ser_y == cls]
        cls_idxs = list(cls_ser.index)
        if num_holdout >= len(cls_idxs):
          raise ValueError("Not enough samples in class %s for %d holdouts!"
              % (cls, num_holdout))
        # Choose holdouts
        random_positions = np.random.permutation(range(len(cls_idxs)))
        [holdout_idxs.append(cls_idxs[n])
            for n in random_positions[:num_holdout]]
      # Fit
      df_X = dropIndices(trinary_data.df_X, holdout_idxs)
      ser_y = dropIndices(trinary_data.ser_y, holdout_idxs)
      svm_ensemble.fit(df_X, ser_y)
      # Evaluate
      df = pd.DataFrame(trinary_data.df_X.loc[holdout_idxs, :])
      df_pred = svm_ensemble.predict(df)
      for idx in holdout_idxs:
          true_cls = trinary_data.ser_y.loc[idx]
          total_correct += df_pred.loc[idx, true_cls]
    accuracy = total_correct/(num_holdout*num_iter*len(all_classes))
    return accuracy

  def evaluateClassifierOnInstances(self, df_X=None, ser_y=None,
      is_plot=True, nrow=6, ncol=4, suptitle=""):
    """
    Constructs evaluation plots for the classifier based on data
    that without labels.

    Parameters
    ----------
    df_X: DataFrame
        columns: feature
        rows: instances
        values: trinary
    ser_y: Series
        rows: instances
        values: int (classes)
    is_plot: bool
    nrow: int
    ncol: integer
    suptitle: str
    """
    if df_X is None:
      df_X = self._df_X
      if ser_y is None:
        ser_y = self._ser_y
    _, axes = plt.subplots(nrow, ncol, figsize=(18,12))
    for irow in range(nrow):
        for icol in range(ncol):
          ax = axes[irow, icol]
          instance_num = irow*ncol + icol + 2
          instance = "T%d" % instance_num
          if instance not in df_X.index:
            break
          ser_X = df_X.loc[instance, :]
          if (icol + 1 == ncol) and (irow==0):
            is_legend = True
          else:
              is_legend = False
          if (irow + 1 == nrow):
            is_xlabel = True
            if icol == 0:
                is_ylabel = True
            else:
                is_ylabel = False
          else:
            is_ylabel = False
            is_xlabel = False
          self.plotFeatureContributions(ser_X, ax=ax,
              title=instance, true_class=ser_y.loc[instance], 
              is_plot=False, is_legend=is_legend, class_names=self._class_names,
                  is_xlabel=is_xlabel, is_ylabel=is_ylabel)
    plt.suptitle(suptitle)
    if not is_plot:
      plt.close()
    else:
      plt.show()

  def plotConditions(self, df_X, condition_strs, state_names=None, ax=None,
      is_plot=True, fontsize_label=10, shading_offset=0.25, state_probs=None): 
    """
    Plots predictions for data that have distinct conditions.

    Parameters
    ----------
    df_X: DataFrame (feature matrix)
    condition_strs: list-str
        Strings that identify conditions
    state_names: names for the states predicted
    shading_offset: float
        Offset from x tick for shading begin and end
    state_probs: list-float
        Probabilities of the states under a null hypothesis
    """
    def getConditionPosition(idx):
      new_condition_strs = list(condition_strs)
      condition_str = _selStrFromList(idx, new_condition_strs)
      return new_condition_strs.index(condition_str)
    #
    def calculateSignificance(state_probs, num, num_rpl, num_cnd, num_exp):
      """
      Calculates the probability of at least one study in
      assuming equally likely outcomes
  
      Parameters
      ----------
      state_probs: float/list-float
      """
      ps = np.array(state_probs)
      num_state = len(state_probs)
      p_ks = 1 - binom.cdf(num - 1, num_rpl, ps)
      # Calculate the product of the p_ks not equal to n
      p_tilde_func = lambda n: np.prod([1 - p_ks[m]
          for m in range(num_state) if m != n])
      p_tilde = sum([p_ks[n]*p_tilde_func(n) for n in range(num_state)])
      q_tilde = p_tilde**num_cnd
      q = 1 - (1 - q_tilde)**num_exp
      return p_ks, p_tilde, q_tilde, q




    plot_indices = df_X.index
    plot_indices = sorted(list(plot_indices), key=getConditionPosition)
    x_vals = range(len(plot_indices))
    newdf_X = df_X.copy()
    newdf_X.index = plot_indices
    df_prediction = self.predict(newdf_X)
    if state_names is None:
      state_names = [str(n) for n in df_prediction.columns]
    if ax is None:
      _, ax = plt.subplots(1)
    markers = ["o", "+", "^", "s", "*"]
    for idx, stage in enumerate(df_prediction.columns):
      ax.scatter(x_vals, df_prediction[stage], marker=markers[idx], s=50)
      ax.plot(plot_indices, np.repeat(-1, len(plot_indices)))
    ax.set_xticklabels(plot_indices, rotation=45, fontsize=fontsize_label)
    ax.legend(state_names, loc="lower left")
    ax.set_ylim([0, 1.1])
    # Add shading
    is_shade = False
    indices = list(df_prediction.index)
    last_condition = _selStrFromList(indices[0], condition_strs)
    start_pos = 0
    for pos, index in enumerate(indices):
      cur_condition = _selStrFromList(index, condition_strs)
      if last_condition != cur_condition:
        last_condition = cur_condition
        if is_shade:
          cur_condition_indices = [i for i in indices if cur_condition in i]
          end_pos = indices.index(cur_condition_indices[0]) - 1
          ax.axvspan(start_pos-shading_offset, end_pos+shading_offset, ymin=0,
              ymax=max(df_prediction.columns), facecolor='grey', alpha=0.2)
        # Toggle shading
        is_shade = True if (not is_shade) else False
        # Starting position for next block of conditions
        start_pos = pos
    # Handle last segment
    if is_shade:
      cur_condition_indices = [i for i in indices if cur_condition in i]
      end_pos = indices.index(cur_condition_indices[-1]) # ending x-value
      ax.axvspan(start_pos-shading_offset, end_pos+shading_offset, ymin=0,
          ymax=max(df_prediction.columns), facecolor='grey', alpha=0.2)
    if is_plot:
      plt.show()

  def plotProgression(self, df_X, repl_strs, time_strs,
      title="", ax=None, label_fontsize=16, is_plot=True):
    """
    Plots the progression of dominate states predicted over time
    for each replication.
    
    Parameters
    ----------
    df_X: DataFrame
        column: Feature
        row: instance (str)
    repl_strs: list-str
        String that identify the replication
    time_strs: list-str
        Strings that identify times, in sequence
    title: str
    ax: Matplotlib.axes
    is_plot: bool

    Notes: must do fit first.
    """
    def deDup(lst):
      seen = set()
      seen_add = seen.add
      return [x for x in lst if not (x in seen or seen_add(x))]
    #   
    if ax is None:
      _, ax = plt.subplots(1)
    if self._class_names is None:
      class_dct = {k: str(k) for k in self._ser_y.values}
      class_names = list(class_dct.values())
    else:
      class_names = list(self._class_names)
    class_names.insert(0, "")
    df_prediction = self.predict(df_X)
    repl_dct = {}
    for repl_str in repl_strs:
      indices = list(df_prediction.index)
      bools = [repl_str in i for i in indices]
      if any(bools):
        indices = [i for i in df_prediction.index if repl_str in i]
        y_vals = np.repeat(np.nan, len(time_strs))
        for idx in indices:
          time_str = _selStrFromList(idx, time_strs)
          pos = time_strs.index(time_str)
          row = df_prediction.loc[idx, :]
          val = row.max()
          # Account for the blank row in the plot
          y_val = 1 + [c for c in row.index if row[c] == val][0]
          y_vals[pos] = y_val
        repl_dct[repl_str] = y_vals
    # Construct plot, starting with longest first
    for y_vals in repl_dct.values():
      ax.plot(time_strs, y_vals, marker="o")
      ax.set_ylim(0, len(class_names))
      yticks = ax.get_yticklabels()[0]
      labels = list(class_names)
      ax.set_xticklabels(time_strs, rotation=90, fontsize=label_fontsize)
      ax.set_yticklabels(labels, fontsize=label_fontsize)
    plt.legend(repl_strs)
    fontsize = label_fontsize + 2
    plt.title(title, fontsize=fontsize)
    if is_plot:
      plt.show()
