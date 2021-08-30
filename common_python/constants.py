"""Constants used in common_python."""

import os

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(CODE_DIR, "tests")
PCL = "pcl"
ENSEMBLE_PATH = os.path.join(CODE_DIR, "ensemble_file.%s" % PCL)

# Column names
ACTUAL = "actual"  # True value
CLASS = "class"
COUNT = "count"  # A count
CLS_FEATURE = "cls_feature"
CMP_FEATURE = "cmp_feature"
FEATURE = "feature"
FEATURE_SET = "feature_set"
FEATURE_VECTOR = "feature_vector"
FRAC = "frac"  # A fractiona
FRACNEG = "fracneg"  # A fractiona
FRACPOS = "fracpos"  # A fractiona
GROUP = "group"
INDEX = "index"
INSTANCE_STR = "instance_str"
INTERCEPT = "intercept"
KEGG_DESCRIPTION = "kegg_description"
KEGG_PATHWAY = "kegg_pathway"
KEGG_GENE = "kegg_gene"
KEGG_EC = "kegg_ec"  # EC number
KEGG_KO = "kgg_ko"  # KEGG orthology
MEAN = "mean"
NUM_POS = "num_pos"
NUM_SAMPLE = "num_sample"
NUM_ZERO = "num_zero"  # Number of zeroes in SIGLVL
PREDICTED = "predicted"  # Predicted valuejjj
PRIOR_PROB = "prior_prob"
SCORE = "score"
STERR = "sterr"  # Standard error (std of mean)
SIGLVL = "siglvl"
SIGLVL_POS = "siglvl_pos"
SIGLVL_NEG = "siglvl_neg"
STD = "std"  # Standard deviation of population
SUM = "sum"  # Sum of values
VALUE = "value"  # General value

# KEGG Access
KEGG_CMD_LIST = "list"
KEGG_CMD_GET = "get"
KEGG_CMDS = [KEGG_CMD_LIST, KEGG_CMD_GET]

# Plotting
PLT_CMAP = "cmap"
PLT_COLOR = "color"
PLT_FIGSIZE = "figsize"
PLT_IS_PLOT = "is_plot"  # Flag to plot
PLT_LEGEND = "legend"
PLT_XLABEL = "xlabel"
PLT_XLIM = "xlim"
PLT_XTICKLABELS = "xticklabels"
PLT_YLABEL = "ylabel"
PLT_YLIM = "ylim"
PLT_YTICKLABELS = "yticklabels"
PLT_TITLE = "title"

# Letters
UPPER_CASE = [x for x in map(chr, range(65, 91))]

# Values
PCLASS = 1  # Positive class
NCLASS = 0  # Negative class
FEATURE_SEPARATOR = "+" # Separator in string with multiple features
TRINARY_VALUES = [-1, 0, 1]

