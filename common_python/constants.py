"""Constants used in common_python."""

import os

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(CODE_DIR, "tests")

# Column names
COUNT = "count"  # A count
FRAC = "frac"  # A fractiona
KEGG_DESCRIPTION = "kegg_description"
KEGG_PATHWAY = "kegg_pathway"
KEGG_GENE = "kegg_gene"
KEGG_EC = "kegg_ec"  # EC number
KEGG_KO = "kgg_ko"  # KEGG orthology
VALUE = "value"  # General value

# KEGG Access
KEGG_CMD_LIST = "list"
KEGG_CMD_GET = "get"
KEGG_CMDS = [KEGG_CMD_LIST, KEGG_CMD_GET]

# Plotting
PLT_XLABEL = "xlabel"
PLT_YLABEL = "ylabel"
PLT_TITLE = "title"
PLT_FIGSIZE = "figsize"
PLT_IS_PLOT = "is_plot"  # Flag to plot
PLT_CMAP = "cmap"

# Letters
UPPER_CASE = [x for x in map(chr, range(65, 91))]
