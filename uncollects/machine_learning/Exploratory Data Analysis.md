# Exploratory Data Analysis

Before tackling a problem with ML, it is usually a good idea to perform exploratory data analysis, to see if there are any obvious patterns (which might give hints on what method to choose), or any obvious problems with the data (e.g., label noise or outliers).

For tabular data with a small number of features, it is common to make a pair plot, in which panel (i, j) shows a scatter plot of variables i and j, and the diagonal entries (i, i) show the marginal density of variable i; all plots are optionally color coded by class label.

For higher-dimensional data, it is common to first perform dimensionality reduction, and then to visualize the data in 2d or 3d.
