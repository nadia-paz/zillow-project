import threading as t

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from statsmodels.stats.outliers_influence import variance_inflation_factor

######## GLOBAL VARIABLES ##########
cat_vars = ['bedrooms', 'bathrooms', 'county_name']
cont_vars = ['sq_feet', 'lot_sqft', 'home_value', 'house_age']
palettes = ['flare', 'Blues_r', 'PuRd_r', 'Accent']
colors_sns = sns.color_palette("flare")
colors_sns2 = sns.color_palette("Blues_r")
##################

# set confidence interval to 95%
alpha = 0.05


####### EXPLORATION FUNCTIONS ###########
def vif(df):
    '''
    the function accepts a data frame as a parameter
    calculates the varince inflation factor
    returns a data frame with results sorted by VIF
    '''
    vif_data = pd.DataFrame()
    vif_data['features'] = df.drop(columns='county_name').columns
    vif_data['VIF'] = [variance_inflation_factor(df.drop(columns='county_name').\
                                                 values, i) for i in range(len(df.columns)-1)]
    return vif_data.sort_values(by='VIF', ascending=False)

########## VISUALIZATIONS ########

def price_distribution(df):
    '''
    the function accepts a zillow data frame as a parameter
    shows the home_value distribution with median and mean lines
    '''
    plt.figure(figsize = (10, 6), facecolor='#fbf3e4')
    plt.suptitle('Home price distribution in the data set', fontsize = 20)
    # create bins for the distribution
    bins = np.arange(0, 2_000_000, 50_000).astype(int)

    # create a histogram object with custom xticks
    ax = sns.histplot(data=df, x = 'home_value',stat='percent', bins=bins,  kde=True)
    ax.set(xticks=[100_000, 250_000, 500_000, 750_000, 1_000_000, 1_500_000, 2_000_000])
    ax.set(xticklabels=['100K', '200K', '500K', '750K', '1M', '1.5M', '2M'])
    # highlight the longest line
    ax.patches[1].set_facecolor('rosybrown')

    # add vertical lines for the mean and median
    plt.vlines(df.home_value.mean(), 0, 6.2, color=(0.82, 0.29, 0.38), label='mean price')
    plt.vlines(df.home_value.median(), 0, 6.8, color='blue', label='median price')

    # sign the x and y axis
    plt.xlabel('Price in USD', fontsize=16)
    plt.ylabel('Percentage', fontsize=16)

    # calculate the values for the text on the graph
    #median = df.home_value.median()
    #mean = df.home_value.mean()
    out = pd.cut(df.home_value, bins)
    interval = str(out.value_counts().reset_index().iloc[0, 0])

    # add text to the graph
    plt.annotate(f'Most common price for houses is {interval}', xy=(100_000, 8), 
                xytext=(750_000, 8), size=16, arrowprops= dict(facecolor='rosybrown'))
    #plt.text(750_000, 4, 'each bin = $50,000')
    #plt.text(1_500_000, 5, f'mean = ${round(mean)}', fontdict={'fontsize':16})
    #plt.text(1_500_000, 3, f'median = ${round(median)}', fontsize=16)  
    #plt.text(1_500_000, 1, f'mean - median = ${round(mean - median)}', fontsize=16)

    # visualize the labels
    plt.legend()

    plt.show()

def counties_viz(df):
    '''
    the function accepts a zillow data frame and creates 
    two subplots that visualize how home prices vary in counties
    the left subplot shows median prices and the right subplot shows mean prices
    '''
    plt.figure(figsize=(18, 6), facecolor='#fbf3e4')

    # Median home prices in LA, Ventura and Orange counties
    plt.suptitle('Home prices in different counties', fontsize=20)
    plt.subplot(121)
    graph = sns.barplot(x='county_name', y='home_value', estimator=np.median, data=df, palette='flare')
    graph.axhline(df.home_value.median(), color = (0.4, 0.4, 0.4), label = 'median')
    plt.title('Median', fontsize=16)
    plt.ylim(0, 550_000)
    plt.legend()

    # Mean home prices in LA, Ventura and Orange counties
    plt.subplot(122)
    graph = sns.barplot(x='county_name', y='home_value', data=df, palette='Accent')
    graph.axhline(df.home_value.mean(), color = (0.4, 0.4, 0.4), label = 'mean')
    plt.title('Mean', fontsize=16)
    plt.ylim(0, 550_000)
    plt.legend()

    plt.show()

def pools_viz(df):
    '''
    the function accepts a zillow data frame as a parameter and creates 2 plots
    that show a difference between houses with pools and without pools
    '''
    plt.figure(figsize=(18, 6), facecolor='#fbf3e4')

    plt.suptitle('Median prices for homes with and without pool', fontsize=20)
    
    # subplot 1 houses with pool prices in different counties
    plt.subplot(121)
    graph = sns.barplot(x='pools', y='home_value', estimator=np.median, hue='county_name', data=df, palette='flare')
    graph.axhline(df.home_value.median(), label='median price')
    plt.title('Prices in different counties', fontsize=16)
    plt.ylim(0, 700_000)
    plt.legend()

    # subplot 2 houses with pool prices in the data set
    plt.subplot(122)
    graph = sns.barplot(x='pools', y='home_value', data=df, estimator=np.median, palette='Accent')
    graph.axhline(df.home_value.median(), label='median price')
    plt.ylim(0, 700_000)
    plt.title('Overall prices', fontsize=16)
    plt.legend()

    plt.show()

def sqft_price_viz(df):
    '''
    the function accepts a zillow data frame as a parameter
    draws two scatter plots that show sq_feet to home_value relation
    one subplot show a regression line and other subplot highlights different counties
    '''
    #sns.lmplot(x='sq_feet', y='home_value', data=df.sample(frac=0.05), col='county_name')
    plt.figure(figsize=(18, 6), facecolor='#fbf3e4')
    # create a custom palette for the 2nd subplot
    palette = ["#3544D1", '#24A54B', '#EF7C64']
    # set a title
    plt.suptitle('Correlation between house area and home prices', fontsize=20)

    # subplot #1
    plt.subplot(121)
    sns.regplot(x='sq_feet', y='home_value', data=df.sample(frac=0.05),
                line_kws={'color':(0.56, 0.198, 0.442)});
    plt.title('With regression line', fontsize=16)

    # subplot #2
    plt.subplot(122)
    sns.scatterplot(x='sq_feet', y='home_value', data=df.sample(frac=0.05), hue='county_name', 
                    palette=sns.color_palette(palette, 3))
    plt.title('With county highlights', fontsize=16)

    plt.show()

def age_price_viz(df):
    '''
    the function accepts a zillow data frame as a parameter and returns
    3 scatter plots with the regression lines
    col='county_name' -> each scatter plot represents a county (LA, Orange, Ventura)
    hue='pools' -> each plot has 2 regression lines that represent houses with/without pools
    '''
    sns.lmplot(x='house_age', y='home_value', data=df.sample(frac=0.05), 
           palette='muted', col='county_name', hue='pools', 
           scatter_kws = {'alpha' : 0.09}, markers=["o", "x"] ); # orange pools, blue no pools

######### STAT TESTS #####

def counties_test(df):
    '''
    the function accepts a zillow data set as a parameter, splits it into 3 data sets
    where each represents separate county
    run the Levene tests to check the assumptions and runs the Kruskal-Wallis stat test
    that compares means of the samples
    '''
    # create 3 data sets that keep the values of the counties
    la = df[df.county_name == 'LA'] # LA county
    ventura = df[df.county_name == 'Ventura'] # Ventura county
    orange = df[df.county_name == 'Orange'] # Orange county
    
    # Levene test
    p = stats.levene(la.home_value, orange.home_value, ventura.home_value)[1]
    if (p < alpha):
        print('Variances are different. Use an non-parametric Kruskal-Wallis test.')
    else:
        print('Variances are equal. Use a parametric ANOVA test')
    
    print()
    
    # Kruskal-Wallis test
    p_kr = stats.kruskal(la.home_value, orange.home_value, ventura.home_value)[1]
    if (p_kr < alpha):
        print('We reject the null hypothesis.')
        print('There is a significant difference in home prices in different counties.')
    else:
        print('We fail to reject the null hypothesis.')
        print('There is no significant difference in home prices in different counties.')

def sqfeet_test(df):
    '''
    the function accepts a zillow data frame as a parameter
    runs a Spearman's rank correlation test and prints it results.
    the function prints a corr coef for each county
    '''
    corr, p = stats.spearmanr(df.sq_feet, df.home_value)
    if (p < alpha):
        print('We reject the null hypothesis.')
        print('There is a linear correlation between home price and its size(square feet)')
        print(f'The correlation coefficient is {round(corr, 2)}')
    else:
        print('We fail to reject the null hypothesis.')
        print('There is no linear correlation between home price and its size(square feet)')

    # get the dataframes for each county
    la = df[df.county_name=='LA']
    orange = df[df.county_name=='Orange']
    ventura = df[df.county_name=='Ventura']

    # create a dictionary that holds 3 data frames
    counties = {
        'LA':la,
        'Ventura':ventura,
        'Orange':orange
    }

    # print correlation coefficients between sq_feet and home_value for each county
    for key in counties:
        corr, _ = stats.spearmanr(counties[key].sq_feet, counties[key].home_value)
        print(f'{key:<8} correlation coefficient is {round(corr, 2): >5}')
        print()

####### END OF EXPLORATION FUNCTIONS #########