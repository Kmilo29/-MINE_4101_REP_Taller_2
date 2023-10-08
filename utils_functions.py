# Data management Libraries
import pandas as pd
import numpy as np

# Data visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Utils
from pycountry_convert import country_alpha2_to_continent_code, country_name_to_country_alpha2


def univariate_analysis_continuos_variables(df):
    """calculates the univariate analysis for continuos variables
  
    Args:
        df (pandas.dataframe)
    """    
    for attribute in df.columns:
        print("---------------------------------------------------------------------------------------------")
        print(f"--------------------------------------{attribute}--------------------------------------------")
        print("---------------------------------------------------------------------------------------------")
        print("\n")
        
        # Descriptive statistics
        print(f"Estadísticas descriptivas\n")
        print(df[attribute].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))
        print("\n")

        # Seen how many values are null
        null_values = df[attribute].isnull().sum()
        print(f"Número de registros nulos: {null_values}")
        print("\n")

        # Seen how many values are cero
        value_counts = df[attribute].value_counts()
        print(f"Número de registros con valor cero: {value_counts.get(0, 0)}")
        print("\n")

        # Histograma
        plt.figure(figsize=(10, 5))
        plt.hist(df[attribute], bins=50, color='skyblue', edgecolor='black')
        plt.title("Histograma")
        plt.xlabel(attribute)
        plt.ylabel("Frequency")
        plt.show()

        # Density plot
        plt.figure(figsize=(10, 5))
        df[attribute].plot(kind='density', color='skyblue')
        plt.title("Gráfico de Densidad")
        plt.xlabel(attribute)
        plt.show()
                                
        # Histogram and density plot
        plt.figure(figsize=(10, 5))
        sns.histplot(df[attribute], bins=50, kde=True, color='orange', edgecolor='k')
        plt.title("Histograma y Gráfico de Densidad")
        plt.xlabel(attribute)
        plt.ylabel('Frequency')
        plt.show()
        print("\n\n")
        


def univariate_analysis_discrete_variables(df):
    """calculates the univariate analysis for discrete variables
  
    Args:
        df (pandas.dataframe)
    """    
    for attribute in df.columns:
        print("---------------------------------------------------------------------------------------------")
        print(f"--------------------------------------{attribute}--------------------------------------------")
        print("---------------------------------------------------------------------------------------------")
        print("\n")
        
        # Descriptive statistics
        print(f"Estadísticas descriptivas\n")
        print(df[attribute].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))
        print("\n")

        # Seen how many values are null
        null_values = df[attribute].isnull().sum()
        print(f"Número de registros nulos: {null_values}")
        print("\n")

        # Seen how many values are cero
        value_counts = df[attribute].value_counts()
        print(f"Número de registros con valor cero: {value_counts.get(0, 0)}")
        print("\n")
        
        # Showing unique values
        unique_values = df[attribute].unique()
        print(f"Los valores unicos son: {unique_values}")
        print("\n")
                
        # Plotting a count plot for polityscore
        plt.figure(figsize=(10,4),dpi=200)
        sns.countplot(x=attribute, data=df)
        plt.title(f"Distribución de {attribute}")
        print("\n\n")


def univariate_analysis_categorical_variables(df):
    """calculates the univariate analysis for categorical variables
  
    Args:
        df (pandas.dataframe)
    """  
    for attribute in df.columns:
        print("---------------------------------------------------------------------------------------------")
        print(f"--------------------------------------{attribute}--------------------------------------------")
        print("---------------------------------------------------------------------------------------------")
        print("\n")

        # Showing unique values
        unique_values = df[attribute].unique()
        print(f"Los valores unicos son: {unique_values}")
        print("\n")
                
        # Calculating the absolute frecuency
        print(f"Frecuencia absoluta\n")
        print(df[attribute].value_counts(dropna=False, normalize=False))   
        print("\n")
        
        # Calculating the relative frecuency
        print(f"Frecuencia relativa\n")
        print(df[attribute].value_counts(dropna=False, normalize=True))
        print("\n")
            
        # Seen how many values are null
        null_values = df[attribute].isnull().sum()
        print(f"Número de registros nulos: {null_values}")
        print("\n")
        
        # countplot
        plt.figure(figsize=(10,4),dpi=200)
        sns.countplot(x=attribute, data=df)
        plt.title(f"Distribución de {attribute}")
        print("\n\n")
        

def identify_continent(country):
    """function to identify the continent base in its country 
    
    Args:
        country (str)

    Returns:
        string 
    """
    try:
        country_code = country_name_to_country_alpha2(country)       
        
        if country_code:
            continent_code = country_alpha2_to_continent_code(country_code)
            if continent_code:
                continente = {
                    'AF': 'Africa',
                    'AS': 'Asia',
                    'EU': 'Europe',
                    'NA': 'America',
                    'SA': 'America',
                    'OC': 'Oceania',
                    'AN': 'Antarctica'
                }.get(continent_code, 'Unknown')

                return continente
            else:
                return "Unknown"
        else:
            return "Unknown"
    except LookupError:
        return "Unknown"
        

def target_vs_features(df, target):
    """ Function for generating scatter plots between 
        the target variable and the features of a dataset
    
    Args:
        df (pandas.dataframe)
        target (str)
    """
    # Filter numeric columns
    numeric_columns = df.select_dtypes(include='number').columns

    # scatterplot
    for column in numeric_columns:
        if column != target:
            plt.figure(figsize=(7, 4))
            plt.scatter(df[column], df[target], alpha=0.5)
            plt.title(f'Scatterplot de {target} vs {column}')
            plt.xlabel(column)
            plt.ylabel(target)
            plt.grid(True)
            plt.show()
            print("\n")