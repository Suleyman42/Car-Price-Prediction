import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import missingno as msno


import pandas as pd

from scipy.stats import chi2_contingency


def plot_price_distribution_log(df):
    """
    Erstellt ein Histogramm der logarithmisch transformierten Preise.
    """
    if 'price' not in df.columns:
        raise ValueError("Die Spalte 'price' ist im DataFrame nicht vorhanden.")

    transformed_prices = np.log1p(df['price'])  # log(1 + price), um mit 0 klarzukommen
    sns.histplot(transformed_prices, kde=True)
    plt.title('Log-transformierte Verteilung der Preise')
    plt.xlabel('log(1 + price)')
    plt.ylabel('Count')
    plt.show()


def plot_price_distribution(df, max_price=None):
    """
    Erstellt ein Histogramm der Preise (ohne log-Transformation).
    Optional: Begrenze die Darstellung auf einen maximalen Preis.
    """
    if 'price' not in df.columns:
        raise ValueError("Die Spalte 'price' ist im DataFrame nicht vorhanden.")

    # Preis filtern, wenn max_price angegeben ist
    if max_price is not None:
        df = df[df['price'] <= max_price]

    sns.histplot(df['price'], kde=True)
    plt.title('Verteilung der Preise')
    plt.xlabel('Preis')
    plt.ylabel('Anzahl')

    if max_price is not None:
        plt.xlim(0, max_price)

    plt.show()



def plot_mutual_info_scores(mi_df):
    """
    Visualisiert Mutual Information Scores als horizontales Balkendiagramm.
    """
    mi_df_sorted = mi_df.sort_values('MI Score')

    plt.figure(figsize=(10, len(mi_df_sorted) * 0.3))
    plt.barh(mi_df_sorted['Feature'], mi_df_sorted['MI Score'])
    plt.xlabel('Mutual Information Score')
    plt.title('Mutual Information zwischen Features und Zielvariable')
    plt.tight_layout()
    plt.show()


def plot_missing_values(df):
    """
    🇩🇪 Zeigt eine Matrix der fehlenden Werte im DataFrame.
    🇬🇧 Displays a matrix visualization of missing values in the DataFrame.

    Parameter:
    ----------
    df : pd.DataFrame
        Der zu untersuchende DataFrame.
    """
    msno.matrix(df)
    plt.title("Fehlende Werte Matrix / Missing Values Matrix")
    plt.show()


def plot_scatter_price_vs_feature(df, feature, price_column='price'):
    """
    🇩🇪 Scatterplot: Preis vs. 1 oder 2 numerische Features.
    🇬🇧 Scatter plot: Price vs. 1 or 2 numeric features.

    Parameter:
    ----------
    df : pd.DataFrame
        DataFrame mit den Daten.
    feature : str oder list[str]
        Name eines Features oder Liste von zwei Features.
    price_column : str, optional
        Name der Zielspalte (default 'price').
    """
    plt.figure(figsize=(10, 6))

    # 1 Feature
    if isinstance(feature, str):
        sns.scatterplot(data=df, x=feature, y=price_column)
        plt.title(f"Preis vs. {feature}")

    # 2 Features
    elif isinstance(feature, list) and len(feature) == 2:
        x_feature = feature[0]
        hue_feature = feature[1]
        sns.scatterplot(data=df, x=x_feature, y=price_column, hue=hue_feature, palette='viridis')
        plt.title(f"Preis vs. {x_feature} mit Farbe = {hue_feature}")
        plt.legend(title=hue_feature)

    else:
        raise ValueError("Feature muss ein String oder eine Liste von genau 2 Spaltennamen sein.")

    plt.xlabel(feature[0] if isinstance(feature, list) else feature)
    plt.ylabel(price_column)
    plt.show()


def plot_boxplot_price_vs_category(df, category, price_column='price'):
    """
    🇩🇪 Boxplot: Preis vs. kategoriales Feature.
    🇬🇧 Box plot: Price vs. categorical feature.

    Parameter:
    ----------
    df : pd.DataFrame
        Der zu untersuchende DataFrame.
    category : str
        Name der kategorialen Spalte.
    price_column : str, optional
        Name der Zielspalte. Standard ist 'price'.
    """
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x=category, y=price_column)
    plt.title(f"Preis vs. {category} / Price vs. {category}")
    plt.xlabel(category)
    plt.ylabel("Preis / Price")
    plt.xticks(rotation=45)
    plt.show()


def plot_correlation_heatmap(df):
    """
    🇩🇪 Heatmap der Korrelationen für numerische Features.
    🇬🇧 Heatmap of correlations for numeric features.

    Parameter:
    ----------
    df : pd.DataFrame
        Der zu untersuchende DataFrame.
    """
    plt.figure(figsize=(12, 10))
    corr_matrix = df.select_dtypes(include=['number']).corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Korrelationsmatrix / Correlation Heatmap")
    plt.show()


def plot_cramers_v_heatmap(df):
    """
    🇩🇪 Heatmap der Cramér's V Werte für kategoriale Features.
    🇬🇧 Heatmap of Cramér's V for categorical features.

    Parameter:
    ----------
    df : pd.DataFrame
        Der zu untersuchende DataFrame.
    """
    # Nur Objekt- und Kategorie-Spalten
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    n = len(cat_cols)
    cramers_matrix = pd.DataFrame(np.zeros((n, n)),
                                  index=cat_cols,
                                  columns=cat_cols)

    # Alle Kombinationen berechnen
    for col1 in cat_cols:
        for col2 in cat_cols:
            cramers_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])

    plt.figure(figsize=(12, 10))
    sns.heatmap(cramers_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Cramér's V Heatmap (Kategoriale Variablen)")
    plt.show()

def cramers_v(x, y):
    """
    Calculate Cramér's V statistic for association between two categorical variables.
    Berechnet den Cramér's V-Koeffizienten für den Zusammenhang zweier kategorialer Variablen.

    Parameters:
    x (pd.Series): First categorical variable / Erste kategoriale Variable
    y (pd.Series): Second categorical variable / Zweite kategoriale Variable

    Returns:
    float: Cramér's V value between 0 and 1 / Cramér's V-Wert zwischen 0 und 1
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    cramers_v_value = np.sqrt(phi2 / min(k - 1, r - 1))



    return cramers_v_value



def plot_avg_price_per_year(df, year_col='Erstzulassung', price_column='price'):
    """
    🇩🇪 Liniendiagramm: Durchschnittlicher Preis pro Jahr.
    🇬🇧 Line plot: Average price per year.

    Parameter:
    ----------
    df : pd.DataFrame
        Der zu untersuchende DataFrame.
    year_col : str, optional
        Name der Spalte mit dem Jahr (Standard: 'Erstzulassung').
    price_column : str, optional
        Name der Zielspalte. Standard ist 'price'.
    """
    avg_price = df.groupby(year_col)[price_column].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=avg_price, x=year_col, y=price_column, marker="o")
    plt.title(f"Durchschnittlicher Preis pro Jahr ({year_col})")
    plt.xlabel(year_col)
    plt.ylabel(f"Durchschnittlicher {price_column}")
    plt.show()

