import pandas as pd
import numpy as np
import re
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder

from scipy.stats import f_oneway, chi2_contingency
def remove_month_from_registration_date(df):
    """
    Entfernt den Monat aus 'Erstzulassung' und extrahiert das Jahr als Int64.
    Unterstützt string, datetime und gemischte Typen.
    """

    df["Erstzulassung"] = df["Erstzulassung"].astype(str)
    df["Erstzulassung"] = df["Erstzulassung"].str.extract(r"(\d{4})")
    df["Erstzulassung"] = pd.to_numeric(df["Erstzulassung"], errors="coerce").astype("Int64")

    return df


def fill_kilometer_with_group_mean(df):
    """
    🇩🇪 Füllt fehlende Kilometerstände ('Kilometerstand') in zwei Schritten:
        1. Wenn das Erstzulassungsjahr 2017, 2018 oder 2019 ist → ersetze NaN mit 0
        2. Für alle anderen Jahre → ersetze NaN mit dem Durchschnitt (Mean) innerhalb des jeweiligen Jahres

    🇬🇧 Fills missing mileage values ('Kilometerstand') in two steps:
        1. If the registration year is 2017, 2018, or 2019 → replace NaNs with 0
        2. For all other years → replace NaNs with the mean mileage for that year

    Parameter:
    - df (pd.DataFrame): DataFrame containing 'Erstzulassung' and 'Kilometerstand' columns.

    Rückgabe / Returns:
    - pd.DataFrame: A copy of the DataFrame with filled 'Kilometerstand' values.
    """
    df = df.copy()  # Sicherheitshalber mit einer Kopie arbeiten

    # Liste der Jahre, bei denen fehlende Kilometer mit 0 ersetzt werden sollen
    special_years = [2017, 2018, 2019]

    # Schritt 1: NaN in 'Kilometerstand' auf 0 setzen für bestimmte Jahre
    mask_special = df['Erstzulassung'].isin(special_years) & df['Kilometerstand'].isna()
    df.loc[mask_special, 'Kilometerstand'] = 0

    # Schritt 2: Für alle anderen fehlenden Werte den Gruppendurchschnitt verwenden
    df['Kilometerstand'] = df.groupby('Erstzulassung')['Kilometerstand'].transform(
        lambda x: x.fillna(x.mean())  # Nur wenn noch NaN übrig ist
    )

    return df




def fill_missing_categories(df, columns, fill_value="missing"):
    """
       Replaces missing values in categorical columns with a specified category.
       Ersetzt fehlende Werte in mehreren kategorialen Spalten durch eine definierte Kategorie.
       """
    df = df.copy()

    for col in columns:
        df[col] = df[col].fillna(fill_value)
        # Direkt nach dem Füllen: Typ ableiten (macht aus object ggf. string oder category)
        df[col] = df[col].infer_objects(copy=False)

    return df

def fill_categorical_by_group(df, group_cols, target_col, default_fill):
    """
    Füllt fehlende Werte in einer kategorialen Spalte basierend auf dem Modus (häufigster Wert)
    innerhalb von Gruppen, die durch eine oder mehrere andere Spalten definiert sind.
    Wird kein Modus gefunden, wird ein Standardwert verwendet.

    Parameters:
    - df (pd.DataFrame): Der zu bearbeitende DataFrame.
    - group_cols (list of str): Liste von Spalten zum Gruppieren (z. B. ['Modell', 'Jahr']).
    - target_col (str): Die Spalte, in der fehlende Werte ersetzt werden sollen.
    - default_fill (str): Ersatzwert, wenn kein Modus gefunden wird (z. B. 'missing').

    Returns:
    - pd.DataFrame: Eine Kopie des DataFrames mit ersetzten Werten in der Zielspalte.
    """
    df = df.copy()


    df[target_col] = df.groupby(group_cols)[target_col] \
        .transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else np.nan))


    df[target_col] = df[target_col].fillna(default_fill)

    return df



def anova_test(df, cat_col, num_col):
    """
    Perform ANOVA test between a categorical and a numerical variable.
    Führt einen ANOVA-Test zwischen einer kategorialen und einer numerischen Spalte durch.

    Parameters:
    df (pd.DataFrame): The dataset / Datensatz
    cat_col (str): Name of the categorical column / Name der kategorialen Spalte
    num_col (str): Name of the numerical column / Name der numerischen Spalte

    Returns:
    f_stat (float): F-statistic / F-Statistik
    p_val (float): p-value / p-Wert
    """
    groups = [df[df[cat_col] == val][num_col].dropna() for val in df[cat_col].unique()]
    f_stat, p_val = f_oneway(*groups)

    # Print the results / Ergebnisse ausgeben
    print("ANOVA Test Result:")
    print("Ergebnis des ANOVA-Tests:")
    print(f"F-Statistic / F-Statistik: {f_stat:.2f}")
    print(f"P-Value / p-Wert: {p_val:.4f}")

    return f_stat, p_val




def calculate_mutual_info_regression(df, target_column):
    """
    Berechnet Mutual Information Scores zwischen Features und Zielvariable
    für Regressionsaufgaben und gibt die Ergebnisse aus.
    """
    df_copy = df.copy()
    X = df_copy.drop(columns=[target_column])
    y = df_copy[target_column]

    # Kategoriale Features in Strings umwandeln + Label Encoding
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = X[col].astype(str)
        X[col] = LabelEncoder().fit_transform(X[col])

    # Fehlende Werte auffüllen (z. B. mit 0 oder Median)
    X.fillna(0, inplace=True)

    # Auch Zielvariable sollte keine NaNs haben
    y = y.fillna(y.median())

    # Mutual Information berechnen
    mi_scores = mutual_info_regression(X, y)

    mi_df = pd.DataFrame({'Feature': X.columns, 'MI Score': mi_scores})
    mi_df.sort_values('MI Score', ascending=False, inplace=True)

    print("\n📊 Mutual Information (Regression):")
    print(mi_df.to_string(index=False))

    return mi_df

def count_origin_per_make_model(df, origin_column="Getriebe"):
    """
    Zählt, wie oft jede Herkunfts-Kategorie ('Herkunft') für jede Kombination aus 'make' und 'model' vorkommt.
    Berücksichtigt auch fehlende Werte (NaN) und stellt das Ergebnis als Kreuztabelle dar.

    Counts how often each category in the 'origin_column' appears for each 'make' and 'model' combination.
    Includes missing values (NaN) and returns the result as a pivoted DataFrame.

    Parameter:
    - df (pd.DataFrame): Der Eingabedatensatz / The input DataFrame
    - origin_column (str): Name der Spalte mit Herkunftsinformationen / Name of the origin column

    Rückgabe:
    - pd.DataFrame: Pivotierte Tabelle mit Herkunfts-Kategorien als Spalten
                   / Pivoted table with origin categories as columns
    """
    grouped = (
        df.groupby(["model", "make"])[origin_column]
        .value_counts(dropna=False)
        .unstack(fill_value=0)
    )
    return grouped

def clean_numeric_column(df, column_name, output_type="float"):
    """
    Extrahiert numerische Werte aus einer Textspalte und wandelt sie in float oder Int64 um.
    Behandelt NaN-Werte korrekt und ignoriert Einheiten/Sonderzeichen.

    Beispiele:
        "6.208 cm³" → 6208.0
        "≈ 328 g/km" → 328
        NaN → NaN

    Parameter:
    ----------
    df : pd.DataFrame
        Der DataFrame mit der Spalte, die bereinigt werden soll.

    column_name : str
        Name der Spalte mit gemischtem Text + Zahlen + Einheiten.

    output_type : str, optional ("float" oder "int")
        Ob das Ergebnis float oder Int64 sein soll (standard: "float").

    Rückgabe:
    ---------
    pd.DataFrame :
        Kopie des DataFrames mit bereinigter numerischer Spalte.
    """
    df = df.copy()

    # Nur Strings verarbeiten, NaN bleibt erhalten
    cleaned = (
        df[column_name]
        .where(df[column_name].notna())  # NaN behalten
        .astype(str)
        .str.replace(r"[^\d,\.]", "", regex=True)  # nur Ziffern, Punkt, Komma
        .str.replace(",", ".", regex=False)  # Komma in Punkt
    )

    # In numerischen Typ konvertieren, NaN bleibt erhalten
    numeric = pd.to_numeric(cleaned, errors="coerce")

    if output_type == "int":
        df[column_name] = numeric.astype("Int64")  # Pandas-Int mit NaN-Unterstützung
    else:
        df[column_name] = numeric  # float (standard)

    return df

def extract_avg_fuel_consumption(df, column_name="Verbrauch"):
    """
    Extrahiert Verbrauchswerte (kombiniert, innerorts, außerorts) aus einer Textspalte,
    erstellt daraus separate Spalten sowie eine Durchschnittsspalte und entfernt die Originalspalte.

    Parameters:
    -----------
    df : pandas.DataFrame
        Das Eingabe-DataFrame mit Verbrauchsangaben als Text.

    column_name : str, optional (default="Verbrauch")
        Der Name der Spalte, die verarbeitet werden soll.

    Returns:
    --------
    pandas.DataFrame
        Kopie des DataFrames mit neuen Spalten:
        - 'kombiniert'
        - 'innerorts'
        - 'außerorts'
        - 'Verbrauch_avg'
        und ohne die Originalspalte.
    """

    df = df.copy()

    def parse_verbrauch(value):
        try:
            if pd.isna(value):
                return None, None, None

            # Entfernen störender Zeichen
            value = str(value)
            value = value.replace('\u2009', '').replace('\u2248', '').replace('*', '')

            # Verbrauchswerte extrahieren
            kombiniert = re.search(r'([\d,]+)\s*l/100km\s*\(kombiniert\)', value)
            innerorts = re.search(r'([\d,]+)\s*l/100km\s*\(innerorts\)', value)
            ausserorts = re.search(r'([\d,]+)\s*l/100km\s*\(außerorts\)', value)

            # Umwandlung zu Float
            kombiniert = float(kombiniert.group(1).replace(',', '.')) if kombiniert else None
            innerorts = float(innerorts.group(1).replace(',', '.')) if innerorts else None
            ausserorts = float(ausserorts.group(1).replace(',', '.')) if ausserorts else None

            return kombiniert, innerorts, ausserorts
        except Exception:
            return None, None, None

    # Verbrauchswerte extrahieren
    df[['kombiniert', 'innerorts', 'außerorts']] = df[column_name].apply(parse_verbrauch).apply(pd.Series)

    # Durchschnitt berechnen (nur wenn Werte vorhanden sind)
    df["Verbrauch_avg"] = df[['kombiniert', 'innerorts', 'außerorts']].mean(axis=1)

    # Originalspalte entfernen
    df.drop(columns=[column_name], inplace=True)

    return df

def preprocess_dataframe(df):
    """
    Führt alle Vorverarbeitungsschritte auf dem gegebenen DataFrame aus.
    Applies all preprocessing steps to the given DataFrame.

    Schritte / Steps:
    - Datumswerte bereinigen / Clean registration dates
    - Fehlende numerische und kategoriale Werte füllen / Fill missing numerical & categorical values
    - Textzahlen in echte Zahlen umwandeln / Convert numeric strings to real numbers

    Parameter / Parameters:
    - df (pd.DataFrame): Rohdaten / Raw input DataFrame

    Rückgabe / Returns:
    - pd.DataFrame: Bereinigter DataFrame / Cleaned DataFrame
    """
    df = df.copy()

    # 🇩🇪 Monat aus 'Erstzulassung' entfernen (nur Jahr behalten)
    # 🇬🇧 Remove month from registration date, keep only year
    df = remove_month_from_registration_date(df)

    # 🇩🇪 Kilometerstand per Jahr mitteln, fehlende Werte füllen
    # 🇬🇧 Fill missing mileage using group mean per year
    df = fill_kilometer_with_group_mean(df)

    # 🇩🇪 Kategoriale Spalten mit Modus nach Gruppen füllen
    # 🇬🇧 Fill categorical columns with group-wise mode
    df = fill_categorical_by_group(df, ["model", "Erstzulassung"], "Getriebe", "missing")
    df = fill_categorical_by_group(df, ["Kraftstoffart", "Erstzulassung", "model"], "Schadstoffklasse", "missing")
    df = fill_categorical_by_group(df, ["Kategorie", "Erstzulassung", "model"], "Anzahl Sitzplätze", 0)
    df = fill_categorical_by_group(df, ["Erstzulassung", "model"], "Klimatisierung", "missing")
    df = fill_categorical_by_group(df, ["Erstzulassung", "model"], "colour", "missing")
    df = fill_categorical_by_group(df, ["Erstzulassung", "model", "Kategorie"], "Airbags", "missing")
    df = fill_categorical_by_group(df, ["Erstzulassung", "model"], "Innenausstattung", "missing")
    df = fill_categorical_by_group(df, ["Erstzulassung", "model", "Hubraum"], "Verbrauch", "missing")
    #df = extract_avg_fuel_consumption(df, "Verbrauch")



    # 🇩🇪 Einzelne fehlende Werte mit Standardwert füllen
    # 🇬🇧 Fill missing values in specific columns with default value
    df = fill_missing_categories(df, ["Herkunft", "Verfügbarkeit"], "missing")

    # 🇩🇪 Text-basierte Zahlen wie "6.208 cm³" bereinigen → float
    # 🇬🇧 Clean text-based numeric columns like "6.208 cm³" → float
    #df = clean_numeric_column(df, "Hubraum", output_type="float")
    #df = clean_numeric_column(df, "CO2-Emissionen", output_type="float")

    # 🇩🇪 Falls nach Bereinigung noch Lücken existieren → füllen nach Leistung & Modell
    # 🇬🇧 Fill any remaining missing values using [year, model, power] group
    df = fill_categorical_by_group(df, ["Erstzulassung", "model", "Leistung"], "Hubraum", "missing")
    df = fill_categorical_by_group(df, ["Erstzulassung", "model", "Leistung"], "CO2-Emissionen", "missing")

    df = df[df['price'] <= 200000]
    return df
