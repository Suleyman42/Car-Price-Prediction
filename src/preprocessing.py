import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
    fehlende Kilometerstände (NaN) durch den durchschnittlichen Kilometerstand (Mean) ersetzen,
    aber nur innerhalb der Gruppe mit demselben Erstzulassungsjahr
    (z.B nur 2019er Autos mit dem Durchschnitt der 2019er Autos, usw.)

    🇩🇪 Füllt NaN-Werte in 'Kilometerstand' mit dem Durchschnitt pro 'Erstzulassung'.
    🇬🇧 Fills NaN values in 'Kilometerstand' using the mean per 'Erstzulassung' group.

    Parameter:
        df (pd.DataFrame): DataFrame mit den Spalten 'Erstzulassung' und 'Kilometerstand'.

    Rückgabe:
        pd.DataFrame: DataFrame mit ersetzten Werten.
    """
    df['Kilometerstand'] = df.groupby('Erstzulassung')['Kilometerstand'].transform(
        lambda x: x.fillna(x.mean())
    )
    return df

# 1. Funktion: Fülle mit dem häufigsten Getriebe-Typ je Gruppe (Modell + Erstzulassung)
def fill_gear_by_group(df):
    df['Getriebe'] = df.groupby(['Modell', 'Erstzulassung'])['Getriebe'] \
                       .transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Unbekannt'))
    return df