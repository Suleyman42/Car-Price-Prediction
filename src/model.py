from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


class CatBoostPriceRegressor:
    """
    🇩🇪 Diese Klasse kapselt den CatBoostRegressor-Workflow zur Vorhersage von Autopreisen.
    🇬🇧 This class encapsulates the CatBoostRegressor workflow for predicting car prices.
    """

    def __init__(
        self,
        iterations=1600,
        learning_rate=0.01,
        depth=16,
        l2_leaf_reg=5,
        loss_function='MAE',
        random_seed=42,
        verbose=200
    ):
        """
        🇩🇪 Konstruktor: Initialisiert die CatBoost-Parameter.
        🇬🇧 Constructor: Initializes the CatBoost parameters.
        """
        self.model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            loss_function=loss_function,
            random_seed=random_seed,
            verbose=verbose
        )
        self.categorical_columns = None  # Standardmäßig keine kategorischen Spalten
        self.train_pool = None  # Wird beim fit() gesetzt

    def train_test_split(self, X, y, test_size, store=True):
        """
        🇩🇪 Teilt die Daten in Training und Test.
        🇬🇧 Splits the data into training and test sets.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        if store:
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
        return X_train, X_test, y_train, y_test

    def set_categorical_columns(self, cat_cols):
        self.categorical_columns = cat_cols

    def fit(self):
        """
        🇩🇪 Trainiert das CatBoost-Modell auf Trainingsdaten.
        🇬🇧 Trains the CatBoost model on training data.
        """
        # Achtung: Stelle sicher, dass self.categorical_columns vor dem Fit gesetzt ist!
        self.train_pool = Pool(self.X_train, self.y_train, cat_features=self.categorical_columns)
        self.model.fit(self.train_pool)

    def evaluate(self, X_train, y_train, X_test, y_test):
        """
        🇩🇪 Bewertet das Modell auf Trainings- und Testdaten.
        🇬🇧 Evaluates the model on training and test data.
        """
        # Vorhersagen berechnen
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        # Trainingsmetriken berechnen
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
        train_r2 = r2_score(y_train, y_train_pred)

        # Testmetriken berechnen
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
        test_r2 = r2_score(y_test, y_test_pred)

        # Metriken ausgeben
        print("🔹 TRAIN vs TEST METRIKEN / METRICS")
        print(f"MAE:   TRAIN {train_mae:.2f}   TEST {test_mae:.2f}")
        print(f"MSE:   TRAIN {train_mse:.2f}   TEST {test_mse:.2f}")
        print(f"RMSE:  TRAIN {train_rmse:.2f}   TEST {test_rmse:.2f}")
        print(f"R²:    TRAIN {train_r2:.2f}   TEST {test_r2:.2f}")

        return {
            "train": {"MAE": train_mae, "MSE": train_mse, "RMSE": train_rmse, "R2": train_r2},
            "test": {"MAE": test_mae, "MSE": test_mse, "RMSE": test_rmse, "R2": test_r2}
        }

    def plot_predictions(self):
        """
        🇩🇪 Erstellt einen Scatterplot der tatsächlichen vs. vorhergesagten Werte
        im linearen und log-log Maßstab.
        🇬🇧 Creates scatter plots of actual vs. predicted values
        in linear and log-log scales.
        """
        # Berechne Test-Vorhersagen
        y_test_pred = self.model.predict(self.X_test)

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        # === 1) Linearer Plot (gezoomt) ===
        axs[0].scatter(self.y_test, y_test_pred, alpha=0.6)
        axs[0].plot(
            [self.y_test.min(), self.y_test.max()],
            [self.y_test.min(), self.y_test.max()],
            'r--', lw=2
        )
        axs[0].set_title("Linear Scale (Zoomed)")
        axs[0].set_xlabel("Actual Values")
        axs[0].set_ylabel("Predicted Values")
        axs[0].set_xlim([0, 300000])
        axs[0].set_ylim([0, 300000])

        # === 2) Log-Log Plot ===
        axs[1].scatter(self.y_test, y_test_pred, alpha=0.6)
        axs[1].plot(
            [self.y_test.min(), self.y_test.max()],
            [self.y_test.min(), self.y_test.max()],
            'r--', lw=2
        )
        axs[1].set_xscale('log')
        axs[1].set_yscale('log')
        axs[1].set_title("Log-Log Scale")
        axs[1].set_xlabel("Actual Values (log scale)")
        axs[1].set_ylabel("Predicted Values (log scale)")

        plt.tight_layout()
        plt.show()

