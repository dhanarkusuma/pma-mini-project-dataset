# this class responsible for data processing tasks
# like creating lag value, data cleanign and so on

import pandas as pd
from sklearn.model_selection import train_test_split


class DataProcessing:
    def __init__(
        self,
        train_path="datasets/train.csv",
        test_path="datasets/test.csv",
        target_col="Y_jumlah_kasus",
    ):
        self.train_path = train_path
        self.test_path = test_path
        self.target_col = target_col

        self.x_train = pd.DataFrame([])
        self.y_train = pd.DataFrame([])
        self.x_test = pd.DataFrame([])
        self.y_test = pd.DataFrame([])
        self.x_validation = pd.DataFrame([])
        self.y_validation = pd.DataFrame([])

    def _initial_preprocess(self, df):
        """
        Melakukan Pre-processing Awal:
        1. Mengubah kolom 'Bulan' menjadi tipe datetime dan menjadikannya index.
        2. Mengganti nama kolom.
        """
        # Mengubah kolom 'Bulan' menjadi tipe datetime (format '%b-%y') dan menjadikannya index
        df["bulan"] = pd.to_datetime(df["bulan"], format="%b-%y")
        df.set_index("bulan", inplace=True)

        # Ganti nama kolom (Asumsi urutan kolom adalah [X1, X2, Y] setelah 'Bulan' diangkat menjadi index)
        try:
            df.columns = ["X1_curah_hujan", "X2_lama_hujan", self.target_col]
        except ValueError:
            print(
                "PERINGATAN: Jumlah kolom tidak sesuai setelah set_index. Periksa data CSV."
            )
            return df

        return df

    def create_time_series_features(self, df):
        """
        Membuat fitur deret waktu (Musiman, Tren, Lag) dari index datetime.
        """
        # Fitur Musiman dan Tren
        df["bulan"] = df.index.month
        df["tahun"] = df.index.year
        df["kuartal"] = df.index.quarter

        # Fitur Lag (nilai Y bulan sebelumnya), sangat penting untuk forecasting
        df["Y_lag_1"] = df[self.target_col].shift(1)
        df["Y_lag_2"] = df[self.target_col].shift(2)
        df["Y_lag_3"] = df[self.target_col].shift(3)

        return df

    def initialize_data(self):
        """
        Memuat, memproses, feature engineering, dan membagi data.
        """
        print(f"Memuat dan memproses data dari: {self.train_path} dan {self.test_path}")

        # 1. Memuat Data Training dan Testing
        try:
            df_train = pd.read_csv(self.train_path)
            df_test = pd.read_csv(self.test_path)
        except FileNotFoundError as e:
            print(f"ERROR: File tidak ditemukan: {e}")
            return

        # 2. Pre-processing Awal
        df_train = self._initial_preprocess(df_train.copy())
        df_test = self._initial_preprocess(df_test.copy())

        # 3. Menggabungkan data sementara untuk Feature Engineering yang Konsisten
        # Ini memastikan fitur Lag_1 di baris pertama test set mengambil nilai terakhir dari train set.
        df_full = pd.concat([df_train, df_test])
        train_idx = df_train.index
        test_idx = df_test.index

        # 4. Pembuatan Fitur (Feature Engineering) pada data gabungan
        df_full_feat = self.create_time_series_features(df_full.copy())

        
        # 5. Memisahkan Kembali Data Training dan Testing
        df_train_feat = df_full_feat.loc[train_idx].copy()
        df_test_feat = df_full_feat.loc[test_idx].copy()

        # 6. Data Cleaning
        # Hapus baris NaN yang muncul akibat fitur lag (biasanya 3 baris pertama di train set)
        df_train_feat.dropna(inplace=True)
        df_test_feat.dropna(inplace=True)

        print(
            f"Data Training setelah feature engineering & cleaning: {df_train_feat.shape}"
        )
        print(
            f"Data Testing setelah feature engineering & cleaning: {df_test_feat.shape}"
        )

        # 7. Membagi X (Fitur) dan y (Target)
        # Semua kolom selain kolom target adalah fitur (X)
        features = [col for col in df_train_feat.columns if col != self.target_col]

        x_train, y_train, x_test, y_test = train_test_split(
            df_train_feat.drop(columns=[self.target_col]),
            df_train_feat[self.target_col],
            test_size=0.2,
            random_state=42,
        )

        self.x_train = x_train
        self.y_train = y_train

        self.x_test = x_test
        self.y_test = y_test

        self.x_validation = df_test_feat[features]
        self.y_validation = df_test_feat[self.target_col]

        print("Inisialisasi data selesai.")

    def get_x_train(self):
        return self.x_train

    def get_y_train(self):
        return self.y_train

    def get_x_test(self):
        return self.x_test

    def get_y_test(self):
        return self.y_test
    
    def get_x_validation(self):
        return self.x_validation
    
    def get_y_validation(self):
        return self.y_validation
