import os
import random
import requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import csv
import matplotlib
matplotlib.use("Agg")  # Tkinter backend yerine non-GUI backend kullan


CRYPTO_LIMIT = 64


# Klasör Oluşturma
os.makedirs("output_csv", exist_ok=True)

# Kripto Çiftlerini TXT Dosyasından Rastgele Seç
with open("usd_pairs.txt", "r") as file:
    crypto_list = [line.strip().replace("USDT", "-USD") for line in file.readlines()[0:]]
random.shuffle(crypto_list)

# %80 Eğitim - %20 Test Split
train_size = int(len(crypto_list) * 0.8)
train_list = crypto_list[:train_size]
test_list = crypto_list[train_size:]


def export_to_csv_results(metrics_dict, file_path):
    # Başlıklar ve verileri hazırlama
    headers = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    # Dosyaya ekleme yapma
    file_exists = os.path.exists(file_path)  # Dosyanın var olup olmadığını kontrol et

    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        
        if not file_exists:  # Dosya yoksa, başlıkları yaz
            writer.writerow(headers)
        
        # Verileri ekle
        writer.writerow(values)


# Simülasyon Fonksiyonu
def simulate_trading(df, trigger_percent=0.05, sell_drop_percent=0.01, initial_capital=1000.0):
    capital = initial_capital
    trade_counter = 0
    in_trade = False
    buy_price = np.nan
    local_min_trade = np.nan
    local_max_trade = np.nan

    df["trade_counter"] = 0
    df["local_min"] = np.nan
    df["trigger"] = np.nan
    df["triggered"] = False
    df["buy_price"] = np.nan
    df["local_max"] = np.nan
    df["sell_target"] = np.nan
    df["sell_signal"] = False
    df["capital"] = np.nan
    df["in_trade"] = False

    for idx, row in df.iterrows():
        current_close = row["Close"]

        if not in_trade:
            df.at[idx, "in_trade"] = False
            df.at[idx, "capital"] = capital

            if np.isnan(local_min_trade) or current_close < local_min_trade:
                local_min_trade = current_close
            df.at[idx, "local_min"] = local_min_trade

            trigger_price = local_min_trade * (1 + trigger_percent)
            df.at[idx, "trigger"] = trigger_price

            if current_close >= trigger_price:
                in_trade = True
                trade_counter += 1
                buy_price = current_close
                local_max_trade = current_close

                df.at[idx, "triggered"] = True
                df.at[idx, "buy_price"] = buy_price
                
            else:
                df.at[idx, "triggered"] = False
        else:
            df.at[idx, "in_trade"] = True
            current_equity = capital * (current_close / buy_price)
            df.at[idx, "capital"] = current_equity

            if current_close > local_max_trade:
                local_max_trade = current_close
            df.at[idx, "local_max"] = local_max_trade

            sell_target = local_max_trade * (1 - sell_drop_percent)
            df.at[idx, "sell_target"] = sell_target

            if current_close <= sell_target and sell_target > buy_price:
                df.at[idx, "sell_signal"] = True
                capital = capital * (current_close / buy_price)
                in_trade = False
                buy_price = np.nan
                local_min_trade = np.nan
                local_max_trade = np.nan
            else:
                df.at[idx, "sell_signal"] = False

        df.at[idx, "trade_counter"] = trade_counter

    initial_price = df.iloc[0]["Close"]
    btc_hold_amount = initial_capital / initial_price
    df["no_trade_capital"] = df["Close"] * btc_hold_amount

    return df

def evaluate_params(df, params, initial_capital=1000.0):
    result_df = simulate_trading(df.copy(), **params, initial_capital=initial_capital)
    final_capital = result_df["capital"].iloc[-1]
    return params, final_capital

def parallel_optimize_parameters(df, param_grid, n_jobs=-1, initial_capital=1000.0):
    grid = list(ParameterGrid(param_grid))
    # tqdm ile ilerleme çubuğu ekliyoruz
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(evaluate_params)(df, params, initial_capital)
        for params in tqdm(grid, desc="Optimizing parameters")
    )
    best_params, best_result = None, None
    results_capital = []
    for params, final_capital in results:
        results_capital.append(final_capital)
        if best_result is None or final_capital > best_result:
            best_result = final_capital
            best_params = params
    average_result = np.mean(results_capital)
    return best_params, best_result, average_result

def compute_performance_metrics(df, initial_capital=1000.0):
    final_capital = df["capital"].iloc[-1]
    trade_count = df["trade_counter"].iloc[-1]
    buy_hold_return = df["no_trade_capital"].iloc[-1]

    capital_series = df["capital"]
    cumulative_max = capital_series.cummax()
    drawdowns = (capital_series - cumulative_max) / cumulative_max
    max_drawdown = drawdowns.min()

    returns = capital_series.pct_change().dropna()
    avg_return = returns.mean()
    std_return = returns.std()
    sharpe_ratio = avg_return / std_return if std_return != 0 else np.nan

    in_trade = df.iloc[-1]["in_trade"]
    metrics = {
        "final_capital": final_capital,
        "buy_hold_return": buy_hold_return,
        "trade_count": trade_count,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "in_trade": in_trade,
    }
    return metrics

def plot_results(df, ticker_name):
    plt.figure(figsize=(12, 6))
    plt.plot(df["Datetime"], df["capital"], label="Trading Strategy")
    plt.plot(
        df["Datetime"], df["no_trade_capital"], label="Buy & Hold", linestyle="dashed"
    )

    buy_signals = df[df["triggered"] == True]
    plt.scatter(
        buy_signals["Datetime"],
        buy_signals["capital"],
        color="green",
        marker="^",
        label="Buy Signal",
        s=100,
    )

    sell_signals = df[df["sell_signal"] == True]
    plt.scatter(
        sell_signals["Datetime"],
        sell_signals["capital"],
        color="red",
        marker="v",
        label="Sell Signal",
        s=100,
    )

    plt.xlabel("Datetime")
    plt.ylabel("Portfolio Value")
    plt.title("Trading Strategy vs Buy & Hold with Trade Signals")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    os.makedirs("./graphics/", exist_ok=True)  # Grafik klasörünü oluşturur
    plt.savefig(f"./graphics/{ticker_name}.jpg")  # Dosyayı kaydeder
    plt.close()  # Belleği temizler
    # plt.show()

# --- Yeni Geliştirme: Drawdown Grafiği --- #
def plot_drawdown(df):
    plt.figure(figsize=(12, 6))
    capital_series = df["capital"]
    cumulative_max = capital_series.cummax()
    drawdown = (capital_series - cumulative_max) / cumulative_max
    plt.plot(df["Datetime"], drawdown, label="Drawdown", color="red")
    plt.xlabel("Datetime")
    plt.ylabel("Drawdown")
    plt.title("Drawdown Plot")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    # plt.show()

# --- Yeni Geliştirme: Optimizasyon Sonuçlarını Toplama ve Heatmap Görselleştirmesi --- #
def collect_optimization_results(df, param_grid, n_jobs=-1, initial_capital=1000.0):
    grid = list(ParameterGrid(param_grid))
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(evaluate_params)(df, params, initial_capital)
        for params in tqdm(grid, desc="Collecting optimization results")
    )
    results_list = []
    for params, final_capital in results:
        res = params.copy()
        res["final_capital"] = final_capital
        results_list.append(res)
    return pd.DataFrame(results_list)

def plot_optimization_heatmap(opt_results_df):
    # Pivot tablo: trigger_percent (satır) ve sell_drop_percent (sütun) bazında final_capital
    pivot_table = opt_results_df.pivot(
        index="trigger_percent", columns="sell_drop_percent", values="final_capital"
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Optimization Heatmap: Final Capital")
    plt.xlabel("Sell Drop Percent")
    plt.ylabel("Trigger Percent")
    # plt.show()

# Eğitim ve Test Simülasyonu
def run_simulation(train_list, test_list, param_grid):
    stats = []
    for crypto in train_list:
        data = yf.download(crypto, period="60d", interval="5m")
        data.reset_index(inplace=True)
        data.columns = ["Datetime" ,"Open", "High", "Low", "Close", "Volume"]
        # %80 Eğitim - %20 Test Split
        train_size = int(len(data) * 0.8)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]

        best_params, _ = optimize_parameters(train_data, param_grid)
        train_capital, _ = simulate_trading(train_data, **best_params)

        test_capital, trade_count = simulate_trading(test_data, **best_params)
        stats.append({
            "crypto": crypto,
            "best_params": best_params,
            "train_capital": train_capital,
            "test_capital": test_capital,
            "trade_count": trade_count
        })
        break

    return pd.DataFrame(stats)

# CSV Yazdırma
def export_to_csv(df, filename):
    df.to_csv(f"output_csv/{filename}", index=False)
    print(f"{filename} başarıyla kaydedildi.")

# Görselleştirme Fonksiyonları
def plot_performance(df):
    plt.figure(figsize=(10, 5))
    sns.barplot(x="crypto", y="test_capital", data=df)
    plt.xticks(rotation=90)
    plt.title("Test Seti Performansı")
    # plt.show()

# Ana Akış
data_frames = {}
# Parametre ızgarası
param_grid = {
    "trigger_percent": [0.01, 0.03, 0.05, 0.07, 0.09],
    "sell_drop_percent": [0.005, 0.01, 0.015, 0.02, 0.025],
}

crpto_count = 0
for crypto in crypto_list:

    print(f"Downloading data for {crypto}...")
    try:
        data = yf.download(crypto, period="5y", interval="1d")
        data.reset_index(inplace=True)
        data.columns = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
            
        print(crypto, data.shape)
        if data.shape[0] < 1643:
            print(f"Not enough data for {crypto}, skipping...")
            continue

        data_frames[crypto] = data
       
        crpto_count += 1
        if crpto_count >= CRYPTO_LIMIT:
            break

    except:
        continue


for crypto, df in data_frames.items():
    
    print(f"Processing {crypto}...")
    df.reset_index(inplace=True)

    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)
    best_params, best_result, avg_result = parallel_optimize_parameters(
        train_df, param_grid, n_jobs=-1
    )

    optimized_df = simulate_trading(test_df, **best_params)
    # optimized_df.to_csv(f".\output\{crypto}_optimized.csv", index=False)

    # Simülasyon sonuçlarını CSV dosyasına kaydet
    export_to_csv(optimized_df, f"{crypto}_simulation_results.csv")
    

    print(f"\n--- {crypto} ---")
    print(f"Best params: {best_params} with final capital: {best_result}")
    print(f"Average final capital (all parameter combinations): {avg_result}")

    metrics = compute_performance_metrics(optimized_df)
    print("Performance Metrics:")

    print(f"{crypto} simulation results saved.")


    print(crypto, best_params, best_result, avg_result)

    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    metrics = {
        "crypto_pair": crypto,
        "sell_drop_percent": best_params["sell_drop_percent"],
        "trigger_percent": best_params["trigger_percent"],
        "initial_capital": 1000,
        "final_capital": metrics["final_capital"],
        "buy_hold_return": metrics["buy_hold_return"],
        "trade_count": metrics["trade_count"],
        "max_drawdown": metrics["max_drawdown"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "in_tade": metrics["in_trade"],
}
    export_to_csv_results(metrics, "output_csv/trading_results.csv")
    plot_results(optimized_df, crypto)
    # plot_drawdown(optimized_df)

    # Yeni: Tüm optimizasyon sonuçlarını toplayıp ısı haritası olarak görselleştiriyoruz.
    # opt_results_df = collect_optimization_results(df, param_grid, n_jobs=-1)
    # plot_optimization_heatmap(opt_results_df)
# results_df = run_simulation(train_list, test_list, param_grid)
# export_to_csv(results_df, "trading_simulation_results.csv")
# plot_performance(results_df)