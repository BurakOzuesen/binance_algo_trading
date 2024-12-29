from binance.client import Client
import random
import time
from datetime import datetime
import math
from playsound import playsound


# Binance API anahtarları
API_KEY = ''
API_SECRET = ''

# Binance istemcisi oluştur
client = Client(API_KEY, API_SECRET)

# TRY çiftlerinden rastgele bir kripto para seç
def select_random_symbol(last_symbol=None):
    exchange_info = client.get_exchange_info()
    symbols = [symbol['symbol'] for symbol in exchange_info['symbols'] if symbol['symbol'].endswith('TRY')]
    if last_symbol:
        symbols = [s for s in symbols if s != last_symbol]
    return random.choice(symbols) if symbols else None

# Lot size bilgisi alınır
def get_lot_size(symbol):
    exchange_info = client.get_exchange_info()
    for s in exchange_info['symbols']:
        if s['symbol'] == symbol:
            for filter_ in s['filters']:
                if filter_['filterType'] == 'LOT_SIZE':
                    return float(filter_['stepSize'])
    return None  # Eğer lot size bulunamazsa

# Alım miktarını Binance'in istediği formatta yuvarla
def round_quantity(quantity, step_size):
    precision = int(round(-math.log(step_size, 10), 0))  # Hassasiyeti hesapla
    return round(quantity, precision)

# Dosyaya yazma fonksiyonu
def write_to_file(data):
    with open("trade_results.txt", "a") as file:
        file.write(data + "\n")

# Bot işlemleri
def trade_bot():

    account_balance = client.get_account()
    # TRY bakiyesini bul
    try_balance = 0
    for balance in account_balance['balances']:
        if balance['asset'] == 'TRY':
            try_balance = float(balance['free'])  # Kullanılabilir bakiye
            break

    last_symbol = None  # En son işlem yapılan kripto para
    trade_amount_try = math.floor(try_balance)  # İşlem için kullanılacak TRY miktarı
    virtual_balance = math.floor(try_balance)  # Sanal başlangıç bakiyesi (100 TRY)

    while True:
        # Yeni bir kripto para seç
        print("Trade_amount_try", trade_amount_try)
        symbol = select_random_symbol(last_symbol)
        if not symbol:
            print("Seçilebilecek başka kripto kalmadı!")
            break
        print(f"Yeni seçilen kripto para: {symbol}")

        last_symbol = symbol

        lot_size = get_lot_size(symbol)
        if not lot_size:
            print(f"{symbol} için lot size bulunamadı, başka bir kripto seçiliyor.")
            continue

        # Alım işlemi (Piyasa fiyatından)
        ticker = client.get_ticker(symbol=symbol)
        buy_price = float(ticker['lastPrice'])
        try:
            quantity = trade_amount_try / buy_price
        except Exception as e:
            continue
        quantity = float(round_quantity(quantity, lot_size))  # Hassasiyeti yuvarla
        if quantity <= 0:
            continue
        
        print(quantity)
        print(buy_price)
        print(quantity*buy_price)
        client.order_market_buy(symbol=symbol, quantity=quantity)  # Gerçek alım işlemi için aktif edin
        print(f"{symbol} için {quantity*buy_price} TRY ile alım yapıldı. Birim fiyat: {buy_price} TRY, Miktar: {quantity}")

        symbol_balance = 0
        for balance in account_balance['balances']:
            if balance['asset'] == symbol[:-3]:
                symbol_balance = float(balance['free'])  # Kullanılabilir bakiye
                break
        
        # Alt değere yuvarlama
        rounded_down = math.floor(symbol_balance / lot_size) * lot_size
        # İşlemin başlangıç zamanı
        start_time = datetime.now()

        # Hedef fiyat ve stop-loss hesaplaması
        target_price = buy_price * 1.01  # %1 artış
        stop_loss_price = buy_price * 0.95  # %1 düşüş
        print(f"Hedef fiyat: {target_price:.2f} TRY, Stop-loss fiyatı: {stop_loss_price:.2f} TRY")

        while True:
            ticker = client.get_ticker(symbol=symbol)

            current_price = float(ticker['lastPrice'])
            print(f"Güncel fiyat: {current_price} TRY")

            # Hedef fiyat kontrolü (Kar realizasyonu)
            if current_price >= target_price:
                profit = (current_price - buy_price) * quantity
                # print("Profit = ", profit)
                print(f"{symbol} hedef fiyata ulaştı, satış yapılıyor. Kazanç: {current_price - buy_price:.2f} TRY")
                playsound('coin.mp3')  # Ses dosyasının yolunu belirtin
                client.order_market_sell(symbol=symbol, quantity=rounded_down)  # Gerçek satış işlemi için aktif edin
                virtual_balance += profit  # Sanal bakiye güncelleme
                break

            # Stop-loss kontrolü
            if current_price <= stop_loss_price:
                profit = (current_price - buy_price) * quantity
                # print("Profit = ", profit)
                playsound('coin.mp3')  # Ses dosyasının yolunu belirtin
                client.order_market_sell(symbol=symbol, quantity=rounded_down)  # Gerçek satış işlemi için aktif edin
                print(f"{symbol} stop-loss fiyatına ulaştı, satış yapılıyor. Zarar: {buy_price - current_price:.2f} TRY")
                virtual_balance += profit  # Sanal bakiye güncelleme
                break

            time.sleep(5)  # 5 saniyede bir fiyat kontrolü

        # İşlemin bitiş zamanı ve süre
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60  # Dakika cinsinden süre
        print(f"{symbol} işlemi tamamlandı. İşlem süresi: {duration:.2f} dakika.")
        print(f"Güncel sanal bakiye: {virtual_balance:.2f} TRY")

        # Sonuçları dosyaya yaz
        result = (f"Kripto: {symbol}, Başlangıç: {start_time}, Bitiş: {end_time}, Süre: {duration:.2f} dakika, "
                  f"Kâr/Zarar: {virtual_balance - 100:.2f} TRY, Güncel Bakiye: {virtual_balance:.2f} TRY")
        write_to_file(result)

        print(f"{symbol} için sonuçlar dosyaya yazıldı. Yeni kripto para seçiliyor...\n")

# Botu başlat
trade_bot()