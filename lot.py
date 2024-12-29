
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

def get_lot_size(symbol):
    exchange_info = client.get_exchange_info()
    for s in exchange_info['symbols']:
        if s['symbol'] == symbol:
            for filter_ in s['filters']:
                if filter_['filterType'] == 'LOT_SIZE':
                    return float(filter_['stepSize'])
    return None  # Eğer lot size bulunamazsa

account_balance = client.get_account()
# TRY bakiyesini bul
try_balance = 0
for balance in account_balance['balances']:
    if balance['asset'] == 'TRY':
        try_balance = float(balance['free'])  # Kullanılabilir bakiye
        break

w_balance = 0
for balance in account_balance['balances']:
    if balance['asset'] == 'ZIL':
        w_balance = float(balance['free'])  # Kullanılabilir bakiye
        break

print(w_balance)

symbol = "ZILTRY"
lot_size = get_lot_size(symbol)
print(lot_size)

balance = w_balance
rounded_down = math.floor(balance / lot_size) * lot_size
print(rounded_down)
playsound('coin.mp3')  # Ses dosyasının yolunu belirtin
