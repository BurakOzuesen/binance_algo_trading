import requests

def get_usd_pairs_to_txt():
    url = "https://api.binance.com/api/v3/exchangeInfo"
    response = requests.get(url)
    
    if response.status_code == 200:
        symbols = response.json().get("symbols", [])
        usd_pairs = [symbol['symbol'] for symbol in symbols if symbol['quoteAsset'] in ['USDT']]
        
        with open("usd_pairs.txt", "w") as file:
            file.write(f"Toplam {len(usd_pairs)} çift bulundu:\n")
            for pair in usd_pairs:
                file.write(pair + "\n")
        
        print("usd_pairs.txt dosyasına başarıyla yazdırıldı!")
    else:
        print("API isteği başarısız oldu:", response.status_code)

get_usd_pairs_to_txt()
