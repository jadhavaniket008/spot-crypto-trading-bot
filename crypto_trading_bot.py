import os
import time
import logging
import pandas as pd
import numpy as np
import requests
import json
import urllib.parse
from cryptography.hazmat.primitives.asymmetric import ed25519
from datetime import datetime, timedelta
from dotenv import load_dotenv
from telegram import Bot
from typing import Dict, List, Tuple
from http.server import HTTPServer, SimpleHTTPRequestHandler
import asyncio
import urllib3
from urllib.parse import urlparse, urlencode
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
COINSWITCH_API_KEY = os.getenv('COINSWITCH_API_KEY')
COINSWITCH_API_SECRET = os.getenv('COINSWITCH_API_SECRET')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
BASE_URL = "https://coinswitch.co"
WALLET_UTILIZATION = 0.9  # 90% of wallet balance
TIMEFRAME = '1440'  # Daily candles in minutes

class HealthCheckHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"OK")

class CoinSwitchAPI:
    """Handles interactions with CoinSwitch API."""
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret

    def _generate_signature(self, method: str, endpoint: str, params: dict = None, payload: dict = None) -> Tuple[str, str]:
        """Generate Ed25519 signature for API authentication."""
        try:
            epoch_time = str(int(time.time() * 1000))
            params = params or {}
            payload = payload or {}

            # Prepare final endpoint with query string
            final_endpoint = endpoint
            if method.upper() == "GET" and params:
                final_endpoint += ('&', '?')[urlparse(endpoint).query == ''] + urlencode(params)

            unquoted_endpoint = urllib.parse.unquote_plus(final_endpoint)

            # Decide how to construct signature message
            if method.upper() == "GET":
                signature_msg = method + unquoted_endpoint + epoch_time
            else:
                signature_msg = method + unquoted_endpoint + json.dumps(payload, separators=(',', ':'), sort_keys=True)

            request_bytes = signature_msg.encode('utf-8')
            secret_key_bytes = bytes.fromhex(self.api_secret)
            secret_key = ed25519.Ed25519PrivateKey.from_private_bytes(secret_key_bytes)
            signature_bytes = secret_key.sign(request_bytes)
            signature = signature_bytes.hex()

            return signature, epoch_time

        except Exception as e:
            logger.error(f"Error generating signature: {str(e)}")
            raise


    def validate_keys(self) -> bool:
        """Validate API key and secret."""
        try:
            endpoint = "/trade/api/v2/validate/keys"
            signature, epoch_time = self._generate_signature('GET', endpoint, params={})
            headers = {
                'X-AUTH-APIKEY': self.api_key,
                'X-AUTH-SIGNATURE': signature,
                'X-AUTH-EPOCH': epoch_time,  # <- FIXED
                'Content-Type': 'application/json'
            }
            logger.info(f"Validating keys with endpoint: {BASE_URL + endpoint}, headers: {headers}")
            response = requests.get(BASE_URL + endpoint, headers=headers, json={}, verify=False)
            logger.info(f"Key validation response: Status {response.status_code}, Content: {response.text}")
            response.raise_for_status()
            data = response.json()
            return data.get("message") == "Valid Access"
        except Exception as e:
            logger.error(f"Error validating keys: {str(e)}")
            return False


    def get_available_symbols(self) -> List[str]:
        """Fetch available trading pairs."""
        try:
            endpoint = "/trade/api/v2/coins"
            params = {"exchange": "coinswitchx"}  # Use "coinswitchx" for CoinSwitch exchange
            signature, epoch_time = self._generate_signature('GET', endpoint, params=params)
            headers = {
                'X-AUTH-APIKEY': self.api_key,
                'X-AUTH-SIGNATURE': signature,
                'X-AUTH-EPOCH': epoch_time,
                'Content-Type': 'application/json'
            }
            logger.info(f"Sending request to {BASE_URL + endpoint} with headers: {headers}")
            response = requests.get(BASE_URL + endpoint, headers=headers, params=params)
            logger.info(f"Response status: {response.status_code}, content: {response.text}")
            response.raise_for_status()
            data = response.json()
            if 'data' in data and data['data'].get('coinswitchx'):  # Check for coinswitchx key
                pairs = data['data']['coinswitchx']
                return [pair for pair in pairs if pair.endswith('/INR')]  # Filter for INR pairs
            else:
                logger.warning("No valid symbols found for coinswitchx")
                return []
        except Exception as e:
            logger.error(f"Error fetching available symbols: {str(e)}")
            return []

    def get_market_data(self, symbol: str) -> pd.DataFrame:
        try:
            endpoint = "/trade/api/v2/candles"
            current_time = int(time.time() * 1000)
            start_time = current_time - (100 * 24 * 60 * 60 * 1000)  # 100 days data
            params = {
                "start_time": start_time,
                "end_time": current_time,
                "symbol": symbol,
                "interval": TIMEFRAME,
                "exchange": "coinswitchx"
            }

            signature, epoch_time = self._generate_signature(
                method="GET",
                endpoint=endpoint,
                params=params,
                payload=None  # No payload for GET
            )

            full_url = f"{BASE_URL}{endpoint}"
            headers = {
                'X-AUTH-APIKEY': self.api_key,
                'X-AUTH-SIGNATURE': signature,
                'X-AUTH-EPOCH': epoch_time,
                'Content-Type': 'application/json'
            }

            logger.info(f"Fetching market data from: {full_url} with headers: {headers}")

            response = requests.get(full_url, headers=headers, params=params, verify=False)
            response.raise_for_status()
            #print("This is the response\n", response.json())
            data = response.json().get('data', [])  # Changed from 'result' to 'data'
            if not data:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()

            df = pd.DataFrame(data)
            column_map = {
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume',
                'start_time': 'timestamp'
            }
            df = df.rename(columns=column_map)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {str(e)}")
            if hasattr(e, 'response'):
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response content: {e.response.text[:500]}")
            return None


    def get_wallet_balance(self) -> float:
        """Fetch INR wallet balance."""
        try:
            endpoint = "/user/portfolio"
            signature, epoch_time = self._generate_signature('GET', endpoint, params={})
            headers = {
                'X-AUTH-APIKEY': self.api_key,
                'X-AUTH-SIGNATURE': signature,
                'X-AUTH-EPOCH': epoch_time,
                'Content-Type': 'application/json'
            }
            response = requests.get(BASE_URL + endpoint, headers=headers, json={})
            logger.info(f"Response status: {response.status_code}, content: {response.text}")
            
            response.raise_for_status()
            balances = response.json()['data']
            for balance in balances:
                if balance['currency'] == 'INR':
                    return float(balance['main_balance'])
            return 0.0
        except Exception as e:
            logger.error(f"Error fetching wallet balance: {str(e)}")
            return 0.0

    def _get_current_price(self, symbol: str) -> float:
        """Fetch current price for a symbol."""
        try:
            endpoint = "/24hr/ticker"
            params = {"symbol": symbol, "exchange": "coinswitchx"}
            signature, epoch_time = self._generate_signature('GET', endpoint, params=params)
            headers = {
                'X-AUTH-APIKEY': self.api_key,
                'X-AUTH-SIGNATURE': signature,
                'X-AUTH-EPOCH': epoch_time,
                'Content-Type': 'application/json'
            }
            response = requests.get(BASE_URL + endpoint, headers=headers, params=params)
            logger.info(f"Response status: {response.status_code}, content: {response.text}")
            response.raise_for_status()
            return float(response.json()['data'][symbol]['lastPrice'])
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {str(e)}")
            return 0.0

    def place_market_order(self, symbol: str, side: str, amount: float) -> bool:
        """Place a market order (simulated with limit order)."""
        try:
            endpoint = "/order"
            price = self._get_current_price(symbol)
            if price <= 0:
                logger.error(f"Invalid price for {symbol}")
                return False
            payload = {
                "side": side,
                "symbol": symbol,
                "type": "limit",  # CoinSwitch only supports limit orders
                "quantity": amount,
                "exchange": "coinswitchx",
                "price": price
            }
            signature, epoch_time = self._generate_signature('POST', endpoint, payload=payload)
            headers = {
                'X-AUTH-APIKEY': self.api_key,
                'X-AUTH-SIGNATURE': signature,
                'X-AUTH-EPOCH': epoch_time,
                'Content-Type': 'application/json'
            }
            response = requests.post(BASE_URL + endpoint, headers=headers, json=payload)
            logger.info(f"Response status: {response.status_code}, content: {response.text}")
            response.raise_for_status()
            logger.info(f"Placed {side} order for {symbol}: {amount} at price {price}")
            
            return True
        except Exception as e:
            logger.error(f"Error placing {side} order for {symbol}: {str(e)}")
            return False

class TechnicalIndicators:
    """Calculate technical indicators for trading signals."""
    
    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD and Signal Line."""
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return sma, upper_band, lower_band

    @staticmethod
    def calculate_sma(data: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return data.rolling(window=period).mean()

class TradingBot:
    """Main trading bot class."""
    
    def __init__(self, api: CoinSwitchAPI, telegram_token: str, chat_id: str):
        self.api = api
        self.bot = Bot(token=telegram_token)
        self.chat_id = chat_id
        self.indicators = TechnicalIndicators()

    async def send_telegram_message(self, message: str):
        """Send message to Telegram."""
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=message)
            logger.info(f"Telegram message sent: {message}")
        except Exception as e:
            logger.error(f"Error sending Telegram message: {str(e)}")

    def calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calculate historical volatility based on daily returns."""
        returns = df['close'].pct_change().dropna()
        return returns.std() * np.sqrt(252)  # Annualized volatility

    def generate_signals(self, df: pd.DataFrame) -> Tuple[bool, bool, str]:
        """Generate buy/sell signals based on technical indicators."""
        df['rsi'] = self.indicators.calculate_rsi(df['close'])
        df['macd'], df['signal_line'] = self.indicators.calculate_macd(df['close'])
        df['sma_fast'] = self.indicators.calculate_sma(df['close'], 20)
        df['sma_slow'] = self.indicators.calculate_sma(df['close'], 50)
        df['bb_mid'], df['bb_upper'], df['bb_lower'] = self.indicators.calculate_bollinger_bands(df['close'])
        df['volume_sma'] = self.indicators.calculate_sma(df['volume'], 20)

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # Buy conditions
        buy_signal = (
            latest['rsi'] < 30 and  # Oversold
            latest['macd'] > latest['signal_line'] and prev['macd'] <= prev['signal_line'] and  # MACD bullish crossover
            latest['sma_fast'] > latest['sma_slow'] and prev['sma_fast'] <= prev['sma_slow'] and  # MA crossover
            latest['close'] < latest['bb_lower'] and  # Below lower Bollinger Band
            latest['volume'] > latest['volume_sma']  # Volume confirmation
        )

        # Sell conditions
        sell_signal = (
            latest['rsi'] > 70 and  # Overbought
            latest['macd'] < latest['signal_line'] and prev['macd'] >= prev['signal_line'] and  # MACD bearish crossover
            latest['sma_fast'] < latest['sma_slow'] and prev['sma_fast'] >= prev['sma_slow'] and  # MA crossover
            latest['close'] > latest['bb_upper']  # Above upper Bollinger Band
        )

        justification = ""
        if buy_signal:
            justification = f"Buy Signal: RSI {latest['rsi']:.2f} (oversold), MACD crossover, MA crossover, below BB lower, high volume"
        elif sell_signal:
            justification = f"Sell Signal: RSI {latest['rsi']:.2f} (overbought), MACD crossover, MA crossover, above BB upper"

        return buy_signal, sell_signal, justification

    async def run(self):
        """Main bot loop."""
        while True:
            try:
                logger.info("Starting market scan...")
                
                symbols = self.api.get_available_symbols()
                if not symbols:
                    await self.send_telegram_message("Error: No symbols available")
                    time.sleep(3600)
                    continue

                # Calculate volatility for each symbol
                signals = []
                for symbol in symbols:
                    df = self.api.get_market_data(symbol)
                    if df is None or len(df) < 60:
                        continue

                    volatility = self.calculate_volatility(df)
                    buy_signal, sell_signal, justification = self.generate_signals(df)
                    if buy_signal or sell_signal:
                        signals.append({
                            'symbol': symbol,
                            'volatility': volatility,
                            'buy': buy_signal,
                            'sell': sell_signal,
                            'justification': justification,
                            'price': df['close'].iloc[-1]
                        })

                if not signals:
                    logger.info("No trading signals found")
                    
                    time.sleep(3600)
                    continue

                # Prioritize by volatility
                signals.sort(key=lambda x: x['volatility'], reverse=True)
                top_signal = signals[0]
                symbol = top_signal['symbol']
                price = top_signal['price']

                # Send signal alert
                await self.send_telegram_message(
                    f"Signal Generated for {symbol}\n"
                    f"Price: ₹{price:.2f}\n"
                    f"Action: {'Buy' if top_signal['buy'] else 'Sell'}\n"
                    f"Justification: {top_signal['justification']}"
                )

                # Execute trade
                balance = self.api.get_wallet_balance()
                if balance <= 0:
                    await self.send_telegram_message("Error: Insufficient INR balance")
                    time.sleep(3600)
                    continue

                amount = (balance * WALLET_UTILIZATION) / price
                side = 'buy' if top_signal['buy'] else 'sell'
                if self.api.place_market_order(symbol, side, amount):
                    await self.send_telegram_message(
                        f"Trade Executed for {symbol}\n"
                        f"Side: {side.upper()}\n"
                        f"Amount: {amount:.6f}\n"
                        f"Price: ₹{price:.2f}"
                    )

                # Wait before next scan (avoid hitting API limits)
                time.sleep(3600)  # 1 hour

            except Exception as e:
                logger.error(f"Bot error: {str(e)}")
                await self.send_telegram_message(f"Bot Error: {str(e)}")
                time.sleep(3600)


async def main():
    port = int(os.getenv("PORT", 10000))
    # Start a simple HTTP server for health checks
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    print(f"Listening on port {port} for Render health check")

    # Run the server in a separate thread or async task if needed
    import threading
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    api = CoinSwitchAPI(COINSWITCH_API_KEY, COINSWITCH_API_SECRET)
    bot = TradingBot(api, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    
    if not api.validate_keys():
        logger.error("Error: Invalid API keys")
        await bot.send_telegram_message("Error: Invalid API keys")
        return
    # data = api.get_market_data("BTC/INR")
    # print(data)
    await bot.run()

    #data = api.get_market_data("ETH/INR")
    #print(data)

if __name__ == "__main__":
    asyncio.run(main())