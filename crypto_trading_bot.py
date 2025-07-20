import os
import time
import logging
import pandas as pd
import numpy as np
import requests
import hmac
import hashlib
import base64
from datetime import datetime, timedelta
from dotenv import load_dotenv
from telegram import Bot
from typing import Dict, List, Tuple
import asyncio

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
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')  # Add your chat ID to .env
BASE_URL = "https://api.coinswitch.co/v2"
WALLET_UTILIZATION = 0.9  # 90% of wallet balance
TIMEFRAME = '1d'  # Daily candles

class CoinSwitchAPI:
    """Handles interactions with CoinSwitch API."""
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret

    def _generate_signature(self, method: str, endpoint: str, params: str = '') -> str:
        """Generate HMAC SHA256 signature for API authentication."""
        message = f"{method}{endpoint}{params}"
        signature = hmac.new(
            bytes(self.api_secret, 'utf-8'),
            msg=bytes(message, 'utf-8'),
            digestmod=hashlib.sha256
        ).digest()
        return base64.b64encode(signature).decode('utf-8')

    def get_market_data(self, symbol: str) -> pd.DataFrame:
        """Fetch historical OHLCV data for a given symbol."""
        try:
            endpoint = f"/market/ohlc?symbol={symbol}&interval={TIMEFRAME}&limit=100"
            signature = self._generate_signature('GET', endpoint)
            headers = {
                'X-AUTH-APIKEY': self.api_key,
                'X-AUTH-SIGNATURE': signature
            }
            response = requests.get(BASE_URL + endpoint, headers=headers)
            response.raise_for_status()
            data = response.json()['data']
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {str(e)}")
            return None

    def get_wallet_balance(self) -> float:
        """Fetch INR wallet balance."""
        try:
            endpoint = "/wallet/balance"
            signature = self._generate_signature('GET', endpoint)
            headers = {
                'X-AUTH-APIKEY': self.api_key,
                'X-AUTH-SIGNATURE': signature
            }
            response = requests.get(BASE_URL + endpoint, headers=headers)
            response.raise_for_status()
            balances = response.json()['data']['balances']
            for balance in balances:
                if balance['currency'] == 'INR':
                    return float(balance['available'])
            return 0.0
        except Exception as e:
            logger.error(f"Error fetching wallet balance: {str(e)}")
            return 0.0

    def place_market_order(self, symbol: str, side: str, amount: float) -> bool:
        """Place a market order."""
        try:
            endpoint = "/trade/order"
            params = f"symbol={symbol}&side={side}&type=market&quantity={amount}"
            signature = self._generate_signature('POST', endpoint, params)
            headers = {
                'X-AUTH-APIKEY': self.api_key,
                'X-AUTH-SIGNATURE': signature,
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            response = requests.post(BASE_URL + endpoint, headers=headers, data=params)
            response.raise_for_status()
            logger.info(f"Placed {side} order for {symbol}: {amount}")
            return True
        except Exception as e:
            logger.error(f"Error placing {side} order for {symbol}: {str(e)}")
            return False

    def get_available_symbols(self) -> List[str]:
        """Fetch available trading pairs."""
        try:
            endpoint = "/market/pairs"
            signature = self._generate_signature('GET', endpoint)
            headers = {
                'X-AUTH-APIKEY': self.api_key,
                'X-AUTH-SIGNATURE': signature
            }
            response = requests.get(BASE_URL + endpoint, headers=headers)
            response.raise_for_status()
            pairs = response.json()['data']
            return [pair['symbol'] for pair in pairs if pair['quote_currency'] == 'INR']
        except Exception as e:
            logger.error(f"Error fetching available symbols: {str(e)}")
            return []

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
    """Initialize and run the trading bot."""
    api = CoinSwitchAPI(COINSWITCH_API_KEY, COINSWITCH_API_SECRET)
    bot = TradingBot(api, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    await bot.run()

if __name__ in ["__main__", "pyodide"]:
    asyncio.run(main())