# -*- coding: utf-8 -*-
"""
🤖 AI Trading Bot – Steg 1 Förbättrad Basversion
Alpaca + RSI/Volym + Bollinger + Telegram + Säker orderhantering
"""
<<<<<<< HEAD
from dotenv import load_dotenv
=======

>>>>>>> b65d31f (Första commit med hela projektet)
import os
import time
import warnings
import logging
from datetime import datetime
from typing import List, Dict, Optional

import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
import requests
import alpaca_trade_api as tradeapi

<<<<<<< HEAD
# Ladda miljövariabler
load_dotenv()

=======
>>>>>>> b65d31f (Första commit med hela projektet)
# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('trading_bot.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# -------------------- Konfiguration --------------------
class TradingBotConfig:
    def __init__(self):
<<<<<<< HEAD
        # 🔒 OBLIGATORISKA API-nycklar (måste finnas i .env)
        self.ALPACA_API_KEY = self.get_env_var('ALPACA_API_KEY')
        self.ALPACA_API_SECRET = self.get_env_var('ALPACA_API_SECRET')
        
        # 📡 VALFRIA konfigurationer (har standardvärden)
        self.ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

        # 📱 Telegram (VALFRITT - fungerar utan)
        self.TELEGRAM_BOT_TOKEN: Optional[str] = os.getenv('TELEGRAM_BOT_TOKEN')
        self.TELEGRAM_CHAT_ID: Optional[str] = os.getenv('TELEGRAM_CHAT_ID')
=======
        # 🔒 Säkra API-nycklar
        self.ALPACA_API_KEY = self.get_env_var('ALPACA_API_KEY')
        self.ALPACA_API_SECRET = self.get_env_var('ALPACA_API_SECRET')
        self.ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

        # Telegram (valfritt)
        self.TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
        self.TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
>>>>>>> b65d31f (Första commit med hela projektet)

        # Trading parametrar
        self.MIN_VOLUME = int(os.getenv('MIN_VOLUME', 1000))
        self.RSI_OVERSOLD = float(os.getenv('RSI_OVERSOLD', 30))
        self.RSI_OVERBOUGHT = float(os.getenv('RSI_OVERBOUGHT', 70))
        self.MIN_VOLUME_SPIKE = float(os.getenv('MIN_VOLUME_SPIKE', 50))  # Volymökning krävs
        self.API_DELAY = float(os.getenv('API_DELAY', 0.5))
        self.MAX_RETRIES = int(os.getenv('MAX_RETRIES', 3))
        
        # 🛡️ Riskkontroller
        self.POSITION_SIZE_PCT = float(os.getenv('POSITION_SIZE_PCT', 0.02))  # 2% per trade
        self.MAX_POSITION_VALUE = float(os.getenv('MAX_POSITION_VALUE', 1000))  # Max $1000 per position
        self.STOP_LOSS_PCT = float(os.getenv('STOP_LOSS_PCT', 0.04))  # 4% stop loss
        self.TAKE_PROFIT_PCT = float(os.getenv('TAKE_PROFIT_PCT', 0.05))  # 5% take profit
        self.MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', 100))  # Max $100 förlust/dag
        
        # Säkerhet
        self.DRY_RUN = os.getenv('DRY_RUN', 'true').lower() == 'true'  # Simulering som standard
        
        logger.info("✅ Konfiguration laddad")
        if self.DRY_RUN:
            logger.warning("⚠️ DRY RUN MODE - Inga riktiga order skickas!")
<<<<<<< HEAD
        
        # Validera konfiguration
        self._validate_config()
    
    def _validate_config(self):
        """Kontrollera att konfigurationen är giltig"""
        # Kontrollera Telegram-konfiguration
        if self.TELEGRAM_BOT_TOKEN and not self.TELEGRAM_CHAT_ID:
            logger.warning("⚠️ TELEGRAM_BOT_TOKEN finns men TELEGRAM_CHAT_ID saknas")
        elif self.TELEGRAM_CHAT_ID and not self.TELEGRAM_BOT_TOKEN:
            logger.warning("⚠️ TELEGRAM_CHAT_ID finns men TELEGRAM_BOT_TOKEN saknas")
        elif self.TELEGRAM_BOT_TOKEN and self.TELEGRAM_CHAT_ID:
            logger.info("📱 Telegram-konfiguration hittad")
        
        # Kontrollera att nycklar har rätt längd (ungefär)
        if len(self.ALPACA_API_KEY.strip()) < 15:
            logger.warning("⚠️ ALPACA_API_KEY verkar för kort")
        if len(self.ALPACA_API_SECRET.strip()) < 30:
            logger.warning("⚠️ ALPACA_API_SECRET verkar för kort")

    def get_env_var(self, key: str) -> str:
        """Förbättrad miljövariabel-hantering - OBLIGATORISKA värden"""
        value = os.getenv(key)
        if not value or value.strip() == '':
            logger.error(f"❌ OBLIGATORISK miljövariabel '{key}' saknas eller tom!")
            logger.error(f"💡 Lägg till '{key}=ditt_värde' i .env filen")
            raise ValueError(f"Miljövariabeln '{key}' saknas!")
        return value.strip()  # Ta bort eventuella mellanslag
=======

    def get_env_var(self, key: str) -> str:
        """Förbättrad miljövariabel-hantering"""
        value = os.getenv(key)
        if not value:
            logger.error(f"❌ Miljövariabeln '{key}' saknas!")
            raise ValueError(f"Miljövariabeln '{key}' saknas!")
        return value
>>>>>>> b65d31f (Första commit med hela projektet)

    def validate_telegram(self) -> bool:
        return bool(self.TELEGRAM_BOT_TOKEN and self.TELEGRAM_CHAT_ID)


# -------------------- Trading Bot --------------------
class TradingBot:
    def __init__(self, config: TradingBotConfig):
        self.config = config
        self.api = None
        self.results = []
        self.daily_pnl = 0.0  # Track daily P&L
        
        self._initialize_alpaca_api()

    def _initialize_alpaca_api(self):
        try:
            self.api = tradeapi.REST(
                self.config.ALPACA_API_KEY,
                self.config.ALPACA_API_SECRET,
                base_url=self.config.ALPACA_BASE_URL
            )
            account = self.api.get_account()
            logger.info(f"✅ Alpaca API ansluten - Konto: {account.status}")
            logger.info(f"💰 Tillgängligt kapital: ${float(account.cash):,.2f}")
<<<<<<< HEAD
        except tradeapi.rest.APIError as e:
            if "unauthorized" in str(e).lower():
                logger.error("❌ ALPACA UNAUTHORIZED - Kontrollera dina API-nycklar i .env filen!")
                logger.error("💡 Se till att ALPACA_API_KEY och ALPACA_API_SECRET är korrekta")
                raise ConnectionError("Felaktiga Alpaca API-nycklar")
            else:
                logger.error(f"❌ Alpaca API-fel: {e}")
                raise ConnectionError(f"Kunde inte ansluta till Alpaca API: {e}")
        except Exception as e:
            logger.error(f"❌ Oväntat Alpaca-fel: {e}")
=======
        except Exception as e:
            logger.error(f"❌ Alpaca API-fel: {e}")
>>>>>>> b65d31f (Första commit med hela projektet)
            raise ConnectionError(f"Kunde inte ansluta till Alpaca API: {e}")

    # ---------- Tickers ----------
    def get_tickers(self, limit: int = 50) -> List[str]:
        try:
            assets = self.api.list_assets(status="active")
            # Filtrera för stora, likvida aktier
            filtered = [
                a.symbol for a in assets 
                if a.tradable and 
                   a.exchange in ['NASDAQ', 'NYSE'] and
                   a.status == 'active' and
                   not a.symbol.startswith('$')  # Undvik specialsymboler
            ]
            logger.info(f"📊 Hämtade {len(filtered[:limit])} aktier att analysera")
            return filtered[:limit]
        except Exception as e:
            logger.warning(f"⚠️ Fallback på populära aktier: {e}")
            return ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'META', 'NVDA', 'AMD']

    # ---------- Telegram ----------
    def send_telegram(self, message: str) -> bool:
        if not self.config.validate_telegram():
            return False
        
        url = f"https://api.telegram.org/bot{self.config.TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": self.config.TELEGRAM_CHAT_ID, 
            "text": message, 
            "parse_mode": "HTML"
        }
        
        for attempt in range(self.config.MAX_RETRIES):
            try:
                r = requests.post(url, json=payload, timeout=10)
                r.raise_for_status()
                logger.info("📩 Telegram skickat")
                return True
            except requests.RequestException as e:
                logger.warning(f"⚠️ Telegram-fel försök {attempt+1}: {e}")
                if attempt < self.config.MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
        
        logger.error("❌ Telegram misslyckades")
        return False

    # ---------- Marknadsdata ----------
    def fetch_data(self, ticker: str) -> Optional[pd.DataFrame]:
        try:
            df = yf.download(ticker, period="3mo", interval="1d", progress=False, auto_adjust=True)
            
            # Grundläggande validering
            if df.empty:
                return None
            
            # Hantera MultiIndex kolumner
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            
            # Kontrollera att nödvändiga kolumner finns
            required_columns = ['Close', 'Volume', 'High', 'Low', 'Open']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.debug(f"❌ {ticker}: Saknar kolumner {missing_columns}")
                return None
            
            # Datakvalitet kontroller - separata if-statements för klarhet
            if df['Close'].isna().all():
                logger.debug(f"❌ {ticker}: Alla stängningspriser är NaN")
                return None
                
            if (df['Volume'] < 0).any():
                logger.debug(f"❌ {ticker}: Negativ volym hittad")
                return None
                
            if len(df) < 20:  # Behöver minst 20 dagars data för RSI
                logger.debug(f"❌ {ticker}: Otillräcklig data ({len(df)} dagar)")
                return None
            
            # Kontrollera för extrema värden som kan vara felaktiga
            if (df['Close'] <= 0).any():
                logger.debug(f"❌ {ticker}: Negativa eller noll-priser")
                return None
                
            if (df['Close'] > 100000).any():  # Priser över $100k är suspekta
                logger.debug(f"❌ {ticker}: Extremt höga priser")
                return None
            
            return df
            
        except Exception as e:
            logger.debug(f"❌ Datafel {ticker}: {e}")
            return None

    # ---------- Indikatorer ----------
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            # RSI
            rsi = RSIIndicator(close=df['Close'], window=14)
            df['RSI'] = rsi.rsi()
            
            # Volymgenomsnitt
            df['vol_avg_10'] = df['Volume'].rolling(10).mean()
            df['vol_avg_20'] = df['Volume'].rolling(20).mean()
            
            # Bollinger Bands
            df['price_sma_20'] = df['Close'].rolling(20).mean()
            df['price_std_20'] = df['Close'].rolling(20).std()
            df['bb_upper'] = df['price_sma_20'] + 2*df['price_std_20']
            df['bb_lower'] = df['price_sma_20'] - 2*df['price_std_20']
            
            # Prisförändringar
            df['price_change_1d'] = df['Close'].pct_change()
            
            return df.dropna()
            
        except Exception as e:
            logger.error(f"❌ Indikatorfel: {e}")
            return pd.DataFrame()

    # ---------- Förbättrad Signal ----------
    def generate_signal(self, latest: pd.Series, ticker: str) -> str:
        try:
            rsi = latest['RSI']
            volume = latest['Volume']
            vol_avg_10 = latest['vol_avg_10']
            vol_avg_20 = latest['vol_avg_20']
            price = latest['Close']
            bb_lower = latest['bb_lower']
            bb_upper = latest['bb_upper']
            
            # Grundläggande filter
            if volume < self.config.MIN_VOLUME:
                return "❌ Låg volym"
            
            # Volymspike krävs för signaler
            vol_spike_10 = ((volume - vol_avg_10) / vol_avg_10) * 100 if vol_avg_10 > 0 else 0
            vol_spike_20 = ((volume - vol_avg_20) / vol_avg_20) * 100 if vol_avg_20 > 0 else 0
            
            if vol_spike_10 < self.config.MIN_VOLUME_SPIKE:
                return "❌ Ingen volymspike"
            
            # STARK KÖPSIGNAL
            if (rsi < self.config.RSI_OVERSOLD and 
                vol_spike_10 > self.config.MIN_VOLUME_SPIKE and
                vol_spike_20 > 25 and  # Även högre än 20-dagars genomsnitt
                price <= bb_lower * 1.02):  # Nära nedre Bollinger Band
                return "✅ STARK KÖP"
            
            # STARK SÄLJSIGNAL  
            elif (rsi > self.config.RSI_OVERBOUGHT and
                  vol_spike_10 > self.config.MIN_VOLUME_SPIKE and
                  price >= bb_upper * 0.98):  # Nära övre Bollinger Band
                return "⚠️ STARK SÄLJ"
            
            return "❌ Ingen signal"
            
        except Exception as e:
            logger.error(f"❌ Signalfel för {ticker}: {e}")
            return "❌ Fel vid signalberäkning"

    # ---------- Säker Orderhantering ----------
    def check_risk_limits(self) -> bool:
        """Kontrollera risklimiter innan order"""
        try:
            account = self.api.get_account()
            
            # Kontrollera daglig förlust
            daily_pnl = float(account.unrealized_pl) + float(account.realized_pl)
            if daily_pnl < -self.config.MAX_DAILY_LOSS:
                logger.warning(f"🚨 Daglig förlustgräns nådd: ${daily_pnl:.2f}")
                return False
            
            # Kontrollera kontostatus
            if account.trading_blocked:
                logger.warning("🚨 Trading blockerat på kontot")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Fel vid riskkontroll: {e}")
            return False

    def calculate_position_size(self, ticker: str, price: float) -> int:
        """Beräkna säker positionsstorlek"""
        try:
            account = self.api.get_account()
            cash = float(account.cash)
            
            # Beräkna storlek baserat på procent av kapital
            size_by_percent = (cash * self.config.POSITION_SIZE_PCT) / price
            
            # Begränsa till max position value
            size_by_max_value = self.config.MAX_POSITION_VALUE / price
            
            # Ta det mindre av de två
            position_size = min(size_by_percent, size_by_max_value)
            
            # Minst 1 aktie
            return max(1, int(position_size))
            
        except Exception as e:
            logger.error(f"❌ Fel vid positionsstorlek-beräkning: {e}")
            return 1

    def place_order(self, ticker: str, signal: str, latest_data: pd.Series):
        """Säker orderhantering med riskkontroller"""
        
        # Riskkontroll
        if not self.check_risk_limits():
            logger.warning(f"🚨 Risklimiter överträdda - ingen order för {ticker}")
            return False
        
        try:
            current_price = float(latest_data['Close'])
            
            if signal == "✅ STARK KÖP":
                qty = self.calculate_position_size(ticker, current_price)
                
                # Dry run kontroll
                if self.config.DRY_RUN:
                    logger.info(f"🎭 [DRY RUN] Skulle köpa {qty} av {ticker} @ ${current_price:.2f}")
                    return True
                
                # Skicka riktig order
                order = self.api.submit_order(
                    symbol=ticker,
                    qty=qty,
                    side='buy',
                    type='market',
                    time_in_force='gtc'
                )
                
                logger.info(f"🟢 Köporder skickad: {ticker} qty {qty} @ ~${current_price:.2f}")
                
                # Skicka Telegram-notis
                message = f"🟢 <b>KÖP ORDER</b>\n{ticker} x {qty}\nPris: ~${current_price:.2f}\nVärde: ~${qty * current_price:.2f}"
                self.send_telegram(message)
                
                return True
                
            elif signal == "⚠️ STARK SÄLJ":
                # Kontrollera om vi äger aktien
                try:
                    position = self.api.get_position(ticker)
                    qty = int(position.qty)
                    
                    if qty <= 0:
                        logger.debug(f"Ingen position i {ticker} att sälja")
                        return False
                    
                    # Dry run kontroll
                    if self.config.DRY_RUN:
                        logger.info(f"🎭 [DRY RUN] Skulle sälja {qty} av {ticker} @ ${current_price:.2f}")
                        return True
                    
                    # Skicka säljorder
                    order = self.api.submit_order(
                        symbol=ticker,
                        qty=qty,
                        side='sell',
                        type='market',
                        time_in_force='gtc'
                    )
                    
                    logger.info(f"🔴 Säljorder skickad: {ticker} qty {qty} @ ~${current_price:.2f}")
                    
                    # Skicka Telegram-notis
                    message = f"🔴 <b>SÄLJ ORDER</b>\n{ticker} x {qty}\nPris: ~${current_price:.2f}\nVärde: ~${qty * current_price:.2f}"
                    self.send_telegram(message)
                    
                    return True
                    
                except Exception:
                    logger.debug(f"Ingen position i {ticker}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Orderfel {ticker}: {e}")
            return False

    # ---------- Huvudanalys ----------
    def analyze_tickers(self):
        """Huvudanalys-loop"""
        logger.info("🚀 Startar trading bot analys...")
        
        tickers = self.get_tickers()
        signals_found = 0
        orders_placed = 0
        
        for i, ticker in enumerate(tickers, 1):
            logger.debug(f"Analyserar {ticker} ({i}/{len(tickers)})")
            
            # Hämta data
            df = self.fetch_data(ticker)
            if df is None:
                continue
            
            # Lägg till indikatorer
            df = self.add_indicators(df)
            if df.empty or len(df) < 20:
                continue
            
            # Generera signal
            latest = df.iloc[-1]
            signal = self.generate_signal(latest, ticker)
            
            # Agera på signaler
            if signal not in ["❌ Ingen signal", "❌ Låg volym", "❌ Ingen volymspike", "❌ Fel vid signalberäkning"]:
                signals_found += 1
                
                logger.info(f"🎯 {signal} för {ticker} | RSI: {latest['RSI']:.1f} | Pris: ${latest['Close']:.2f}")
                
                # Försök placera order
                if self.place_order(ticker, signal, latest):
                    orders_placed += 1
                
                # Skicka grundläggande Telegram-notis
                vol_change = ((latest['Volume'] - latest['vol_avg_10']) / latest['vol_avg_10']) * 100
                telegram_msg = (f"{signal} för <b>{ticker}</b>\n"
                               f"RSI: {latest['RSI']:.1f}\n"
                               f"Pris: ${latest['Close']:.2f}\n"
                               f"Volymökning: +{vol_change:.1f}%")
                self.send_telegram(telegram_msg)
            
            # Rate limiting
            time.sleep(self.config.API_DELAY)
        
        # Sammanfattning
        logger.info("="*60)
        logger.info(f"🎯 Analys slutförd!")
        logger.info(f"📊 Analyserade aktier: {len(tickers)}")
        logger.info(f"🚨 Signaler hittade: {signals_found}")
        logger.info(f"📈 Order placerade: {orders_placed}")
        logger.info("="*60)
        
        # Skicka sammanfattning till Telegram
        summary = (f"📊 <b>ANALYS SLUTFÖRD</b>\n"
                  f"Aktier: {len(tickers)}\n"
                  f"Signaler: {signals_found}\n"
                  f"Order: {orders_placed}")
        self.send_telegram(summary)


<<<<<<< HEAD
# -------------------- Test Functions --------------------
def test_connections():
    """Testa API-anslutningar innan huvudkörning"""
    logger.info("🔧 Testar anslutningar...")
    
    config = TradingBotConfig()
    
    # Test Alpaca anslutning
    try:
        api = tradeapi.REST(
            config.ALPACA_API_KEY, 
            config.ALPACA_API_SECRET, 
            config.ALPACA_BASE_URL,
            api_version='v2'
        )
        account = api.get_account()
        logger.info(f"✅ Alpaca ansluten! Kontostatus: {account.status}")
        logger.info(f"💰 Kontosaldo: ${float(account.cash):,.2f}")
        logger.info(f"📊 Köpkraft: ${float(account.buying_power):,.2f}")
    except Exception as e:
        logger.error(f"❌ Alpaca-anslutning misslyckades: {e}")
        return False
    
    # Test Telegram anslutning (om konfigurerat)
    if config.validate_telegram():
        try:
            test_message = "🤖 Trading Bot: Anslutningstest lyckades!"
            response = requests.post(
                f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage",
                json={"chat_id": config.TELEGRAM_CHAT_ID, "text": test_message},
                timeout=10
            )
            if response.status_code == 200:
                logger.info("✅ Telegram fungerar! Testmeddelande skickat.")
            else:
                logger.warning(f"⚠️ Telegram-problem: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"❌ Telegram-test misslyckades: {e}")
            logger.info("ℹ️ Boten fungerar utan Telegram, men notifikationer skickas inte")
    else:
        logger.info("ℹ️ Telegram inte konfigurerat - notifikationer inaktiverade")
    
    logger.info("✅ Anslutningstester slutförda!")
    return True


=======
>>>>>>> b65d31f (Första commit med hela projektet)
# -------------------- Main --------------------
def main():
    try:
        logger.info("🤖 AI Trading Bot startar...")
        
        config = TradingBotConfig()
        bot = TradingBot(config)
        bot.analyze_tickers()
        
        logger.info("🎉 Trading bot slutförd framgångsrikt!")
        
    except ValueError as e:
        logger.error(f"❌ Konfigurationsfel: {e}")
        print("\n🔧 KONFIGURATIONSHJÄLP:")
        print("Sätt miljövariabler:")
        print("export ALPACA_API_KEY='din_api_nyckel'")
        print("export ALPACA_API_SECRET='din_secret_nyckel'")
        print("export DRY_RUN='false'  # För riktiga order")
        
    except Exception as e:
        logger.error(f"❌ Oväntat fel: {e}")
        raise

<<<<<<< HEAD

if __name__ == "__main__":
    # Kör anslutningstester först
    if not test_connections():
        logger.error("❌ Anslutningstester misslyckades - avslutar")
        exit(1)
    
    # Kör huvudprogrammet om testerna lyckas
=======
if __name__ == "__main__":
>>>>>>> b65d31f (Första commit med hela projektet)
    main()
