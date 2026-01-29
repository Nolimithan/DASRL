#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
yfinance proxy setup utility

This script helps configure yfinance proxy settings to resolve network access issues.

Usage:
    python setup_proxy.py                               # Enable default proxy
    python setup_proxy.py --proxy http://127.0.0.1:8080  # Specify proxy address
    python setup_proxy.py --disable                     # Disable proxy
    python setup_proxy.py --test                        # Test proxy connectivity
"""

import os
import sys
import argparse
import requests
import yfinance as yf
from datetime import datetime

def setup_proxy(proxy_url: str = 'http://127.0.0.1:7890', enable: bool = True):
    """
    Configure proxy settings.

    Args:
        proxy_url: Proxy server address.
        enable: Whether to enable the proxy.
    """
    if enable:
        print(f"Enabling proxy: {proxy_url}")
        
        # Set environment variables
        os.environ['HTTP_PROXY'] = proxy_url
        os.environ['HTTPS_PROXY'] = proxy_url
        os.environ['ENABLE_YFINANCE_PROXY'] = 'true'
        os.environ['YFINANCE_PROXY'] = proxy_url
        
        # Set proxy for requests session
        try:
            session = requests.Session()
            session.proxies = {
                'http': proxy_url,
                'https': proxy_url
            }
            # Set global session for yfinance
            yf.download._session = session
            print("✅ Proxy configured successfully")
            
        except Exception as e:
            print(f"❌ Error while setting proxy: {e}")
            return False
            
    else:
        print("Disabling proxy...")
        
        # Clear environment variables
        for var in ['HTTP_PROXY', 'HTTPS_PROXY', 'ENABLE_YFINANCE_PROXY', 'YFINANCE_PROXY']:
            if var in os.environ:
                del os.environ[var]
        
        print("✅ Proxy disabled")
    
    return True

def test_proxy_connection(proxy_url: str = None):
    """
    Test proxy connectivity.

    Args:
        proxy_url: Proxy server address; if None, test current settings.
    """
    print("Testing proxy connectivity...")
    
    if proxy_url:
        setup_proxy(proxy_url, True)
    
    # Test basic network connectivity
    try:
        print("1. Testing basic network connectivity...")
        response = requests.get('https://httpbin.org/ip', timeout=10)
        if response.status_code == 200:
            print(f"   ✅ Network OK, IP: {response.json().get('origin', 'unknown')}")
        else:
            print(f"   ❌ Network error, status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Network request failed: {e}")
        return False
    
    # Test Yahoo Finance connectivity
    try:
        print("2. Testing Yahoo Finance connectivity...")
        response = requests.get('https://query1.finance.yahoo.com/v7/finance/options/AAPL', timeout=10)
        if response.status_code == 200:
            print("   ✅ Yahoo Finance connection OK")
        else:
            print(f"   ⚠️ Yahoo Finance error, status code: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Yahoo Finance request failed: {e}")
    
    # Test yfinance data fetch
    try:
        print("3. Testing yfinance data fetch...")
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="2d")
        
        if not data.empty:
            print(f"   ✅ yfinance data fetched, {len(data)} records")
            print(f"   Latest close: ${data['Close'].iloc[-1]:.2f}")
        else:
            print("   ❌ yfinance data fetch failed: empty data")
            return False
    except Exception as e:
        print(f"   ❌ yfinance data fetch failed: {e}")
        return False
    
    print("✅ All tests passed!")
    return True

def show_current_settings():
    """Show current proxy settings."""
    print("Current proxy settings:")
    print("="*50)
    
    # Check environment variables
    http_proxy = os.environ.get('HTTP_PROXY', 'Not set')
    https_proxy = os.environ.get('HTTPS_PROXY', 'Not set')
    enable_proxy = os.environ.get('ENABLE_YFINANCE_PROXY', 'false')
    yfinance_proxy = os.environ.get('YFINANCE_PROXY', 'Not set')
    
    print(f"HTTP_PROXY: {http_proxy}")
    print(f"HTTPS_PROXY: {https_proxy}")
    print(f"ENABLE_YFINANCE_PROXY: {enable_proxy}")
    print(f"YFINANCE_PROXY: {yfinance_proxy}")
    print("="*50)
    
    # Check yfinance session settings
    try:
        if hasattr(yf.download, '_session') and yf.download._session:
            session_proxies = getattr(yf.download._session, 'proxies', {})
            if session_proxies:
                print(f"yfinance session proxy: {session_proxies}")
            else:
                print("yfinance session proxy: Not set")
        else:
            print("yfinance session proxy: Not set")
    except:
        print("yfinance session proxy: check failed")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='yfinance proxy setup utility',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python setup_proxy.py                                    # Enable default proxy
  python setup_proxy.py --proxy http://127.0.0.1:8080    # Specify proxy address
  python setup_proxy.py --disable                         # Disable proxy
  python setup_proxy.py --test                            # Test current proxy settings
  python setup_proxy.py --test --proxy http://127.0.0.1:7890  # Test specified proxy
  python setup_proxy.py --show                            # Show current settings

Notes:
1. Ensure the proxy server is running and reachable
2. Proxy settings only apply to the current Python session
3. For permanent proxy settings, use environment variables
4. If SSL errors occur, you may need proxy SSL configuration
        """
    )
    
    parser.add_argument('--proxy', type=str, default='http://127.0.0.1:7890',
                        help='Proxy server address (default: http://127.0.0.1:7890)')
    parser.add_argument('--disable', action='store_true',
                        help='Disable proxy')
    parser.add_argument('--test', action='store_true',
                        help='Test proxy connectivity')
    parser.add_argument('--show', action='store_true',
                        help='Show current proxy settings')
    
    args = parser.parse_args()
    
    print("yfinance proxy setup utility")
    print("="*50)
    
    if args.show:
        show_current_settings()
        return
    
    if args.test:
        proxy_to_test = args.proxy if not args.disable else None
        success = test_proxy_connection(proxy_to_test)
        if not success:
            print("Proxy test failed. Please check proxy settings.")
            sys.exit(1)
        return
    
    if args.disable:
        setup_proxy(enable=False)
    else:
        setup_proxy(args.proxy, True)
        
        # Verify proxy settings
        print("\nVerifying proxy settings...")
        if test_proxy_connection():
            print("\n✅ Proxy configured! You can run experiments with:")
            print(f"python hedge_strategy_experiment.py --enable_proxy --proxy {args.proxy}")
        else:
            print("\n❌ Proxy verification failed. Please check settings.")
            sys.exit(1)

if __name__ == "__main__":
    main() 

