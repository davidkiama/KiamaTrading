#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Auto-detect strategy name from filename for trade logging
"""

import sys
from pathlib import Path
import os

# ========================================
# GET STRATEGY NAME FROM FILENAME
# ========================================


def get_strategy_name():
    """
    Automatically detect strategy name from the executing script's filename.
    Converts filename to readable format.

    Examples:
        ichimoku_fvg_multiTF.py  → Ichimoku_Fvg_MultiTF
        ema_crossover_strategy.py → Ema_Crossover_Strategy
        strategy.py              → Strategy

    Returns
    -------
    str
        Formatted strategy name
    """
    # Get the filename of the script that called this
    filename = os.path.basename(sys.argv[0])

    # Remove .py extension
    strategy_name = filename.replace('.py', '')

    # Convert snake_case to Title_Case
    # This splits on underscores and capitalizes each word
    formatted_name = '_'.join(word.capitalize()
                              for word in strategy_name.split('_'))

    return formatted_name


# ========================================
# ALTERNATIVE: Get folder name as strategy
# ========================================

def get_strategy_name_from_folder():
    """
    Get strategy name from the folder the script is in.
    Useful if your folder name is more descriptive than the script name.

    Examples:
        ichimoku_fvg_multiTF/ichimoku_fvg_strategy.py → Ichimoku_Fvg_MultiTF
        ema_crossover/strategy.py                     → Ema_Crossover

    Returns
    -------
    str
        Formatted folder name
    """
    # Get parent folder name
    folder_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

    # Convert snake_case to Title_Case
    formatted_name = '_'.join(word.capitalize()
                              for word in folder_name.split('_'))

    return formatted_name


# ========================================
# GLOBAL STRATEGY NAME (set once at startup)
# ========================================

# Choose one of these:
STRATEGY_NAME = get_strategy_name()  # Uses filename
# STRATEGY_NAME = get_strategy_name_from_folder()  # Uses folder name


# ========================================
# UPDATED LOG FUNCTION
# ========================================

def log_executed_trade(instrument, signal, entry, sl, tp, timeframe, rr_ratio=None):
    """
    Log a trade with automatic strategy name detection.

    No need to pass strategy_name - it's auto-detected!

    Parameters
    ----------
    instrument : str
        Trading instrument (e.g., "AUD_USD")
    signal : str
        "BUY" or "SELL"
    entry : float
        Entry price
    sl : float
        Stop loss price
    tp : float
        Take profit price
    timeframe : str
        Timeframe executed (e.g., "15M", "1H")
    rr_ratio : float, optional
        Risk to Reward ratio
    """
    from trade_logger import logger

    logger.log_trade(
        strategy_name=STRATEGY_NAME,
        instrument=instrument,
        signal=signal,
        entry=entry,
        sl=sl,
        tp=tp,
        timeframe=timeframe,
        rr_ratio=rr_ratio
    )


# ========================================
# USAGE IN YOUR STRATEGY
# ========================================

"""
INTEGRATION EXAMPLE:

In your ichimoku_fvg_strategy.py:

    import sys
    from pathlib import Path
    root = Path(__file__).parent.parent
    sys.path.insert(0, str(root))
    
    from auto_strategy_name import log_executed_trade, STRATEGY_NAME
    from trade_logger import logger  # if you need other logger functions
    
    # In your trading_job():
    def trading_job():
        # ... your trading logic ...
        
        if signal == 'BUY':
            # ... place order ...
            
            # Now just call it WITHOUT strategy name:
            log_executed_trade(
                instrument=INSTRUMENT,
                signal="BUY",
                entry=entry,
                sl=sl,
                tp=tp,
                timeframe="15M"
            )
            
            # The strategy name is automatically: "Ichimoku_Fvg_Strategy"
            # (from your filename ichimoku_fvg_strategy.py)
"""


# ========================================
# TEST - Run to see auto-detection
# ========================================

if __name__ == "__main__":
    print("\n🧪 Auto Strategy Name Detection Test\n")
    print(f"Current filename: {os.path.basename(sys.argv[0])}")
    print(f"Detected strategy name: {STRATEGY_NAME}")
    print(f"Folder name method: {get_strategy_name_from_folder()}")
    print("\n✓ Test complete!\n")
