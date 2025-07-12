#!/usr/bin/env python3
"""
Sector Data and Stock Universe
"""

# Sector-based stock universe
SECTOR_STOCKS = {
    "Technology": [
        "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "TSLA", "NVDA", "NFLX", "ADBE",
        "CRM", "ORCL", "INTC", "AMD", "QCOM", "AVGO", "TXN", "AMAT", "LRCX", "KLAC",
        "MRVL", "MU", "SNPS", "CDNS", "FTNT", "PANW", "CRWD", "ZS", "OKTA", "DDOG"
    ],
    "Healthcare": [
        "JNJ", "UNH", "PFE", "ABBV", "TMO", "ABT", "DHR", "BMY", "LLY", "MRK",
        "AMGN", "GILD", "MDLZ", "CVS", "CI", "HUM", "ANTM", "BIIB", "REGN", "VRTX",
        "ISRG", "SYK", "BSX", "EW", "HOLX", "VAR", "DXCM", "ALGN", "ILMN", "MRNA"
    ],
    "Financial Services": [
        "BRK-B", "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SPGI",
        "CB", "MMC", "PGR", "TRV", "ALL", "AIG", "MET", "PRU", "AFL", "HIG",
        "V", "MA", "PYPL", "SQ", "FISV", "FIS", "ADP", "PAYX", "TFC", "USB"
    ],
    "Consumer Discretionary": [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "BKNG", "CMG",
        "ORLY", "AZO", "ROST", "YUM", "DG", "DLTR", "BBY", "TGT", "KSS", "M",
        "F", "GM", "NCLH", "CCL", "RCL", "MAR", "HLT", "MGM", "LVS", "WYNN"
    ],
    "Consumer Staples": [
        "PG", "KO", "PEP", "WMT", "COST", "MDLZ", "CL", "KMB", "GIS", "K",
        "HSY", "SJM", "CPB", "CAG", "HRL", "MKC", "CHD", "CLX", "TSN", "TAP",
        "KR", "SYY", "ADM", "BG", "CF", "MOS", "FMC", "CTVA", "DD", "DOW"
    ],
    "Energy": [
        "XOM", "CVX", "COP", "EOG", "SLB", "PSX", "VLO", "MPC", "HES", "DVN",
        "FANG", "APA", "OXY", "HAL", "BKR", "NOV", "FTI", "HP", "RIG", "VAL",
        "KMI", "OKE", "EPD", "ET", "WMB", "TRGP", "ONEOK", "MPLX", "PAA", "EQT"
    ],
    "Industrials": [
        "BA", "CAT", "HON", "UPS", "RTX", "LMT", "GE", "MMM", "DE", "UNP",
        "FDX", "WM", "EMR", "ETN", "ITW", "PH", "CMI", "ROK", "DOV", "XYL",
        "PCAR", "NOC", "GD", "LHX", "TDG", "CTAS", "FAST", "PAYX", "VRSK", "IEX"
    ],
    "Materials": [
        "LIN", "APD", "ECL", "SHW", "FCX", "NEM", "CTVA", "DD", "DOW", "PPG",
        "NUE", "STLD", "X", "CLF", "AA", "CENX", "MP", "LAC", "ALB", "SQM",
        "FMC", "CF", "MOS", "ADM", "BG", "IFF", "CE", "VMC", "MLM", "FAST"
    ],
    "Real Estate": [
        "AMT", "PLD", "CCI", "EQIX", "PSA", "EXR", "AVB", "EQR", "WELL", "DLR",
        "SPG", "O", "VICI", "IRM", "WY", "PCG", "ARE", "VTR", "ESS", "MAA",
        "UDR", "CPT", "FRT", "REG", "KIM", "BXP", "HST", "RHP", "PEI", "MAC"
    ],
    "Utilities": [
        "NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL", "SRE", "PEG", "ED",
        "FE", "ETR", "ES", "AWK", "PPL", "CMS", "DTE", "WEC", "LNT", "EVRG",
        "ATO", "CNP", "NI", "OGE", "PNW", "IDA", "MDU", "BKH", "AVA", "POR"
    ],
    "Communication Services": [
        "GOOGL", "GOOG", "META", "NFLX", "DIS", "CMCSA", "VZ", "T", "TMUS", "CHTR",
        "ATVI", "EA", "TTWO", "NTES", "BILI", "SPOT", "ROKU", "PINS", "SNAP", "TWTR",
        "DISH", "SIRI", "LBRDA", "LBRDK", "FWONA", "FWONK", "BATRK", "BATRA", "LSXMA", "LSXMK"
    ]
}

# Sector ETFs for benchmarking
SECTOR_ETFS = {
    "Technology": "XLK",
    "Healthcare": "XLV", 
    "Financial Services": "XLF",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Communication Services": "XLC"
}

def get_sector_stocks(sectors):
    """Get all stocks for selected sectors"""
    stocks = []
    for sector in sectors:
        if sector in SECTOR_STOCKS:
            stocks.extend(SECTOR_STOCKS[sector])
    return list(set(stocks))  # Remove duplicates

def get_sector_etf(sector):
    """Get ETF for a specific sector"""
    return SECTOR_ETFS.get(sector, "SPY")

def get_all_sectors():
    """Get list of all available sectors"""
    return list(SECTOR_STOCKS.keys())

def get_stocks_by_sector(sector):
    """Get stocks for a specific sector"""
    return SECTOR_STOCKS.get(sector, [])

