"""Exceptions shared by all market-data sources (ccxt exchanges and the Alpaca adapter)."""


class DataSourceError(Exception):
    """A data source rejected a request or returned something unusable. Not retried."""


class TransientDataError(DataSourceError):
    """Temporary failure (network, rate limit, 5xx). Safe to retry with backoff."""
