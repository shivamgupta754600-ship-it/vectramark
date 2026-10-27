from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MarketSymbol = Literal[
    "XAUUSD",
    "XAGUSD",
    "NASDAQ100",
    "SP500",
    "DJIA",
]


class MarketRequest(BaseModel):
    """Validated request for market information."""

    model_config = ConfigDict(extra="forbid")

    symbol: MarketSymbol = Field(
        ...,
        description="Internal VectraMark market identifier.",
        examples=["XAUUSD"],
    )


class MarketResponse(BaseModel):
    """Structured response containing market metadata."""

    symbol: MarketSymbol
    name: str
    category: str
    description: str
