import pandas as pd
from typing import List, Dict, Tuple, Any
import json
import numpy as np

# Use classes from the trader's perspective
from datamodel import (
    TradingState,
    OrderDepth,
    Listing,
    Trade,
    Observation,
    ConversionObservation,
    Order,
    ProsperityEncoder,
    Symbol,
    Product,
    Position
)
# The Trader class from the user's file
from template2 import Trader

# Load the trader
trader = Trader()

# Read the CSV data
try:
    df = pd.read_csv("data/round1/prices_round_1_day_-2.csv", delimiter=";")
except FileNotFoundError:
    print("Error: prices_round_1_day_0.csv not found in data/round1/. Make sure the path is correct.")
    exit()

# Group data by timestamp
grouped_by_timestamp = df.groupby('timestamp')

# Initialize simulation state variables
current_position: Dict[Product, Position] = {}
# Initialize traderData (if the trader uses it persistantly)
traderData = ""
# Store results for PNL calculation
all_orders_placed: List[Dict[Symbol, List[Order]]] = []
last_state: TradingState = None

print(f"Starting backtest over {len(grouped_by_timestamp)} timestamps...")

# Iterate through each timestamp
for timestamp, group in grouped_by_timestamp:
    # Build the TradingState for this timestamp
    listings: Dict[Symbol, Listing] = {}
    order_depths: Dict[Symbol, OrderDepth] = {}
    market_trades: Dict[Symbol, List[Trade]] = {} # Assuming no market trades in this data
    own_trades: Dict[Symbol, List[Trade]] = {}    # Assuming no own trades feedback in this data

    # Populate data for each product present at this timestamp
    for _, row in group.iterrows():
        product = row['product']
        symbol = product # Assuming symbol is the same as product for round 1

        # Create Listing
        listings[symbol] = Listing(
            symbol=symbol,
            product=product,
            denomination="SEASHELLS" # Assuming denomination
        )

        # Create Order Depth
        depth = OrderDepth()
        buy_orders: Dict[int, int] = {}
        sell_orders: Dict[int, int] = {}
        for level in range(1, 4):
            bid_price = row.get(f'bid_price_{level}')
            bid_volume = row.get(f'bid_volume_{level}')
            ask_price = row.get(f'ask_price_{level}')
            ask_volume = row.get(f'ask_volume_{level}')

            if pd.notna(bid_price) and pd.notna(bid_volume) and bid_volume > 0:
                buy_orders[int(bid_price)] = int(bid_volume)
            if pd.notna(ask_price) and pd.notna(ask_volume) and ask_volume > 0:
                # Sell orders quantity should be negative in datamodel.OrderDepth
                sell_orders[int(ask_price)] = -int(ask_volume)

        depth.buy_orders = buy_orders
        depth.sell_orders = sell_orders
        order_depths[symbol] = depth

        # Initialize position if not seen before
        if product not in current_position:
            current_position[product] = 0

        # Initialize empty trade lists for this product
        market_trades[symbol] = []
        own_trades[symbol] = []


    # Create dummy Observations (required by TradingState, but data not present)
    plain_value_observations = {}
    conversion_observations = {} 

    observations = Observation(
        plainValueObservations=plain_value_observations,
        conversionObservations=conversion_observations
    )


    # Create the full TradingState
    state = TradingState(
        traderData=traderData,
        timestamp=int(timestamp),
        listings=listings,
        order_depths=order_depths,
        own_trades=own_trades,
        market_trades=market_trades,
        position=current_position.copy(), # Pass a copy
        observations=observations
    )

    # Run the trader logic
    orders_dict, conversions, next_traderData = trader.run(state)

    # Store the results
    all_orders_placed.append(orders_dict)
    traderData = next_traderData # Persist trader data for the next iteration
    last_state = state # Keep track of the last state for final PNL calculation


    for symbol, orders in orders_dict.items():
        product = listings[symbol].product
        for order in orders:
            # order.quantity > 0 : BUY, increases position
            # order.quantity < 0 : SELL, decreases position
            current_position[product] = current_position.get(product, 0) + order.quantity

# --- PNL Calculation ---
def calculate_pnl(all_orders: List[Dict[Symbol, List[Order]]], final_state: TradingState) -> float:
    cash = 0.0
    final_inventory: Dict[Product, Position] = {}

    if final_state is None:
        print("Warning: No final state available for PNL calculation.")
        return 0.0

    # Calculate cash flow from orders and track final inventory
    for orders_at_timestamp in all_orders:
        for symbol, order_list in orders_at_timestamp.items():
            product = final_state.listings[symbol].product
            for order in order_list:
                # Buy order (quantity > 0): Cash decreases (- price * quantity)
                # Sell order (quantity < 0): Cash increases (- price * quantity)
                cash -= order.price * order.quantity
                final_inventory[product] = final_inventory.get(product, 0) + order.quantity

    print("\nFinal Inventory:", final_inventory)

    # Value final inventory using the mid-price from the *last* state
    print("Valuing final inventory using last state's mid-prices:")
    for product, final_pos in final_inventory.items():
        if final_pos == 0:
            continue

        symbol = product # Assuming symbol == product
        if symbol in final_state.order_depths:
            order_depth = final_state.order_depths[symbol]
            
            best_bid = -np.inf
            if order_depth.buy_orders:
                best_bid = max(order_depth.buy_orders.keys())

            best_ask = np.inf
            if order_depth.sell_orders:
                 best_ask = min(order_depth.sell_orders.keys())

            # Handle cases where book is empty or one-sided
            if best_bid == -np.inf and best_ask == np.inf:
                 mid_price = 0 # Or use last known trade price if available
                 print(f"  - {symbol}: No Bids/Asks found. Using mid_price = 0")
            elif best_bid == -np.inf:
                 mid_price = best_ask # Use best ask if no bids
                 print(f"  - {symbol}: No Bids found. Using mid_price = {mid_price} (Best Ask)")
            elif best_ask == np.inf:
                 mid_price = best_bid # Use best bid if no asks
                 print(f"  - {symbol}: No Asks found. Using mid_price = {mid_price} (Best Bid)")
            else:
                 mid_price = (best_bid + best_ask) / 2.0
                 print(f"  - {symbol}: Best Bid={best_bid}, Best Ask={best_ask}. Using mid_price = {mid_price:.2f}")

            inventory_value = final_pos * mid_price
            cash += inventory_value
            print(f"  - {symbol}: Position={final_pos}, Value Added={inventory_value:.2f}")
        else:
            print(f"  - Warning: Product {product} not found in final state's order depths. Cannot value position.")


    return cash

# Calculate and print the final PNL
final_pnl = calculate_pnl(all_orders_placed, last_state)
print(f"\nBacktest finished.")
print(f"Final PNL: {final_pnl:.2f}")