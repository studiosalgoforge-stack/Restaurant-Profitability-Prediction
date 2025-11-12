# main_simulator.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import os

# ------- CONFIG -------
st.set_page_config(page_title="Advanced Restaurant Profit Simulator", layout="wide", page_icon="🍽️")
HISTORY_CSV = "menu_history.csv"  # file created/updated in current working dir

# ------- HELPERS -------
def rupee(x):
    try:
        return f"₹{float(x):,.2f}"
    except Exception:
        return x

def detect_unit(item_name: str) -> str:
    if not isinstance(item_name, str) or item_name.strip() == "":
        return "Pieces (pcs)"
    n = item_name.lower()
    if any(k in n for k in ["milk", "juice", "shake", "soup", "latte", "coffee", "tea", "water"]):
        return "Litres (L)"
    if any(k in n for k in ["paneer", "rice", "flour", "dal", "cheese", "butter", "meat", "chicken", "fish"]):
        return "Grams (g)"
    return "Pieces (pcs)"

def strategy_generator(item, qty_sold, margin_pct, ingredients):
    """
    Rule-based 'AI' suggestions. Returns list of paragraphs (strings).
    Also includes numeric suggestions (e.g., reduce ingredient X by Y%).
    """
    paras = []
    # headline
    if margin_pct >= 40:
        paras.append(f"✅ **{item}**: Excellent margin (~{margin_pct:.1f}%). Continue to prioritize quality & scale.")
        paras.append("**Tactics:** Highlight as a flagship item, try premium upsell, use targeted ads or homepage placement.")
    elif margin_pct >= 25:
        paras.append(f"🟢 **{item}**: Good margin (~{margin_pct:.1f}%). Room to increase volume.")
        paras.append("**Tactics:** Bundles/combos, loyalty discounts, and supplier negotiation for small additional margin.")
    elif margin_pct >= 10:
        paras.append(f"🟠 **{item}**: Thin margin (~{margin_pct:.1f}%). Optimize costs carefully.")
        paras.append("**Tactics:** Reduce waste, slightly increase price or portion control, and monitor customer reaction.")
    else:
        paras.append(f"🔴 **{item}**: Low/negative margin (~{margin_pct:.1f}%). Action required.")
        paras.append("**Tactics:** Consider recipe rework, substitute expensive ingredients, or reposition as loss-leader paired with high-margin items.")
    # numeric suggestion: reduce top cost ingredient by 10% and show effect
    if ingredients:
        # sort ingredients by total cost
        sorted_ing = sorted(ingredients, key=lambda d: d["TotalCost"], reverse=True)
        top = sorted_ing[0]
        top_name = top["Ingredient"]
        top_cost = top["TotalCost"]
        total_ing_cost = sum(i["TotalCost"] for i in ingredients)
        if total_ing_cost > 0:
            reduction_pct = 10  # suggest 10% reduction
            new_total = total_ing_cost - top_cost * (reduction_pct / 100)
            # Need selling_price to compute effect — we'll present relative effect per unit:
            paras.append(f"**Numeric idea:** If you can reduce **{top_name}** usage/cost by {reduction_pct}%,"
                         f" ingredient cost per unit reduces from ₹{total_ing_cost:.2f} to ₹{new_total:.2f} (saves ₹{(total_ing_cost-new_total):.2f} per unit).")
    # demand & upsell advice depending on qty_sold
    if qty_sold >= 200:
        paras.append("**Volume tip:** This item sells well — small margin improvements will generate big absolute profit.")
    elif qty_sold >= 50:
        paras.append("**Growth tip:** Moderate sales — promote with staff recommendations and social posts.")
    else:
        paras.append("**Awareness tip:** Low sales — try menu photography, combo placement, or limited-time offers to test demand.")
    return paras

def price_optimizer(ingredient_cost_per_unit, target_margin_pct):
    """
    returns suggested selling price to achieve target margin and expected profit per unit.
    margin_pct = (price - ingredient_cost) / price => price = ingredient_cost / (1 - margin_pct)
    """
    if target_margin_pct >= 100:
        return None, None
    m = target_margin_pct / 100.0
    if m >= 1.0:
        return None, None
    suggested_price = ingredient_cost_per_unit / (1 - m) if (1 - m) != 0 else None
    if suggested_price is None:
        return None, None
    profit = suggested_price - ingredient_cost_per_unit
    return round(suggested_price, 2), round(profit, 2)

def demand_predictor(history_df, item_name, periods_ahead=1):
    """
    Simple linear trend predictor on quantity sold over time for given item.
    history_df must contain 'ItemName', 'SavedAt'(iso str), 'QuantitySold'.
    Returns predicted next period sales (float) or None if not enough data.
    """
    df = history_df[history_df["ItemName"] == item_name].copy()
    if df.shape[0] < 3:
        return None  # not enough data
    # convert dates to ordinal
    df["t"] = pd.to_datetime(df["SavedAt"]).map(datetime.toordinal)
    df = df.sort_values("t")
    x = df["t"].values.astype(float)
    y = df["QuantitySold"].values.astype(float)
    # fit line
    try:
        coef = np.polyfit(x, y, 1)  # degree 1
        trend = np.poly1d(coef)
        last_t = x[-1]
        future_t = last_t + periods_ahead * 7  # interpret period as 1 week ~ 7 days
        pred = float(trend(future_t))
        return max(pred, 0.0)
    except Exception:
        return None

def ensure_history_file():
    if not os.path.exists(HISTORY_CSV):
        df = pd.DataFrame(columns=[
            "ItemName","SellingPrice","IngredientCost","ProfitPerUnit","TotalProfit","Margin(%)","QuantitySold","SavedAt","StrategySummary"
        ])
        df.to_csv(HISTORY_CSV, index=False)

def append_history(record):
    ensure_history_file()
    df = pd.read_csv(HISTORY_CSV)
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df.to_csv(HISTORY_CSV, index=False)

# ------- Session State init -------
if "ingredients_list" not in st.session_state:
    st.session_state.ingredients_list = []
if "last_calc" not in st.session_state:
    st.session_state.last_calc = None
if "saved_items" not in st.session_state:
    # saved_items kept in-memory; we also write to CSV
    st.session_state.saved_items = []
# Load history CSV into session once
ensure_history_file()
history_df = pd.read_csv(HISTORY_CSV)
st.session_state.history_df = history_df

# ------- Layout -------
st.title("🍽️ Advanced Restaurant Profit Simulator — (All features)")

sidebar_actions = st.sidebar
sidebar_actions.markdown("## Options")
if st.sidebar.button("Reload history"):
    st.session_state.history_df = pd.read_csv(HISTORY_CSV)
    st.success("History reloaded from disk.")

# Optional: allow user to upload ingredient cost master CSV (Ingredient, CostPerUnit) to auto-fill
st.sidebar.markdown("### Optional: Ingredient master (CSV)")
ing_master_file = st.sidebar.file_uploader("Upload ingredients_cost.csv (optional)", type=["csv"])
ing_master_df = None
if ing_master_file is not None:
    try:
        ing_master_df = pd.read_csv(ing_master_file)
        st.sidebar.success("Ingredient master loaded.")
    except Exception as e:
        st.sidebar.error("Failed to read CSV: " + str(e))
# Sidebar: download full history
if st.sidebar.button("Download full history CSV"):
    ensure_history_file()
    with open(HISTORY_CSV, "rb") as f:
        st.sidebar.download_button("Download history", f, file_name="menu_history.csv", mime="text/csv")

# ------- Pages -------
page = st.sidebar.radio("Navigation", ["Simulator (single item)", "Dashboard (all items)"], index=0)

# ---------- SIMULATOR ----------
if page == "Simulator (single item)":
    st.header("Single Item Analyzer")

    # two columns: left input + strategy, right charts & save
    left, right = st.columns([1, 1])

    with left:
        # Item selection and units
        item_list = ["Cold Coffee", "Paneer Butter Masala", "Veg Sandwich", "Mango Shake", "Pasta", "Burger", "Pizza", "Masala Dosa"]
        sel = st.selectbox("Select item (or choose Custom)", item_list + ["Custom..."])
        if sel == "Custom...":
            item_name = st.text_input("Enter custom item name", value="")
        else:
            item_name = sel

        qty_sold = st.number_input("Quantity sold (period)", min_value=1, step=1, value=1)
        unit = detect_unit(item_name)
        st.caption(f"Detected unit: **{unit}**")

        st.markdown("#### Ingredients (add multiple)")
        col_ing = st.columns([3,1,1])
        ing_name = col_ing[0].text_input("Ingredient name", key="ing_name")
        # auto fill cost from master if available
        default_cost = 0.0
        if ing_master_df is not None and ing_name:
            row = ing_master_df[ing_master_df['Ingredient'].str.lower()==ing_name.lower()]
            if not row.empty and 'CostPerUnit' in row.columns:
                default_cost = float(row.iloc[0]['CostPerUnit'])
        ing_qty = col_ing[1].number_input("Qty used", min_value=0.0, step=0.1, key="ing_qty")
        ing_cost = col_ing[2].number_input("Cost per unit (₹)", min_value=0.0, step=0.01, value=default_cost, key="ing_cost")

        if st.button("➕ Add ingredient"):
            if not ing_name:
                st.warning("Please enter ingredient name.")
            else:
                total = ing_qty * ing_cost
                st.session_state.ingredients_list.append({
                    "Ingredient": ing_name,
                    "QtyUsed": ing_qty,
                    "CostPerUnit": ing_cost,
                    "TotalCost": total
                })
                st.success(f"Added ingredient: {ing_name}")

        if st.session_state.ingredients_list:
            st.markdown("**Ingredients added (per unit)**")
            ing_df = pd.DataFrame(st.session_state.ingredients_list)
            ing_df_display = ing_df.copy()
            ing_df_display["TotalCost"] = ing_df_display["TotalCost"].apply(rupee)
            st.table(ing_df_display[["Ingredient","QtyUsed","CostPerUnit","TotalCost"]])

        if st.button("Clear ingredients"):
            st.session_state.ingredients_list = []
            st.success("Ingredients cleared")

        st.markdown("#### Pricing")
        selling_price = st.number_input("Selling price per unit (₹)", min_value=0.0, step=0.01, value=0.0)

        # Price optimization suggestions shown here (left)
        st.markdown("##### Price optimization suggestions (target margins)")
        opt_cols = st.columns(3)
        targets = [20,30,40]
        for i,target in enumerate(targets):
            suggested_price, expected_profit = price_optimizer(
                sum(i2["TotalCost"] for i2 in st.session_state.ingredients_list) if st.session_state.ingredients_list else 0.0,
                target
            )
            txt = "—"
            if suggested_price:
                txt = f"Sell @ {rupee(suggested_price)} → profit/unit {rupee(expected_profit)}"
            opt_cols[i].write(f"Target {target}%")
            opt_cols[i].write(txt)

        if st.button("Calculate profit & generate strategy"):
            if not item_name:
                st.error("Please enter/select item name.")
            elif not st.session_state.ingredients_list:
                st.error("Add at least one ingredient.")
            else:
                total_ing_cost = sum(i["TotalCost"] for i in st.session_state.ingredients_list)
                profit_per_unit = selling_price - total_ing_cost
                total_profit = profit_per_unit * qty_sold
                margin_pct = (profit_per_unit / selling_price * 100) if selling_price>0 else 0.0
                # build last_calc
                st.session_state.last_calc = {
                    "ItemName": item_name,
                    "SellingPrice": round(selling_price,2),
                    "IngredientCost": round(total_ing_cost,2),
                    "ProfitPerUnit": round(profit_per_unit,2),
                    "TotalProfit": round(total_profit,2),
                    "Margin(%)": round(margin_pct,2),
                    "QuantitySold": int(qty_sold),
                    "Ingredients": st.session_state.ingredients_list.copy(),
                    "SavedAt": datetime.utcnow().isoformat()
                }
                # strategy paragraphs (detailed)
                paras = strategy_generator(item_name, qty_sold, margin_pct, st.session_state.ingredients_list)
                st.session_state.last_calc["StrategySummary"] = " ".join(paras[:2])  # short summary
                st.session_state.last_calc["StrategyDetailed"] = paras

                st.success("Calculated — see visualization & strategy on the right")

    with right:
        st.subheader("Result, charts & actions")

        if st.session_state.last_calc is None:
            st.info("No calculation yet — add ingredients and click 'Calculate profit & generate strategy'.")
        else:
            calc = st.session_state.last_calc
            st.markdown(f"### {calc['ItemName']} — Key numbers")
            c1,c2,c3 = st.columns(3)
            c1.metric("Ingredient cost", rupee(calc["IngredientCost"]))
            c2.metric("Selling price", rupee(calc["SellingPrice"]))
            c3.metric("Profit / unit", rupee(calc["ProfitPerUnit"]))

            # compact bar chart: Ingredient cost vs Selling price vs Profit/unit
            graph_df = pd.DataFrame({
                "Metric": ["Ingredient cost","Selling price","Profit per unit"],
                "Value": [calc["IngredientCost"], calc["SellingPrice"], calc["ProfitPerUnit"]]
            })
            fig = px.bar(graph_df, x="Metric", y="Value", text="Value", color="Metric",
                         color_discrete_sequence=["#EF553B","#00CC96","#636EFA"])
            fig.update_traces(texttemplate="₹%{text:.2f}", textposition="outside")
            fig.update_layout(height=380, width=520, margin=dict(t=30,b=10))
            st.plotly_chart(fig, use_container_width=False)

            # Strategy (detailed) displayed here under ingredients
            st.markdown("### Strategy (detailed)")
            for para in calc.get("StrategyDetailed", []):
                st.write(para)

            # Demand prediction (from history)
            pred = demand_predictor(st.session_state.history_df, calc["ItemName"])
            if pred is not None:
                st.info(f"Demand predictor (next period estimate): ~{int(round(pred))} units")
            else:
                st.info("Demand predictor: need 3+ historical records of this item to predict (saved in history).")

            # Price optimizer quick buttons (set selling price to suggested)
            st.markdown("### Quick price optimization")
            price_opt_cols = st.columns(3)
            for i, tgt in enumerate([20,30,40]):
                suggested_price, _ = price_optimizer(calc["IngredientCost"], tgt)
                if suggested_price:
                    if price_opt_cols[i].button(f"Set price → {tgt}% (₹{suggested_price})"):
                        # set selling price in session and recalc
                        st.session_state.last_calc["SellingPrice"] = suggested_price
                        st.success(f"Selling price set to ₹{suggested_price} in the current result (re-run calc to save).")

            # Save to dashboard/history
            if st.button("Save result to Dashboard & History"):
                rec = {
                    "ItemName": calc["ItemName"],
                    "SellingPrice": calc["SellingPrice"],
                    "IngredientCost": calc["IngredientCost"],
                    "ProfitPerUnit": calc["ProfitPerUnit"],
                    "TotalProfit": calc["TotalProfit"],
                    "Margin(%)": calc["Margin(%)"],
                    "QuantitySold": calc["QuantitySold"],
                    "SavedAt": calc["SavedAt"],
                    "StrategySummary": calc.get("StrategySummary","")
                }
                # append to in-memory saved items
                st.session_state.saved_items.append(rec)
                # append to disk history
                append_history(rec)
                # reload history session copy
                st.session_state.history_df = pd.read_csv(HISTORY_CSV)
                st.success("Saved to Dashboard and history CSV ✅")

# ---------- DASHBOARD ----------
elif page == "Dashboard (all items)":
    st.header("Dashboard — All items & analytics")

    # load history and combine with session saved items (dedupe by timestamp)
    hist = pd.read_csv(HISTORY_CSV) if os.path.exists(HISTORY_CSV) else pd.DataFrame()
    session_saved = pd.DataFrame(st.session_state.saved_items) if st.session_state.saved_items else pd.DataFrame()
    # merge/display: prefer history on disk (includes previously saved)
    combined = pd.concat([hist, session_saved], ignore_index=True, sort=False) if not hist.empty else session_saved
    if combined.empty:
        st.info("No saved items yet. Save calculations from Simulator page first.")
    else:
        # compute color badges column as marker emoji + numeric margin
        def badge(m):
            try:
                m = float(m)
            except:
                return ""
            if m >= 40:
                return "🟢"
            elif m >= 20:
                return "🟡"
            elif m >= 10:
                return "🟠"
            else:
                return "🔴"

        combined_display = combined.copy()
        # Format currency columns
        for col in ["SellingPrice","IngredientCost","ProfitPerUnit","TotalProfit"]:
            if col in combined_display.columns:
                combined_display[col] = combined_display[col].apply(rupee)
        # ensure StrategySummary present
        if "StrategySummary" not in combined_display.columns:
            combined_display["StrategySummary"] = ""
        # add Badge column
        combined_display["Badge"] = combined["Margin(%)"].apply(lambda m: badge(m) if "Margin(%)" in combined.columns else "")
        # reorder columns for display
        cols_order = ["Badge","ItemName","QuantitySold","SellingPrice","IngredientCost","ProfitPerUnit","TotalProfit","Margin(%)","StrategySummary","SavedAt"]
        cols_order = [c for c in cols_order if c in combined_display.columns]
        st.markdown("### Items table (sorted by Margin % — high to low)")
        combined_display_sorted = combined_display.sort_values(by="Margin(%)", ascending=False, na_position="last")
        st.dataframe(combined_display_sorted[cols_order], use_container_width=True)

        # Download combined CSV
        csv = combined.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download combined report (CSV)", data=csv, file_name="menu_history_combined.csv", mime="text/csv")

        # Trend plot: select item to view history + demand prediction
        st.markdown("---")
        st.subheader("Item history & trend (select item)")
        item_options = combined["ItemName"].unique().tolist()
        item_sel = st.selectbox("Choose item to inspect", item_options)
        if item_sel:
            item_hist = pd.read_csv(HISTORY_CSV)
            item_hist = item_hist[item_hist["ItemName"]==item_sel].copy()
            if not item_hist.empty:
                item_hist["SavedAt_dt"] = pd.to_datetime(item_hist["SavedAt"])
                item_hist = item_hist.sort_values("SavedAt_dt")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=item_hist["SavedAt_dt"], y=item_hist["QuantitySold"], mode="lines+markers", name="QuantitySold"))
                fig.add_trace(go.Bar(x=item_hist["SavedAt_dt"], y=item_hist["TotalProfit"], name="TotalProfit", yaxis="y2", opacity=0.6))
                # dual axis
                fig.update_layout(
                    title=f"History for {item_sel}",
                    xaxis_title="Saved at",
                    yaxis_title="Quantity Sold",
                    yaxis2=dict(title="Total Profit (₹)", overlaying="y", side="right"),
                    height=450
                )
                st.plotly_chart(fig, use_container_width=True)

                # demand predictor
                pred = demand_predictor(item_hist, item_sel)
                if pred is not None:
                    st.info(f"Predicted next-period sales (simple trend): ~{int(round(pred))} units")
                else:
                    st.info("Not enough history to predict (need >=3 saved records for this item).")
            else:
                st.info("No history rows for selected item yet.")

        # clear history option
        st.markdown("---")
        clear_col1, clear_col2 = st.columns([1,1])
        with clear_col1:
            if st.button("Clear in-memory saved items"):
                st.session_state.saved_items = []
                st.success("Cleared session-saved items (history CSV untouched).")
        with clear_col2:
            if st.button("Clear history CSV (permanent)"):
                # Confirm action
                if os.path.exists(HISTORY_CSV):
                    os.remove(HISTORY_CSV)
                    ensure_history_file()
                    st.session_state.history_df = pd.read_csv(HISTORY_CSV)
                    st.success("History CSV cleared.")
                else:
                    st.info("No history file found.")

# Footer
st.markdown("---")
# signature purposely omitted per request
