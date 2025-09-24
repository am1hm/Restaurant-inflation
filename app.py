import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# =========================
# إعدادات الصفحة
# =========================
st.set_page_config(page_title="حاسبة التضخم", layout="wide")
st.markdown("""
<style>
/* تعديلات على الستايل */
.kpi {
  padding: 16px; border-radius: 16px; background: #0e1117; border: 1px solid #2b2f3a; 
}
.kpi h3 { margin: 0 0 6px 0; font-size: 0.9rem; color: #9aa4b2; }
.kpi .big { font-size: 1.4rem; font-weight: 700; }
.kpi .delta-up { color: #ff6b6b; }
.kpi .delta-down { color: #19c37d; }
.small { color:#9aa4b2; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

st.title("📈 حاسبة تضخم أسعار المطاعم")
st.caption("أداة لتحليل أسعار المواد الغذائية ومقارنة الموردين - خاصة بالمطاعم")

# =========================
# رفع الملف - هنا بتكون البداية
# =========================
file = st.file_uploader("ارفع ملف المشتريات (CSV أو Excel)", type=["csv", "xlsx", "xls"])
if not file:
    st.info("مافي ملف مرفوع. أمثلة على الأعمدة: تاريخ, مادة, مورد, سعر_الوحدة")
    st.stop()

# شيك على نوع الملف
ext = os.path.splitext(file.name)[1].lower()
if ext == ".csv":
    df = pd.read_csv(file)
else:
    # هذي المكتبات لازم تكون مثبتة عشان تقرأ الإكسل
    engine = "openpyxl" if ext == ".xlsx" else "xlrd"
    df = pd.read_excel(file, engine=engine)

# =========================
# تعيين الأعمدة - عشان نتعامل مع أي تنسيق
# =========================
st.sidebar.subheader("تعيين الأعمدة")

# هذي الأسماء البديلة عشان نتعرف على الأعمدة تلقائياً
aliases = {
    "date": ["date", "purchase_date", "transaction_date"],
    "ingredient": ["ingredient", "item", "material"],
    "category": ["category", "group"],
    "supplier": ["supplier", "vendor", "market"],
    "quantity": ["quantity", "qty"],
    "unit": ["unit", "uom"],
    "unit_price": ["unit_price", "price_per_unit", "price"],
    "currency": ["currency", "currency_name"]
}

# دالة بسيطة تختار العمود المناسب
def smart_pick(col, candidates):
    cols_lower = {c.lower(): c for c in df.columns}
    for a in candidates:
        if a in cols_lower:
            return cols_lower[a]
    return None

defaults = {k: smart_pick(k, v) for k, v in aliases.items()}

colmap = {}
for key in ["date", "ingredient", "supplier", "unit_price", "category", "quantity", "unit", "currency"]:
    options = ["—"] + list(df.columns)
    default = defaults.get(key)
    idx = options.index(default) if default in options else 0
    sel = st.sidebar.selectbox(f"{key}", options, index=idx)
    if sel != "—":
        colmap[key] = sel

# نعيد تسمية الأعمدة عشان نوحدها
df = df.rename(columns={v: k for k, v in colmap.items()})

# الأعمدة الأساسية اللي لازم تكون موجودة
required = ["date", "ingredient", "supplier", "unit_price"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"الأعمدة الناقصة: {missing}. راجع تعيين الأعمدة أو الملف.")
    st.stop()

# =========================
# تنظيف البيانات - أهم جزء
# =========================
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).copy()

# تنظيف النصوص
df["ingredient"] = df["ingredient"].astype(str).str.strip()
df["supplier"] = df["supplier"].astype(str).str.strip()
df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
df = df.dropna(subset=["unit_price"]).copy()

# لو ما في بعض الأعمدة نضيفها افتراضياً
if "category" not in df.columns: 
    df["category"] = "غير مصنف"
if "quantity" not in df.columns: 
    df["quantity"] = np.nan
if "unit" not in df.columns: 
    df["unit"] = np.nan
if "currency" not in df.columns: 
    df["currency"] = "SAR"

# تحويل العملات - الأسعار التقريبية
currency_map_to_SAR = {"SAR": 1.0, "USD": 3.75, "EUR": 4.0, "QAR": 1.02} 
df["currency"] = df["currency"].astype(str).str.upper().str.strip()
df["unit_price_sar"] = df.apply(
    lambda r: float(r["unit_price"]) * currency_map_to_SAR.get(str(r["currency"]), 1.0),
    axis=1
)

# توحيد الوحدات - يمكن تحتاج تعديل حسب بياناتك
unit_norm = {"kg": 1.0, "g": 0.001, "l": 1.0, "pack": 1.0, "pcs": 1.0, "كجم": 1.0, "جرام": 0.001}
df["unit"] = df["unit"].astype(str).str.lower().str.strip()
df["unit_price_norm"] = df.apply(
    lambda r: r["unit_price_sar"] / unit_norm.get(r["unit"], 1.0) if pd.notna(r["unit"]) else r["unit_price_sar"], 
    axis=1
)

# نستخرج الشهر والسنة
df["month"] = df["date"].dt.to_period("M").astype(str)
df["year"] = df["date"].dt.year.astype(int)

# =========================
# الفلاتر - عشان نركز على البيانات المهمة
# =========================
with st.sidebar:
    st.subheader("الفلاتر")
    
    ingredients = sorted(df["ingredient"].unique().tolist())
    chosen_ings = st.multiselect("المواد", ingredients, default=ingredients[:min(len(ingredients), 10)])

    suppliers = sorted(df["supplier"].unique().tolist())
    chosen_supp = st.multiselect("الموردين", suppliers, default=suppliers)

    date_min, date_max = df["date"].min().date(), df["date"].max().date()
    date_range = st.date_input("الفترة الزمنية", value=(date_min, date_max), min_value=date_min, max_value=date_max)

    st.divider()
    iqr_toggle = st.toggle("فلترة الأسعار الشاذة", value=True)
    iqr_k = st.slider("حدة الفلترة", min_value=1.0, max_value=3.0, value=1.5, step=0.1)

# تطبيق الفلترات الأساسية
if isinstance(date_range, tuple):
    start_d, end_d = date_range
else:
    start_d, end_d = date_min, date_max

use = df[(df["date"].dt.date >= start_d) & (df["date"].dt.date <= end_d)].copy()
if chosen_ings: 
    use = use[use["ingredient"].isin(chosen_ings)]
if chosen_supp: 
    use = use[use["supplier"].isin(chosen_supp)]
    
if use.empty:
    st.warning("لا يوجد بيانات تنطبق على الفلاتر المحددة.")
    st.stop()

# فلترة الأسعار الشاذة - هذي مهمة عشان نتائج دقيقة
def iqr_filter(g, col="unit_price_norm", k=1.5):
    if len(g) < 5:
        return g  # ما نفلتر المجموعات الصغيرة
    q1, q3 = g[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - k*iqr, q3 + k*iqr
    return g[(g[col] >= low) & (g[col] <= high)]

if iqr_toggle:
    use = use.groupby(["ingredient", "supplier"], group_keys=False).apply(lambda g: iqr_filter(g, k=iqr_k))

# =========================
# التحليلات الشهرية - هنا بتكون الحسبة
# =========================
monthly = use.groupby(["ingredient", "month"], as_index=False)["unit_price_norm"].mean()
monthly = monthly.sort_values(["ingredient", "month"])
monthly["MoM%"] = monthly.groupby("ingredient")["unit_price_norm"].pct_change() * 100
monthly["YoY%"] = monthly.groupby("ingredient")["unit_price_norm"].pct_change(12) * 100

# سلة الأسعار العامة
basket = monthly.groupby("month", as_index=False)["unit_price_norm"].mean().rename(columns={"unit_price_norm": "basket_price"})
basket = basket.sort_values("month")
basket["MoM%"] = basket["basket_price"].pct_change() * 100
basket["YoY%"] = basket["basket_price"].pct_change(12) * 100

# أحدث شهر وأرقامه
latest_month = basket["month"].max()
prev_month = basket["month"].sort_values().shift().dropna().max()

basket_latest = float(basket.loc[basket["month"] == latest_month, "basket_price"].values[0])
basket_mom = float(basket.loc[basket["month"] == latest_month, "MoM%"].values[0]) if latest_month in set(basket["month"]) else np.nan
basket_yoy = float(basket.loc[basket["month"] == latest_month, "YoY%"].values[0]) if latest_month in set(basket["month"]) else np.nan

# أكثر مادة ارتفعت
last_m = monthly.groupby("ingredient").tail(1).dropna(subset=["MoM%"])
top_riser = last_m.sort_values("MoM%", ascending=False).head(1)

# أكثر مادة مستقرة
def last_6m_stability(g):
    g = g.tail(6)
    return pd.Series({"std6": g["unit_price_norm"].std()})

stab = monthly.groupby("ingredient").apply(last_6m_stability).dropna().sort_values("std6")
most_stable = stab.head(1)

# تحليل الموردين لمادة معينة
st.sidebar.subheader("بطاقة المورد")
sel_ing_supplier = st.sidebar.selectbox("اختار مادة", sorted(use["ingredient"].unique()))

by_ms = use[use["ingredient"] == sel_ing_supplier].groupby(["supplier", "month"], as_index=False)["unit_price_norm"].mean()
by_ms = by_ms.sort_values(["supplier", "month"])

def last3_mean(g):
    return pd.Series({"avg_last3": g.tail(3)["unit_price_norm"].mean(), "last_price": g.tail(1)["unit_price_norm"].values[0]})

supplier_card = by_ms.groupby("supplier").apply(last3_mean).reset_index().dropna()
best_supplier_row = supplier_card.sort_values("avg_last3").head(1)

# =========================
# لوحة المعلومات - النتائج
# =========================
st.subheader("النظرة العامة")

c0, c1, c2, c3 = st.columns(4)

with c0:
    st.markdown(f'<div class="kpi"><h3>آخر شهر في البيانات</h3><div class="big">{latest_month}</div><div class="small">{start_d} → {end_d}</div></div>', unsafe_allow_html=True)

with c1:
    delta_class = "delta-up" if basket_mom and basket_mom > 0 else "delta-down"
    st.markdown(f'<div class="kpi"><h3>التضخم الشهري</h3><div class="big {delta_class}">{basket_mom:.2f}%</div><div class="small">مقارنة بالشهر الماضي</div></div>', unsafe_allow_html=True)

with c2:
    if np.isnan(basket_yoy):
        yoy_txt = "—"
        delta_class2 = ""
    else:
        delta_class2 = "delta-up" if basket_yoy > 0 else "delta-down"
        yoy_txt = f"{basket_yoy:.2f}%"
    st.markdown(f'<div class="kpi"><h3>التضخم السنوي</h3><div class="big {delta_class2}">{yoy_txt}</div><div class="small">مقارنة بالعام الماضي</div></div>', unsafe_allow_html=True)

with c3:
    if not top_riser.empty:
        ing = top_riser["ingredient"].values[0]
        momv = float(top_riser["MoM%"].values[0])
        color = "delta-up" if momv > 0 else "delta-down"
        st.markdown(f'<div class="kpi"><h3>أعلى مادة صعوداً</h3><div class="big {color}">{ing}: {momv:.1f}%</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="kpi"><h3>أعلى مادة صعوداً</h3><div class="big">—</div></div>', unsafe_allow_html=True)

c4, c5 = st.columns(2)
with c4:
    if not most_stable.empty:
        ing_s = most_stable.index[0]
        stdv = float(most_stable["std6"].values[0])
        st.markdown(f'<div class="kpi"><h3>أكثر مادة استقراراً</h3><div class="big">{ing_s}</div><div class="small">التغير: {stdv:.2f}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="kpi"><h3>أكثر مادة استقراراً</h3><div class="big">—</div></div>', unsafe_allow_html=True)

with c5:
    if not best_supplier_row.empty:
        bs_name = best_supplier_row["supplier"].values[0]
        bs_avg = float(best_supplier_row["avg_last3"].values[0])
        st.markdown(f'<div class="kpi"><h3>أفضل مورد ({sel_ing_supplier})</h3><div class="big">{bs_name}</div><div class="small">متوسط 3 أشهر: {bs_avg:.2f} ريال</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="kpi"><h3>أفضل مورد ({sel_ing_supplier})</h3><div class="big">—</div></div>', unsafe_allow_html=True)

st.divider()

# =========================
# التبويبات التفصيلية
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["أسعار المواد", "مقارنة الموردين", "أعلى التغيرات", "تحميل البيانات"])

with tab1:
    st.subheader("تطور أسعار مادة معينة")
    sel_ing = st.selectbox("اختار مادة", sorted(use["ingredient"].unique()))
    chart_ing = monthly[monthly["ingredient"] == sel_ing].copy()
    
    if chart_ing.empty:
        st.info("مافي بيانات لهذه المادة.")
    else:
        fig_line = px.line(chart_ing, x="month", y="unit_price_norm", markers=True, title=f"{sel_ing} - السعر الشهري")
        fig_line.update_layout(xaxis_title="الشهر", yaxis_title="ريال/وحدة")
        st.plotly_chart(fig_line, use_container_width=True)
        
        st.write("آخر 12 شهر:")
        st.dataframe(chart_ing[["month", "MoM%", "YoY%"]].tail(12).round(2), use_container_width=True)

with tab2:
    st.subheader("مقارنة أسعار الموردين لنفس المادة")
    sel_ing2 = st.selectbox("المادة", sorted(use["ingredient"].unique()), key="ing2")
    comp = use[use["ingredient"] == sel_ing2].groupby(["supplier", "month"], as_index=False)["unit_price_norm"].mean()
    
    if comp.empty:
        st.info("مافي بيانات للمقارنة.")
    else:
        fig_comp = px.line(comp, x="month", y="unit_price_norm", color="supplier", markers=True, title=f"{sel_ing2} - حسب المورد")
        fig_comp.update_layout(xaxis_title="الشهر", yaxis_title="ريال/وحدة")
        st.plotly_chart(fig_comp, use_container_width=True)

with tab3:
    st.subheader("أعلى المواد تغيراً في السعر")
    group_month = use.groupby(["ingredient", "month"], as_index=False)["unit_price_norm"].mean()
    group_month = group_month.sort_values("month")
    
    if group_month["month"].nunique() >= 2:
        last_m_str = group_month["month"].max()
        prev_m_str = sorted(group_month["month"].unique())[-2]
        last_prices = group_month[group_month["month"] == last_m_str][["ingredient", "unit_price_norm"]].set_index("ingredient")
        prev_prices = group_month[group_month["month"] == prev_m_str][["ingredient", "unit_price_norm"]].set_index("ingredient")
        joined = last_prices.join(prev_prices, lsuffix="_last", rsuffix="_prev", how="left")
        joined["MoM%"] = ((joined["unit_price_norm_last"] - joined["unit_price_norm_prev"]) / joined["unit_price_norm_prev"] * 100)
        joined = joined.dropna().sort_values("MoM%", ascending=False).head(15)
        st.dataframe(joined.round(2), use_container_width=True)
    else:
        st.info("نحتاج شهرين على الأقل للمقارنة.")

with tab4:
    st.subheader("تحميل البيانات بعد المعالجة")
    out = use.copy()
    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button("تحميل CSV", csv, "بيانات_المعالجة.csv", "text/csv")

# =========================
# عرض عينة من البيانات
# =========================
st.divider()
with st.expander("عرض بيانات العينة (أول 100 صف)"):
    st.dataframe(use.head(100), use_container_width=True)