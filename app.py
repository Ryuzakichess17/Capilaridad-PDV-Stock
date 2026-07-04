import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# Configuración inicial de la página
st.set_page_config(page_title="Análisis de Capilaridad", layout="wide")

@st.cache_data
def load_and_process_data():
    df = pd.read_excel(
        "data/Base_PDVs_lat_long_2.xlsx",
        dtype={'NUMERODEDOCUMENTO': str}
    )
    
    # --- 1. PROCESAR ESTADOS Y MATRIZ DE TRANSICIÓN ---
    feb_base = df[df['PERIODO_CREACION'] == 202602]
    feb_in_feb = feb_base[feb_base['PERIODO_ACTIVIDAD'] == 202602]
    feb_in_jun = feb_base[feb_base['PERIODO_ACTIVIDAD'] == 202606]
    
    feb_sales_feb = set(feb_in_feb[feb_in_feb['FLAG_ACTIVO_ACTIV'] == 'CON VENTA']['NUMERODEDOCUMENTO'])
    feb_no_sales_feb = set(feb_in_feb[feb_in_feb['FLAG_ACTIVO_ACTIV'] == 'SIN VENTA']['NUMERODEDOCUMENTO'])
    
    feb_sales_jun = set(feb_in_jun[feb_in_jun['FLAG_ACTIVO_ACTIV'] == 'CON VENTA']['NUMERODEDOCUMENTO'])
    feb_no_sales_jun = set(feb_in_jun[feb_in_jun['FLAG_ACTIVO_ACTIV'] == 'SIN VENTA']['NUMERODEDOCUMENTO'])
    
    # --- CÁLCULO DE FLUJOS DIRECTOS ---
    cv_to_cv = len(feb_sales_feb.intersection(feb_sales_jun)) 
    cv_to_sv = len(feb_sales_feb.intersection(feb_no_sales_jun)) 
    cv_to_out = len(feb_sales_feb) - cv_to_cv - cv_to_sv 
    
    sv_to_cv = len(feb_no_sales_feb.intersection(feb_sales_jun)) 
    sv_to_sv = len(feb_no_sales_feb.intersection(feb_no_sales_jun)) 
    sv_to_out = len(feb_no_sales_feb) - sv_to_cv - sv_to_sv 
    
    flujos = {
        'cv_to_cv': cv_to_cv, 'cv_to_sv': cv_to_sv, 'cv_to_out': cv_to_out,
        'sv_to_cv': sv_to_cv, 'sv_to_sv': sv_to_sv, 'sv_to_out': sv_to_out
    }
    
    dropped_dnis = list(feb_sales_feb - feb_sales_jun)
    
    # --- 2. EXTRAER DATOS PARA EL MAPA ---
    columnas_mapa = ['NUMERODEDOCUMENTO', 'LATITUD', 'LONGITUD', 'SOCIO', 'KAM', 'PERIODO_CREACION', 'PERIODO_ACTIVIDAD', 'FLAG_ACTIVO_ACTIV', 'ACTIVIDAD', 'TIPO', 'FLAG']
    
    dropped_locs = feb_in_feb[feb_in_feb['NUMERODEDOCUMENTO'].isin(dropped_dnis)][columnas_mapa].dropna(subset=['LATITUD', 'LONGITUD'])
    
    df_con_venta = df[df['FLAG_ACTIVO_ACTIV'] == 'CON VENTA']
    ultimos_periodos = df_con_venta[df_con_venta['NUMERODEDOCUMENTO'].isin(dropped_dnis)].groupby('NUMERODEDOCUMENTO')['PERIODO_ACTIVIDAD'].max().reset_index()
    ultimos_periodos.rename(columns={'PERIODO_ACTIVIDAD': 'ULTIMO_PERIODO'}, inplace=True)
    
    dropped_locs = dropped_locs.merge(ultimos_periodos, on='NUMERODEDOCUMENTO', how='left')
    dropped_locs['PERIODO_ACTIVIDAD'] = dropped_locs['ULTIMO_PERIODO'].fillna(dropped_locs['PERIODO_ACTIVIDAD'])
    
    new_logins = df[(df['PERIODO_CREACION'] > 202602) & (df['PERIODO_ACTIVIDAD'] == 202606)][columnas_mapa].dropna(subset=['LATITUD', 'LONGITUD'])
    
    return flujos, dropped_locs, new_logins, df

def get_matches_within_radius(dropped, new, radius_m):
    lat1 = np.radians(dropped['LATITUD'].values[:, np.newaxis])
    lon1 = np.radians(dropped['LONGITUD'].values[:, np.newaxis])
    lat2 = np.radians(new['LATITUD'].values)
    lon2 = np.radians(new['LONGITUD'].values)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000 
    distances = c * r
    
    i, j = np.where(distances <= radius_m)
    
    if len(i) > 0:
        export_df = pd.DataFrame({
            'DNI_PDV_Viejo': dropped.iloc[i]['NUMERODEDOCUMENTO'].values,
            'FLAG_Origen_Viejo': dropped.iloc[i]['FLAG'].values, 
            'TIPO_Viejo': dropped.iloc[i]['TIPO'].values, 
            'Socio_PDV_Viejo': dropped.iloc[i]['SOCIO'].values,
            'KAM_PDV_Viejo': dropped.iloc[i]['KAM'].values,
            'Ultimo_Periodo_Actividad_Viejo': dropped.iloc[i]['PERIODO_ACTIVIDAD'].values, 
            'FLAG_ACTIVO_ACTIV_Viejo': dropped.iloc[i]['FLAG_ACTIVO_ACTIV'].values, 
            'Cant_Ventas_Viejo': dropped.iloc[i]['ACTIVIDAD'].values,
            'Latitud_Viejo': dropped.iloc[i]['LATITUD'].values,
            'Longitud_Viejo': dropped.iloc[i]['LONGITUD'].values,
            
            'DNI_PDV_Nuevo': new.iloc[j]['NUMERODEDOCUMENTO'].values,
            'FLAG_Origen_Nuevo': new.iloc[j]['FLAG'].values, 
            'TIPO_Nuevo': new.iloc[j]['TIPO'].values, 
            'Socio_PDV_Nuevo': new.iloc[j]['SOCIO'].values,
            'KAM_PDV_Nuevo': new.iloc[j]['KAM'].values,
            'Creacion_PDV_Nuevo': new.iloc[j]['PERIODO_CREACION'].values,
            'Periodo_Actividad_Nuevo': new.iloc[j]['PERIODO_ACTIVIDAD'].values,
            'FLAG_ACTIVO_ACTIV_Nuevo': new.iloc[j]['FLAG_ACTIVO_ACTIV'].values, 
            'Cant_Ventas_Nuevo': new.iloc[j]['ACTIVIDAD'].values,
            'Latitud_Nuevo': new.iloc[j]['LATITUD'].values,
            'Longitud_Nuevo': new.iloc[j]['LONGITUD'].values,
            
            'Distancia_Metros': np.round(distances[i, j], 2)
        })
    else:
        export_df = pd.DataFrame()
        
    matched_new_pdvs_count = new.iloc[j]['NUMERODEDOCUMENTO'].nunique()
    
    dropped_matched = dropped.iloc[np.unique(i)].copy()
    dropped_matched['Tipo_Mapa'] = 'PDV Caído (Febrero)' 
    new_matched = new.iloc[np.unique(j)].copy()
    new_matched['Tipo_Mapa'] = 'PDV Nuevo (Reemplazo)' 
    
    map_data = pd.concat([dropped_matched, new_matched])
    
    map_data['PERIODO_CREACION'] = map_data['PERIODO_CREACION'].astype(str)
    map_data['PERIODO_ACTIVIDAD'] = map_data['PERIODO_ACTIVIDAD'].astype(str)
    
    # LÓGICA DE TAMAÑO Y ORDEN DE CAPAS
    map_data['Tamaño_Punto'] = map_data['Tipo_Mapa'].apply(lambda x: 10 if x == 'PDV Caído (Febrero)' else 7)
    map_data['Orden_Dibujo'] = map_data['Tipo_Mapa'].apply(lambda x: 0 if x == 'PDV Caído (Febrero)' else 1)
    map_data = map_data.sort_values(by='Orden_Dibujo')
    
    return matched_new_pdvs_count, map_data, export_df

# ==========================================
# INTERFAZ DE STREAMLIT (UI)
# ==========================================

st.title("🗺️ Análisis de Capilaridad y Re-creación de PDVs")

flujos, dropped_locs, new_logins, df_raw = load_and_process_data()

st.markdown("---")
st.subheader("📊 Cuadro de Movimientos: Febrero vs Junio")

total_cv_feb = flujos['cv_to_cv'] + flujos['cv_to_sv'] + flujos['cv_to_out']
total_sv_feb = flujos['sv_to_cv'] + flujos['sv_to_sv'] + flujos['sv_to_out']

st.markdown("#### 🟢 PDV STOCK + CAPTURA: (Con Venta en Febrero)")
c1, c2, c3, c4 = st.columns(4)
c1.metric(label="1️⃣ Stock Inicial", value=str(total_cv_feb))
c2.metric(label="✅ Retenidos (Venden en Jun)", value=str(flujos['cv_to_cv']))
c3.metric(label="🔻 Cayeron a Sin Venta", value=str(flujos['cv_to_sv']))
c4.metric(label="❌ Desaparecieron de Base", value=str(flujos['cv_to_out']))

st.markdown("#### 🟠 PDV STOCK + CAPTURA: (Sin Venta en Febrero)")
c5, c6, c7, c8 = st.columns(4)
c5.metric(label="2️⃣ Stock Inicial", value=str(total_sv_feb))
c6.metric(label="💤 Siguen Sin Venta", value=str(flujos['sv_to_sv']))
c7.metric(label="🚀 Despertaron (Venden en Jun)", value=str(flujos['sv_to_cv']))
c8.metric(label="❌ Desaparecieron de Base", value=str(flujos['sv_to_out']))

perdida_total = flujos['cv_to_sv'] + flujos['cv_to_out']
st.info(f"**Pérdida Real Total:** Se apagaron **{perdida_total} PDVs operativos** que daban ventas en Febrero.")

st.markdown("---")
col_izq, col_der = st.columns([1, 2])

with col_izq:
    st.subheader("Encontrar PDVs con la misma ubicación")
    radio = st.slider("Radio de búsqueda (en metros):", min_value=0, max_value=50, value=0, step=1)
    
    recreaciones_count, map_data, df_detalle = get_matches_within_radius(dropped_locs, new_logins, radio)
    st.metric(label=f"Nuevos PDVs a {radio}m o menos", value=str(recreaciones_count))
    
    st.markdown("---")
    st.subheader("🔍 Filtro de Enfoque")
    pdv_search = st.text_input("Ingresa un DNI de PDV:", "").strip()
    
    if pdv_search and not df_detalle.empty:
        df_detalle = df_detalle[(df_detalle['DNI_PDV_Viejo'] == pdv_search) | (df_detalle['DNI_PDV_Nuevo'] == pdv_search)]
        if df_detalle.empty:
            st.warning(f"No hay cruces para '{pdv_search}' a {radio}m.")
            map_data = pd.DataFrame()
        else:
            st.success(f"Enfocando PDV: {pdv_search}")
            dnis_a_mostrar = pd.concat([df_detalle['DNI_PDV_Viejo'], df_detalle['DNI_PDV_Nuevo']]).unique()
            map_data = map_data[map_data['NUMERODEDOCUMENTO'].isin(dnis_a_mostrar)]
            
    st.markdown("---")
    if not df_detalle.empty:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df_detalle.to_excel(writer, index=False, sheet_name='Cruce_Capilaridad')
        
        st.download_button(
            label="📥 Descargar Base Cruzada",
            data=buffer.getvalue(),
            file_name=f"Cruce_PDVs_Radio_{radio}m.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

with col_der:
    st.subheader("Visualización Geográfica")
    if map_data.empty:
        st.warning("No hay coincidencias para mostrar en el mapa.")
    else:
        fig = px.scatter_mapbox(
            map_data, lat="LATITUD", lon="LONGITUD", color="Tipo_Mapa",
            color_discrete_map={'PDV Caído (Febrero)': 'red', 'PDV Nuevo (Reemplazo)': 'green'},
            size="Tamaño_Punto",
            size_max=10,
            hover_name="Tipo_Mapa",
            hover_data={
                "LATITUD": False, "LONGITUD": False, "Tipo_Mapa": False, "Tamaño_Punto": False, "Orden_Dibujo": False,
                "NUMERODEDOCUMENTO": True, "FLAG": True, "TIPO": True, "SOCIO": True, "KAM": True, 
                "PERIODO_CREACION": True, "PERIODO_ACTIVIDAD": True, 
                "FLAG_ACTIVO_ACTIV": True, "ACTIVIDAD": True  
            },
            zoom=5 if not pdv_search else 10, 
            mapbox_style="open-street-map"
        )
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 3. AUTOPSIA DE PDVS CAÍDOS (Deep Dive)
# ==========================================
st.markdown("---")
st.title("📉 Analisis de PDVs Caídos")
st.markdown("Análisis histórico de los PDVs que eran productivos en Febrero y dejaron de vender hacia Junio.")

mes_inicio = 202602
mes_fin = 202606

dnis_caidos = dropped_locs['NUMERODEDOCUMENTO'].unique()

if len(dnis_caidos) > 0:
    df_historia_caidos = df_raw[
        (df_raw['NUMERODEDOCUMENTO'].isin(dnis_caidos)) & 
        (df_raw['PERIODO_ACTIVIDAD'] >= mes_inicio) & 
        (df_raw['PERIODO_ACTIVIDAD'] <= mes_fin)
    ].copy()
    
    df_historia_caidos['ACTIVIDAD'] = pd.to_numeric(df_historia_caidos['ACTIVIDAD'], errors='coerce').fillna(0)
    
    df_activos_mes = df_historia_caidos[df_historia_caidos['ACTIVIDAD'] > 0]
    
    tendencia = df_activos_mes.groupby('PERIODO_ACTIVIDAD').agg(
        Ventas=('ACTIVIDAD', 'sum'),
        LOGINs=('NUMERODEDOCUMENTO', 'nunique')
    ).reset_index()
    
    todos_los_meses = pd.DataFrame({'PERIODO_ACTIVIDAD': range(mes_inicio, mes_fin + 1)})
    tendencia = pd.merge(todos_los_meses, tendencia, on='PERIODO_ACTIVIDAD', how='left').fillna(0)
    
    tendencia['Caídos_vs_Mes_Anterior'] = tendencia['LOGINs'].diff().fillna(0).apply(lambda x: abs(x) if x < 0 else 0).astype(int)
    tendencia['Productividad'] = np.where(tendencia['LOGINs'] > 0, (tendencia['Ventas'] / tendencia['LOGINs']).round(2), 0)
    
    tendencia['PERIODO_ACTIVIDAD'] = tendencia['PERIODO_ACTIVIDAD'].astype(str)
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.subheader("Curva de Sobrevivencia (LOGINs productivos)")
        
        fig_tendencia = make_subplots(specs=[[{"secondary_y": True}]])

        fig_tendencia.add_trace(
            go.Bar(x=tendencia['PERIODO_ACTIVIDAD'], y=tendencia['LOGINs'], name="LOGINs (Vendieron)", marker_color='#3498DB', opacity=0.7),
            secondary_y=False,
        )
        
        fig_tendencia.add_trace(
            go.Scatter(x=tendencia['PERIODO_ACTIVIDAD'], y=tendencia['Ventas'], name="Total Ventas", mode='lines+markers', line=dict(color='#E74C3C', width=3)),
            secondary_y=True,
        )
        
        fig_tendencia.update_xaxes(type='category', title_text="Mes")
        fig_tendencia.update_yaxes(title_text="Cantidad de LOGINs", secondary_y=False)
        fig_tendencia.update_yaxes(title_text="Total de Ventas", secondary_y=True)
        fig_tendencia.update_layout(margin={"r":0,"t":30,"l":0,"b":0}, hovermode="x unified")
        
        st.plotly_chart(fig_tendencia, use_container_width=True)
        
        st.markdown("**Resumen Mensual:**")
        st.dataframe(tendencia.rename(columns={'PERIODO_ACTIVIDAD': 'Periodo'}), use_container_width=True, hide_index=True)
        
    with col_g2:
        st.subheader("Caídas según el SOCIO de Origen (Febrero)")
        foto_mes_base = df_historia_caidos[df_historia_caidos['PERIODO_ACTIVIDAD'] == mes_inicio]
        bajas_por_socio = foto_mes_base.drop_duplicates(subset=['NUMERODEDOCUMENTO'])['SOCIO'].value_counts().reset_index()
        bajas_por_socio.columns = ['SOCIO', 'Cantidad de PDVs Perdidos']
        
        fig_socio = px.bar(
            bajas_por_socio.head(10), x='SOCIO', y='Cantidad de PDVs Perdidos',
            color='Cantidad de PDVs Perdidos', color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_socio, use_container_width=True)

    # --- DESCARGA DEL DETALLE HISTÓRICO CORREGIDO + CRUCE DE REEMPLAZOS ---
    st.markdown("### 📥 Descargar Historico de los Caídos")
    
    df_pivot = df_historia_caidos.pivot_table(
        index=['NUMERODEDOCUMENTO', 'PERIODO_CREACION'], 
        columns='PERIODO_ACTIVIDAD', 
        values='ACTIVIDAD', 
        aggfunc='sum'
    ).reset_index().fillna(0)
    
    if not df_detalle.empty:
        reemplazos = df_detalle.groupby('DNI_PDV_Viejo')['DNI_PDV_Nuevo'].apply(lambda x: ', '.join(x.astype(str))).reset_index()
        df_pivot = df_pivot.merge(reemplazos, left_on='NUMERODEDOCUMENTO', right_on='DNI_PDV_Viejo', how='left')
        
        col_reemplazo = f'Reemplazo_a_{radio}m'
        df_pivot.rename(columns={'DNI_PDV_Nuevo': col_reemplazo}, inplace=True)
        df_pivot.drop(columns=['DNI_PDV_Viejo'], inplace=True)
        df_pivot[col_reemplazo] = df_pivot[col_reemplazo].fillna('Sin Reemplazo')
    else:
        df_pivot[f'Reemplazo_a_{radio}m'] = 'Sin Reemplazo'
    
    buffer_caidos = io.BytesIO()
    with pd.ExcelWriter(buffer_caidos, engine='xlsxwriter') as writer:
        df_pivot.to_excel(writer, index=False, sheet_name='Historia_Caidos')
    
    st.download_button(
        label=f"📥 Descargar Evolución (Con Flag de Reemplazos a {radio}m)",
        data=buffer_caidos.getvalue(),
        file_name=f"Autopsia_PDVs_Caidos_y_Reemplazos_{mes_inicio}_al_{mes_fin}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.success("No hubo caídas en este período para analizar.")
