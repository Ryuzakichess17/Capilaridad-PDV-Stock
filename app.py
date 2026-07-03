import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io

# Configuración inicial de la página
st.set_page_config(page_title="Análisis de Capilaridad y Canibalización", layout="wide")

# Función para cargar y procesar datos (con caché para que sea rápido)
@st.cache_data
def load_and_process_data():
    df = pd.read_excel(
        r"C:\Users\hector.andagua\Downloads\Entel\GEO_CAPILARIDAD_JUNIO\data\Base_PDVs_lat_long_2.xlsx",
        dtype={'NUMERODEDOCUMENTO': str}
    )
    
    # --- 1. PROCESAR ESTADOS DE LOS PDVs ---
    feb_base = df[df['PERIODO_CREACION'] == 202602]
    feb_in_feb = feb_base[feb_base['PERIODO_ACTIVIDAD'] == 202602]
    feb_in_jun = feb_base[feb_base['PERIODO_ACTIVIDAD'] == 202606]
    
    feb_sales_feb = feb_in_feb[feb_in_feb['FLAG_ACTIVO_ACTIV'] == 'CON VENTA']['NUMERODEDOCUMENTO'].unique()
    feb_sales_jun = feb_in_jun[feb_in_jun['FLAG_ACTIVO_ACTIV'] == 'CON VENTA']['NUMERODEDOCUMENTO'].unique()
    
    dropped_dnis = set(feb_sales_feb) - set(feb_sales_jun)
    dropped_in_jun_df = feb_in_jun[(feb_in_jun['NUMERODEDOCUMENTO'].isin(dropped_dnis)) & (feb_in_jun['FLAG_ACTIVO_ACTIV'] == 'SIN VENTA')]
    
    # Métricas para la tabla resumen
    siguen_con_venta = len(set(feb_sales_feb).intersection(set(feb_sales_jun)))
    pasaron_sin_venta = dropped_in_jun_df['NUMERODEDOCUMENTO'].nunique()
    desaparecieron = len(dropped_dnis) - pasaron_sin_venta
    
    resumen_df = pd.DataFrame({
        "Estado Actual (Junio) de la Base de Febrero": ["Siguen CON VENTA", "Pasaron a SIN VENTA (Aún existen)", "Desaparecieron de la base"],
        "Cantidad de PDVs": [siguen_con_venta, pasaron_sin_venta, desaparecieron] # CAMBIADO A PDVs
    })
    
    # --- 2. EXTRAER DATOS PARA EL MAPA ---
    columnas_mapa = ['NUMERODEDOCUMENTO', 'LATITUD', 'LONGITUD', 'SOCIO', 'KAM', 'PERIODO_CREACION', 'PERIODO_ACTIVIDAD', 'FLAG_ACTIVO_ACTIV', 'ACTIVIDAD']
    
    dropped_locs = feb_in_feb[feb_in_feb['NUMERODEDOCUMENTO'].isin(dropped_dnis)][columnas_mapa].dropna(subset=['LATITUD', 'LONGITUD'])
    new_logins = df[(df['PERIODO_CREACION'] > 202602) & (df['PERIODO_ACTIVIDAD'] == 202606)][columnas_mapa].dropna(subset=['LATITUD', 'LONGITUD'])
    
    return resumen_df, dropped_locs, new_logins

# Función matemática para calcular distancia y generar el cruce detallado para exportar
def get_matches_within_radius(dropped, new, radius_m):
    lat1 = np.radians(dropped['LATITUD'].values[:, np.newaxis])
    lon1 = np.radians(dropped['LONGITUD'].values[:, np.newaxis])
    lat2 = np.radians(new['LATITUD'].values)
    lon2 = np.radians(new['LONGITUD'].values)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000 # Radio de la Tierra en metros
    distances = c * r
    
    i, j = np.where(distances <= radius_m)
    
    # --- CONSTRUIR BASE DE EXPORTACIÓN (Renombrado a PDV) ---
    if len(i) > 0:
        export_df = pd.DataFrame({
            'DNI_PDV_Viejo': dropped.iloc[i]['NUMERODEDOCUMENTO'].values,
            'Socio_PDV_Viejo': dropped.iloc[i]['SOCIO'].values,
            'KAM_PDV_Viejo': dropped.iloc[i]['KAM'].values,
            'Estado_Venta_Viejo': dropped.iloc[i]['FLAG_ACTIVO_ACTIV'].values,
            'Cant_Ventas_Viejo': dropped.iloc[i]['ACTIVIDAD'].values,
            'Latitud_Viejo': dropped.iloc[i]['LATITUD'].values,
            'Longitud_Viejo': dropped.iloc[i]['LONGITUD'].values,
            
            'DNI_PDV_Nuevo': new.iloc[j]['NUMERODEDOCUMENTO'].values,
            'Socio_PDV_Nuevo': new.iloc[j]['SOCIO'].values,
            'KAM_PDV_Nuevo': new.iloc[j]['KAM'].values,
            'Creacion_PDV_Nuevo': new.iloc[j]['PERIODO_CREACION'].values,
            'Periodo_Actividad_Nuevo': new.iloc[j]['PERIODO_ACTIVIDAD'].values,
            'Estado_Venta_Nuevo': new.iloc[j]['FLAG_ACTIVO_ACTIV'].values,
            'Cant_Ventas_Nuevo': new.iloc[j]['ACTIVIDAD'].values,
            'Latitud_Nuevo': new.iloc[j]['LATITUD'].values,
            'Longitud_Nuevo': new.iloc[j]['LONGITUD'].values,
            
            'Distancia_Metros': np.round(distances[i, j], 2)
        })
    else:
        export_df = pd.DataFrame()
        
    matched_new_pdvs_count = new.iloc[j]['NUMERODEDOCUMENTO'].nunique()
    
    # Preparar datos consolidados para el mapa interactivo
    dropped_matched = dropped.iloc[np.unique(i)].copy()
    dropped_matched['Tipo'] = 'PDV Caído (Febrero)' # CAMBIADO A PDV
    
    new_matched = new.iloc[np.unique(j)].copy()
    new_matched['Tipo'] = 'PDV Nuevo (Reemplazo)' # CAMBIADO A PDV
    
    map_data = pd.concat([dropped_matched, new_matched])
    return matched_new_pdvs_count, map_data, export_df

# ==========================================
# INTERFAZ DE STREAMLIT (UI)
# ==========================================

st.title("🗺️ Análisis de Capilaridad y Re-creación de PDVs") # TITULO CAMBIADO
st.markdown("Este panel analiza qué sucedió con los puntos de venta originales de Febrero y permite detectar posibles **re-creaciones de cuentas** evaluando la cercanía geográfica (GPS) entre PDVs dados de baja y PDVs nuevos.")

# Cargar los datos
resumen_df, dropped_locs, new_logins = load_and_process_data()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. ¿Qué pasó con la base inicial?")
    st.dataframe(resumen_df, hide_index=True)
    st.info("💡 **Conclusión:** La gran mayoría de cuentas que reportan caída no migraron ni desaparecieron; simplemente cambiaron su estatus a 'SIN VENTA'.")
    
    st.markdown("---")
    st.subheader("2. Simulador de Canibalización")
    st.markdown("Ajusta el radio para ver cuántos PDVs nuevos se crearon cerca de un PDV que pasó a 'Sin Venta'.")
    
    # Slider interactivo de 0 a 50 metros
    radio = st.slider("Radio de búsqueda (en metros):", min_value=0, max_value=50, value=0, step=1)
    
    # Calcular cruce
    recreaciones_count, map_data, df_detalle = get_matches_within_radius(dropped_locs, new_logins, radio)
    
    st.metric(label=f"Nuevos PDVs detectados a {radio}m o menos", value=f"{recreaciones_count} PDVs")
    if radio == 0:
        st.caption("*(Con 0m buscamos coordenadas idénticas al milímetro)*")
        
    # --- NUEVO: FILTRO POR DNI PARA ENFOCAR EL MAPA ---
    st.markdown("---")
    st.subheader("🔍 Filtro de Enfoque Rápido")
    pdv_search = st.text_input("Ingresa un DNI de PDV (Viejo o Nuevo) para enfocar el mapa en su ubicación:", "").strip()
    
    # Lógica de filtrado
    if pdv_search and not df_detalle.empty:
        # Filtramos la tabla de exportación para buscar solo donde coincida ese PDV
        df_detalle = df_detalle[(df_detalle['DNI_PDV_Viejo'] == pdv_search) | (df_detalle['DNI_PDV_Nuevo'] == pdv_search)]
        
        if df_detalle.empty:
            st.warning(f"No se encontraron cruces para el PDV '{pdv_search}' en un radio de {radio}m.")
            map_data = pd.DataFrame() # Vaciamos el mapa si no hay cruces
        else:
            st.success(f"Mostrando ubicación cruzada para el PDV: {pdv_search}")
            # Filtramos el mapa para mostrar ÚNICAMENTE este PDV y su(s) reemplazo(s)
            dnis_a_mostrar = pd.concat([df_detalle['DNI_PDV_Viejo'], df_detalle['DNI_PDV_Nuevo']]).unique()
            map_data = map_data[map_data['NUMERODEDOCUMENTO'].isin(dnis_a_mostrar)]
            
    # --- BOTÓN DE EXPORTACIÓN A EXCEL ---
    st.markdown("---")
    st.subheader("3. Exportar Hallazgos")
    if df_detalle.empty:
        st.warning("No hay registros disponibles para exportar.")
    else:
        # Si está filtrado, muestra cuántos cruces hay para ese PDV en particular. Si no, muestra el total.
        st.write(f"Se encontraron **{len(df_detalle)} parejas/cruces** para exportar.")
        
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df_detalle.to_excel(writer, index=False, sheet_name='Cruce_Capilaridad')
        
        st.download_button(
            label="📥 Descargar Base en Excel (.xlsx)",
            data=buffer.getvalue(),
            file_name=f"Cruce_PDVs_Radio_{radio}m.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        st.caption("Vista previa de la base a exportar:")
        st.dataframe(df_detalle[['DNI_PDV_Nuevo', 'Cant_Ventas_Nuevo', 'Distancia_Metros']].head(5), hide_index=True)

with col2:
    st.subheader("Visualización Geográfica")
    if map_data.empty:
        st.warning("No hay coincidencias para mostrar en el mapa.")
    else:
        # Crear el mapa interactivo
        fig = px.scatter_mapbox(
            map_data,
            lat="LATITUD",
            lon="LONGITUD",
            color="Tipo",
            color_discrete_map={
                'PDV Caído (Febrero)': 'red',
                'PDV Nuevo (Reemplazo)': 'green'
            },
            hover_name="Tipo",
            hover_data={
                "LATITUD": False,
                "LONGITUD": False,
                "Tipo": False,
                "NUMERODEDOCUMENTO": True,
                "SOCIO": True,
                "KAM": True,
                "PERIODO_CREACION": True,
                "PERIODO_ACTIVIDAD": True,
                "FLAG_ACTIVO_ACTIV": True,
                "ACTIVIDAD": True  
            },
            zoom=5 if not pdv_search else 15, # Si el usuario busca un PDV en específico, el mapa se acerca automáticamente (zoom 15)
            mapbox_style="open-street-map"
        )
        
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)