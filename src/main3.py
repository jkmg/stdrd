import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import plotly.graph_objects as go

# --- CONFIGURACIÓN Y MODELOS ---
def model_langmuir(x, Bmax, Kd):
    return (Bmax * x / (Kd + x))

def model_mass_action(x, b, Kd, protconc):
    return b * ((x + protconc + Kd) - np.sqrt(np.square(x + protconc + Kd) - 4 * x * protconc)) / 2

# --- FUNCIONES DE APOYO ---
def create_epitope_buildup_plot(df, t_short):
    fig = go.Figure()
    t_max_plot = 6.5
    t_smooth = np.linspace(0, t_max_plot, 100)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, (_, row) in enumerate(df.iterrows()):
        proton, std_long, ksat = str(row['Proton Name']), row['STD long'], row['ksat']
        color = colors[i % len(colors)]
        y_smooth = std_long * (1 - np.exp(-ksat * t_smooth))
        fig.add_trace(go.Scatter(x=t_smooth, y=y_smooth, mode='lines', name=proton, line=dict(color=color)))
        
    fig.update_layout(title="STD Build-up Curves", template="plotly_white")
    return fig

# --- LÓGICA DE LA APLICACIÓN ---
def main_menu():
    st.set_page_config(page_title="STD NMR App", layout="wide")
    
    # Sidebar
    choice = st.sidebar.selectbox('Main Menu', ['Binding Epitope Determination', 'Dissociation Constant Determination'])
    tsat_short = st.sidebar.number_input("Short Saturation Time (s)", value=0.75)
    
    # 1. INICIALIZACIÓN DE DATOS (Session State)
    if 'df_epitope' not in st.session_state:
        st.session_state.df_epitope = pd.DataFrame({
            'Proton Name': ["H1", "H2", "H3"], 
            'STD short': [1.7, 4.3, 4.8], 
            'STD long': [4.8, 14.3, 12.2]
        })

    if 'df_kd' not in st.session_state:
        st.session_state.df_kd = pd.DataFrame({
            'Ligand Concentration (µM)': [150.0, 300.0, 600.0], 
            'STD short': [37.0, 27.0, 16.0], 
            'STD long': [59.0, 50.0, 39.0]
        })

    if choice == 'Binding Epitope Determination':
        st.write("### Binding Epitope Determination")
        
        # Mostrar AgGrid
        gb = GridOptionsBuilder.from_dataframe(st.session_state.df_epitope)
        gb.configure_default_column(editable=True)
        grid_res = AgGrid(
            st.session_state.df_epitope, 
            gridOptions=gb.build(), 
            update_mode=GridUpdateMode.VALUE_CHANGED,
            theme='balham',
            height=250
        )
        
        # Sincronizar cambios de la tabla al state
        st.session_state.df_epitope = pd.DataFrame(grid_res.data)

        # Botones de acción
        col1, col2 = st.columns(2)
        if col1.button("➕ Add Row"):
            new_row = pd.DataFrame([[f"H{len(st.session_state.df_epitope)+1}", 0.0, 0.0]], 
                                  columns=['Proton Name', 'STD short', 'STD long'])
            st.session_state.df_epitope = pd.concat([st.session_state.df_epitope, new_row], ignore_index=True)
            st.rerun()

        if col2.button("🚀 Calculate", type="primary"):
            try:
                df = st.session_state.df_epitope.copy()
                for c in ['STD short', 'STD long']: df[c] = pd.to_numeric(df[c], errors='coerce')
                df = df.dropna()
                
                df['ksat'] = (-np.log((df['STD long'] - df['STD short']) / df['STD long'])) / tsat_short
                df['STD0'] = df['ksat'] * df['STD long']
                df['Epitope (%)'] = (df['STD0'] / df['STD0'].max()) * 100
                
                st.dataframe(df)
                st.plotly_chart(create_epitope_buildup_plot(df, tsat_short), use_container_width=True)
            except Exception as e:
                st.error("Check your inputs. Ensure 'STD long' is greater than 'STD short' and no zeros.")

    elif choice == 'Dissociation Constant Determination':
        st.write("### Dissociation Constant Determination")
        protconc = st.sidebar.number_input("Protein Conc. (µM)", value=20.0)
        
        gb = GridOptionsBuilder.from_dataframe(st.session_state.df_kd)
        gb.configure_default_column(editable=True)
        grid_res = AgGrid(st.session_state.df_kd, gridOptions=gb.build(), update_mode=GridUpdateMode.VALUE_CHANGED, theme='balham', height=250)
        
        st.session_state.df_kd = pd.DataFrame(grid_res.data)

        col1, col2 = st.columns(2)
        if col1.button("➕ Add Concentration"):
            new_row = pd.DataFrame([[0.0, 0.0, 0.0]], columns=st.session_state.df_kd.columns)
            st.session_state.df_kd = pd.concat([st.session_state.df_kd, new_row], ignore_index=True)
            st.rerun()

        if col2.button("🚀 Calculate Kd", type="primary"):
            try:
                df = st.session_state.df_kd.copy()
                for c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
                df = df.dropna()
                
                df['ksat'] = (-np.log((df['STD long'] - df['STD short']) / df['STD long'])) / tsat_short
                df['STD0'] = df['ksat'] * df['STD long']
                df['STD_AF0'] = df['STD0'] * (df['Ligand Concentration (µM)'] / protconc)
                
                X, Y = df['Ligand Concentration (µM)'].values, df['STD_AF0'].values
                popt, _ = curve_fit(model_langmuir, X, Y, bounds=(0, 5000))
                
                st.success(f"Kd: {popt[1]:.2f} µM")
                
                # Gráfico simple
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=X, y=Y, mode='markers', name='Data'))
                x_fit = np.linspace(0, max(X)*1.2, 100)
                fig.add_trace(go.Scatter(x=x_fit, y=model_langmuir(x_fit, *popt), mode='lines', name='Fit'))
                st.plotly_chart(fig)
            except:
                st.error("Error in calculation. Check your data.")

if __name__ == '__main__':
    main_menu()