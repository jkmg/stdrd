import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, DataReturnMode, GridUpdateMode
import plotly.graph_objects as go

# --- FUNCIONES GLOBALES CACHEADAS ---
@st.cache_data
def convert_df_to_csv(df): 
    return df.to_csv(index=False).encode('utf-8')

# --- MODELOS MATEMÁTICOS ---
def model_langmuir(x, Bmax, Kd):
    return (Bmax * x / (Kd + x))

def model_mass_action(x, b, Kd, protconc):
    return b * ((x + protconc + Kd) - np.sqrt(np.square(x + protconc + Kd) - 4 * x * protconc)) / 2

# --- FUNCIONES DE GRÁFICADO ---
def create_epitope_buildup_plot(df, t_short):
    fig = go.Figure()
    t_max_plot = max(6.5, t_short * 3)
    t_smooth = np.linspace(0, t_max_plot, 100)
    base_points = [0.0, t_short, 1.5, 3.0, 4.5, 6.0]
    t_markers = np.unique(np.sort(base_points))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # i es siempre entero gracias a enumerate, evitando el error de string formatting
    for i, (idx, row) in enumerate(df.iterrows()):
        proton = str(row['Proton Name'])
        std_long = row['STD long'] 
        ksat = row['ksat']
        color = colors[i % len(colors)]
        
        y_smooth = std_long * (1 - np.exp(-ksat * t_smooth))
        fig.add_trace(go.Scatter(x=t_smooth, y=y_smooth, mode='lines', name=f'{proton} (Curve)',
                                line=dict(width=3, color=color), legendgroup=proton))
        
        y_markers = std_long * (1 - np.exp(-ksat * t_markers))
        fig.add_trace(go.Scatter(x=t_markers, y=y_markers, mode='markers', name=f'{proton} (Points)',
                                marker=dict(size=10, color=color, line=dict(width=2, color='DarkSlateGrey')),
                                legendgroup=proton, showlegend=False))
    
    fig.update_layout(title="STD Build-up Curves per Proton", xaxis_title="Saturation Time (s)", 
                      yaxis_title="STD Factor", template="plotly_white")
    return fig

def create_fit_plot(df, x_col, y_col, model_func, popt, title, protconc=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode='markers', name='Exp. Data',
                            marker=dict(color='#D05732', size=10, line=dict(width=2, color='DarkSlateGrey'))))
    x_smooth = np.linspace(0, max(df[x_col]) * 1.1, 100)
    y_smooth = model_func(x_smooth, *popt, protconc) if protconc else model_func(x_smooth, *popt)
    fig.add_trace(go.Scatter(x=x_smooth, y=y_smooth, mode='lines', name='Fit', line=dict(color='#2D708E', width=3, dash='dash')))
    fig.update_layout(title=title, xaxis_title='Ligand Conc. (µM)', yaxis_title='STD_AF0', template='plotly_white')
    return fig

def get_aggrid_options(df):
    # JS corregido para asegurar que la fila añadida tenga una estructura compatible
    js_add_row = JsCode("""
        function(e) {
            let api = e.api;
            let rowPos = e.rowIndex + 1; 
            api.applyTransaction({addIndex: rowPos, add: [{}]});
        };
    """)
    
    cellRenderer_addButton = JsCode('''
        class BtnCellRenderer {
            init(params) {
                this.eGui = document.createElement('div');
                this.eGui.innerHTML = `<button style="background-color: #71DC87; border-radius: 8px; cursor: pointer;">+ Add</button>`;
            }
            getGui() { return this.eGui; }
        };
    ''')
    
    gd = GridOptionsBuilder.from_dataframe(df)
    gd.configure_default_column(editable=True, groupable=False)
    gd.configure_column(field='🔧', onCellClicked=js_add_row, cellRenderer=cellRenderer_addButton, editable=False, width=100)
    return gd.build()

def main_menu():
    st.set_page_config(page_title="STD NMR App", layout="wide")
    choice = st.sidebar.selectbox('Main Menu', ['Binding Epitope Determination', 'Dissociation Constant Determination'])
    
    if choice == 'Binding Epitope Determination':
        st.write("### Binding Epitope Determination")
        tsat_short = st.sidebar.number_input("Short Saturation Time (s)", value=0.75)
        
        # DataFrame inicial limpio
        d = {'Proton Name': ["H1", "H2", "H3"], 'STD short': [1.7, 4.3, 4.8], 'STD long': [4.8, 14.3, 12.2]}
        df_initial = pd.DataFrame(data=d).reset_index(drop=True)
        
        gridoptions = get_aggrid_options(df_initial)
        
        # USAR ESTA CONFIGURACIÓN DE AGGRID PARA EVITAR ERRORES DE ÍNDICE
        response = AgGrid(
            df_initial, 
            gridOptions=gridoptions, 
            editable=True, 
            allow_unsafe_jscode=True, 
            theme='balham', 
            height=300,
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED, # CRITICO
            update_mode=GridUpdateMode.MODEL_CHANGED,            # CRITICO
            key='grid_epitope'
        )
        
        # Recuperar los datos usando .data (atributo)
        raw_df = pd.DataFrame(response.data) 
        
        if not raw_df.empty:
            try:
                # Convertir a números y limpiar filas vacías
                for col in ['STD short', 'STD long']:
                    raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce')
                df_proc = raw_df.dropna(subset=['STD long', 'STD short']).copy()

                if not df_proc.empty:
                    # Cálculos
                    df_proc['ksat'] = (-np.log((df_proc['STD long'] - df_proc['STD short']) / df_proc['STD long'])) / tsat_short
                    df_proc['STD0'] = df_proc['ksat'] * df_proc['STD long']
                    max_std0 = df_proc['STD0'].max()
                    df_proc['Epitope (%)'] = (df_proc['STD0'] / max_std0) * 100 if max_std0 > 0 else 0
                    
                    st.dataframe(df_proc, use_container_width=True)
                    st.plotly_chart(create_epitope_buildup_plot(df_proc, tsat_short), use_container_width=True)
            except Exception as e:
                st.info("Complete all fields in the table to see results.")

    elif choice == 'Dissociation Constant Determination':
        st.write("### Dissociation Constant Determination")
        tsat_short = st.sidebar.number_input("Short Saturation Time (s)", value=0.75)
        protconc = st.sidebar.number_input("Total Protein Concentration (µM)", value=20.0)

        df_init_kd = pd.DataFrame({'Ligand Concentration (µM)': [150, 300, 600], 'STD short': [37, 27, 16], 'STD long': [59, 50, 39]}).reset_index(drop=True)
        gridoptions = get_aggrid_options(df_init_kd)
        
        response = AgGrid(
            df_init_kd, 
            gridOptions=gridoptions, 
            editable=True, 
            allow_unsafe_jscode=True, 
            theme='balham', 
            height=300,
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            update_mode=GridUpdateMode.MODEL_CHANGED,
            key='grid_kd'
        )
        
        raw_df = pd.DataFrame(response.data)
        
        if not raw_df.empty:
            try:
                for col in raw_df.columns:
                    if col != '🔧': raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce')
                df_proc = raw_df.dropna().copy()
                
                if len(df_proc) >= 3:
                    df_proc['ksat'] = (-np.log((df_proc['STD long'] - df_proc['STD short']) / df_proc['STD long'])) / tsat_short
                    df_proc['STD0'] = df_proc['ksat'] * df_proc['STD long']
                    df_proc['STD_AF0'] = df_proc['STD0'] * (df_proc['Ligand Concentration (µM)'] / protconc)
                    
                    X, Y = df_proc["Ligand Concentration (µM)"].values, df_proc["STD_AF0"].values
                    ans, _ = curve_fit(model_langmuir, X, Y, bounds=(0, 5000))
                    st.success(f"Calculated Kd (Langmuir): {ans[1]:.2f} µM")
                    st.plotly_chart(create_fit_plot(df_proc, "Ligand Concentration (µM)", "STD_AF0", model_langmuir, ans, "Kd Fit"), use_container_width=True)
            except:
                st.info("Fill the table values to calculate Kd.")

if __name__ == '__main__':
    main_menu()