import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
import plotly.graph_objects as go

# --- FUNCIONES GLOBALES CACHEADAS ---
@st.cache_data
def convert_df_to_csv(df): 
    """Convierte el dataframe a un formato CSV descargable de forma eficiente."""
    return df.to_csv(index=False).encode('utf-8')

# --- MODELOS MATEMÁTICOS ---
def model_langmuir(x, Bmax, Kd):
    return (Bmax * x / (Kd + x))

def model_mass_action(x, b, Kd, protconc):
    return b * ((x + protconc + Kd) - np.sqrt(np.square(x + protconc + Kd) - 4 * x * protconc)) / 2

# --- FUNCIONES DE GRÁFICADO (PLOTLY) ---
def create_epitope_buildup_plot(df, t_short):
    """Genera un gráfico con las curvas STD build-up y puntos intermedios hasta 6s."""
    fig = go.Figure()
    
    t_max_plot = max(6.5, t_short * 3)
    t_smooth = np.linspace(0, t_max_plot, 100)
    
    base_points = [0.0, t_short, 1.5, 3.0, 4.5, 6.0]
    t_markers = np.unique(np.sort(base_points))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # CORRECCIÓN: Usamos enumerate para que 'i' sea siempre un entero (0, 1, 2...) 
    # independientemente de cómo sea el índice del DataFrame.
    for i, (idx, row) in enumerate(df.iterrows()):
        proton = str(row['Proton Name'])
        std_long = row['STD long'] 
        ksat = row['ksat']
        
        color = colors[i % len(colors)]
        
        y_smooth = std_long * (1 - np.exp(-ksat * t_smooth))
        
        fig.add_trace(go.Scatter(
            x=t_smooth, y=y_smooth,
            mode='lines',
            name=f'{proton} (Curve)',
            line=dict(width=3, color=color),
            legendgroup=proton
        ))
        
        y_markers = std_long * (1 - np.exp(-ksat * t_markers))
        
        fig.add_trace(go.Scatter(
            x=t_markers, y=y_markers,
            mode='markers',
            name=f'{proton} (Points)',
            marker=dict(size=10, color=color, line=dict(width=2, color='DarkSlateGrey')),
            legendgroup=proton,
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, t_max_plot], y=[std_long, std_long],
            mode='lines',
            line=dict(width=1.5, color=color, dash='dot'),
            legendgroup=proton,
            showlegend=False,
            hoverinfo='skip'
        ))

    fig.update_layout(
        title="STD Build-up Curves per Proton",
        xaxis_title="Saturation Time (s)",
        yaxis_title="STD Factor",
        template="plotly_white",
        hovermode="x unified"
    )
    return fig

def create_fit_plot(df, x_col, y_col, model_func, popt, title, protconc=None):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df[x_col], y=df[y_col], 
        mode='markers', 
        name='Experimental Data',
        marker=dict(color='#D05732', size=10, line=dict(width=2, color='DarkSlateGrey'))
    ))
    
    x_smooth = np.linspace(0, max(df[x_col]) * 1.1, 100)
    
    if protconc is not None:
        y_smooth = model_func(x_smooth, popt[0], popt[1], protconc)
    else:
        y_smooth = model_func(x_smooth, popt[0], popt[1])
        
    fig.add_trace(go.Scatter(
        x=x_smooth, y=y_smooth, 
        mode='lines', 
        name='Fitted Curve',
        line=dict(color='#2D708E', width=3, dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Ligand Concentration (µM)',
        yaxis_title='STD_AF0',
        template='plotly_white',
        hovermode="x unified"
    )
    return fig

def get_aggrid_options(df):
    js_add_row = JsCode("""
        function(e) {
            let api = e.api;
            let rowPos = e.rowIndex + 1; 
            api.applyTransaction({addIndex: rowPos, add: [{}]})    
        };
    """)
    
    cellRenderer_addButton = JsCode('''
        class BtnCellRenderer {
            init(params) {
                this.eGui = document.createElement('div');
                this.eGui.innerHTML = `
                <span>
                    <style>
                    .btn_add {
                        background-color: #71DC87;
                        border: 2px solid black;
                        color: #D05732;
                        text-align: center;
                        display: inline-block;
                        font-size: 12px;
                        font-weight: bold;
                        height: 2em;
                        width: 10em;
                        border-radius: 12px;
                        padding: 0px;
                        cursor: pointer;
                    }
                    </style>
                    <button class="btn_add">&#x2193; Add</button>
                </span>
            `;
            }
            getGui() { return this.eGui; }
        };
    ''')
    
    gd = GridOptionsBuilder.from_dataframe(df)
    gd.configure_default_column(editable=True)
    gd.configure_column(field='🔧', onCellClicked=js_add_row, cellRenderer=cellRenderer_addButton)
    return gd.build()

def main_menu():
    st.set_page_config(page_title="STD NMR App", layout="wide")
    menu_options = ['Binding Epitope Determination', 'Dissociation Constant Determination']
    choice = st.sidebar.selectbox('Main Menu', menu_options)
    
    if choice == 'Binding Epitope Determination':
        st.write("""### Determination of STD NMR Binding Epitopes""")
        st.sidebar.header('CSV File Upload Option')
        
        uploaded_file1 = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        tsat_short = st.sidebar.number_input("Short Saturation Time (s)", value=0.75, min_value=0.1, max_value=1.5)
        
        df_to_process = None

        if uploaded_file1 is not None:
            df1 = pd.read_csv(uploaded_file1, header=None)
            df1.columns = ['Proton Name', 'STD short', 'STD long']
            df_to_process = df1
        else:
            d = {'Proton Name': ["H1", "H2", "H3"], 'STD short': [1.7, 4.3, 4.8], 'STD long': [4.8, 14.3, 12.2]}
            df_initial = pd.DataFrame(data=d)
            gridoptions = get_aggrid_options(df_initial)
            
            with st.form('STD NMR Reduced Dataset Form'):
                response = AgGrid(df_initial, gridOptions=gridoptions, editable=True, allow_unsafe_jscode=True, theme='balham', height=250, key='grid_epitope')
                submitted = st.form_submit_button("Confirm item(s) 🔒", type="primary")
                
            if submitted:
                # CORRECCIÓN: Usamos response['rowData'] para evitar errores de reindexado
                raw_df = pd.DataFrame(response['rowData'])
                df_to_process = raw_df.copy()

        if df_to_process is not None and not df_to_process.empty:
            st.subheader("Results")
            try:
                df_to_process['STD long'] = pd.to_numeric(df_to_process['STD long'], errors='coerce')
                df_to_process['STD short'] = pd.to_numeric(df_to_process['STD short'], errors='coerce')
                df_to_process = df_to_process.dropna(subset=['STD long', 'STD short'])

                df_to_process['ksat'] = (-np.log((df_to_process['STD long'] - df_to_process['STD short']) / df_to_process['STD long'])) / tsat_short
                df_to_process['STD0'] = df_to_process['ksat'] * df_to_process['STD long']
                
                max_std0 = np.max(df_to_process['STD0'])
                if max_std0 > 0:
                    df_to_process['Epitope (%)'] = (df_to_process['STD0'] / max_std0) * 100
                
                st.dataframe(df_to_process, use_container_width=True)
                
                st.write("### STD Build-up Kinetics")
                fig_buildup = create_epitope_buildup_plot(df_to_process, tsat_short)
                st.plotly_chart(fig_buildup, use_container_width=True)
                
                csv = convert_df_to_csv(df_to_process)
                st.download_button("Press to Download 🗳️", csv, "STD0_Reduced_Dataset.csv", "text/csv")
            except Exception as e:
                st.error(f"Error in calculations: {e}")

    elif choice == 'Dissociation Constant Determination':
        st.write("""### Dissociation Constant Determination""")
        uploaded_file2 = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        tsat_short = st.sidebar.number_input("Short Saturation Time (s)", value=0.75, min_value=0.1, max_value=1.5)
        protconc = st.sidebar.number_input("Total Protein Concentration (µM)", value=20.0)

        df_to_process = None

        if uploaded_file2 is not None:
            df2 = pd.read_csv(uploaded_file2, header=None)
            df2.columns = ['Ligand Concentration (µM)', 'STD short', 'STD long']
            df_to_process = df2
        else:
            d = {'Ligand Concentration (µM)': [150, 300, 600, 1500], 'STD short': [37, 27, 16, 9], 'STD long': [59, 50, 39, 26]}
            df_initial = pd.DataFrame(data=d)
            gridoptions = get_aggrid_options(df_initial)
            
            with st.form('STD NMR Dissociation Form'):
                response = AgGrid(df_initial, gridOptions=gridoptions, editable=True, allow_unsafe_jscode=True, theme='balham', height=250, key='grid_kd')
                submitted = st.form_submit_button("Confirm item(s) 🔒", type="primary")
            
            if submitted:
                # CORRECCIÓN: Usamos response['rowData']
                raw_df = pd.DataFrame(response['rowData'])
                df_to_process = raw_df.copy()

        if df_to_process is not None and not df_to_process.empty:
            try:
                for col in ['Ligand Concentration (µM)', 'STD short', 'STD long']:
                    df_to_process[col] = pd.to_numeric(df_to_process[col], errors='coerce')
                df_to_process = df_to_process.dropna()

                df_to_process['ksat'] = (-np.log((df_to_process['STD long'] - df_to_process['STD short']) / df_to_process['STD long'])) / tsat_short
                df_to_process['STD0'] = df_to_process['ksat'] * df_to_process['STD long']
                df_to_process['ratio'] = df_to_process['Ligand Concentration (µM)'] / protconc
                df_to_process['STD_AF0'] = df_to_process['STD0'] * df_to_process['ratio']
                
                X = df_to_process["Ligand Concentration (µM)"].values
                Y = df_to_process["STD_AF0"].values
                
                tab1, tab2 = st.tabs(["Langmuir Isotherm", "Law of Mass Action"])

                with tab1:
                    try:
                        ans1, cov1 = curve_fit(model_langmuir, X, Y, absolute_sigma=False, bounds=[[0,0], [5000,5000]])
                        res_langmuir = pd.DataFrame([{'Kd Langmuir': ans1[1], 'Standard Deviation': np.sqrt(np.diag(cov1))[1]}])
                        st.dataframe(res_langmuir, use_container_width=True)
                        fig_lang = create_fit_plot(df_to_process, "Ligand Concentration (µM)", "STD_AF0", model_langmuir, ans1, "Binding Curve (Langmuir)")
                        st.plotly_chart(fig_lang, use_container_width=True)
                    except: st.warning("Langmuir fit failed.")

                with tab2:
                    try:
                        fit_func_mass = lambda x, b, Kd: model_mass_action(x, b, Kd, protconc)
                        ans2, cov2 = curve_fit(fit_func_mass, X, Y, absolute_sigma=False, bounds=[[0,0], [5000,5000]]) 
                        res_mass = pd.DataFrame([{'Kd Law of Mass': ans2[1], 'Standard Deviation': np.sqrt(np.diag(cov2))[1]}])
                        st.dataframe(res_mass, use_container_width=True)
                        fig_mass = create_fit_plot(df_to_process, "Ligand Concentration (µM)", "STD_AF0", model_mass_action, ans2, "Binding Curve (Mass Action)", protconc)
                        st.plotly_chart(fig_mass, use_container_width=True)
                    except: st.warning("Mass Action fit failed.")

            except Exception as e:
                st.error(f"Data processing error: {e}")

if __name__ == '__main__':
    main_menu()