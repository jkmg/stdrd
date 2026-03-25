import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
# Importados según tus requirements (aunque no se usen explícitamente en esta vista)
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

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

# --- CONFIGURACIÓN DE INTERFAZ ---
def get_aggrid_options(df):
    """Genera las opciones de configuración para AgGrid con el botón 'Add'."""
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
        st.write("""### Determination of STD NMR Binding Epitopes using the Reduced Dataset Approach from STD factors at 2 saturation times""")
        st.sidebar.header('CSV File Upload Option')
        
        uploaded_file1 = st.sidebar.file_uploader("**You can also upload a CSV File (without headers)**", type=["csv"])
        tsat_short = st.sidebar.number_input("**Enter the Short Saturation Time (s)**", value=0.75, min_value=0.1, max_value=1.5)
        
        df_to_process = None

        if uploaded_file1 is not None:
            df1 = pd.read_csv(uploaded_file1, header=None)
            df1.columns = ['Proton Name', 'STD short', 'STD long']
            df_to_process = df1
            st.write("#### Uploaded Data:")
            st.dataframe(df1)
        else:
            st.write("##### Enter below the Short Saturation Time employed, in seconds")
            d = {
                'Proton Name': ["H1", "H2", "H3"],
                'STD short': [1.7, 4.3, 4.8],
                'STD long': [4.8, 14.3, 12.2]
            }
            df_initial = pd.DataFrame(data=d)
            gridoptions = get_aggrid_options(df_initial)
            
            with st.form('STD NMR Reduced Dataset Form'):
                st.write("""#### :blue[Complete the table below] :writing_hand:""")
                response = AgGrid(df_initial, gridOptions=gridoptions, editable=True, allow_unsafe_jscode=True, theme='balham', height=250)
                st.write("*Note: Don't forget to hit enter ↩ on new entry.*")
                submitted = st.form_submit_button("Confirm item(s) 🔒", type="primary")
                
            if submitted:
                # Extraemos y limpiamos los datos de AgGrid
                raw_df = pd.DataFrame(response['data'])
                # Eliminamos filas que el usuario haya añadido pero dejado vacías
                raw_df = raw_df.dropna(subset=['STD short', 'STD long']) 
                df_to_process = raw_df.copy()

        if df_to_process is not None and not df_to_process.empty:
            st.subheader("Results")
            try:
                # Coerción de tipos para evitar errores si el usuario introdujo texto accidentalmente
                df_to_process['STD long'] = pd.to_numeric(df_to_process['STD long'], errors='coerce')
                df_to_process['STD short'] = pd.to_numeric(df_to_process['STD short'], errors='coerce')
                df_to_process = df_to_process.dropna() # Limpiamos NaNs generados por la coerción

                df_to_process['ksat'] = (-np.log((df_to_process['STD long'] - df_to_process['STD short']) / df_to_process['STD long'])) / tsat_short
                df_to_process['STD0'] = df_to_process['ksat'] * df_to_process['STD long']
                df_to_process['Epitope (%)'] = (df_to_process['STD0'] / np.max(df_to_process['STD0'])) * 100
                
                # Opcional: limpiar las columnas originales para la vista final si se desea
                # df_to_process.drop(['STD short', 'STD long'], inplace=True, axis=1)
                
                st.table(df_to_process)
                
                col1, col2 = st.columns(2)
                csv = convert_df_to_csv(df_to_process)
                col1.write("Save locally")
                col2.download_button("Press to Download 🗳️", csv, "STD0_Reduced_Dataset.csv", "text/csv")
            except Exception as e:
                st.error(f"Error in calculations: {e}. Please check your inputs to avoid division by zero or invalid logs.")

    elif choice == 'Dissociation Constant Determination':
        st.write("""### Dissociation Constant Determination by STD NMR using the Reduced Dataset Approach""")
        st.sidebar.header('CSV File Upload Option')
        
        uploaded_file2 = st.sidebar.file_uploader("**Upload a CSV File (Concentration, STD short, STD long)**", type=["csv"])
        tsat_short = st.sidebar.number_input("**Enter the Short Saturation Time (s)**", value=0.75, min_value=0.10, max_value=1.50)
        protconc = st.sidebar.number_input("**Enter the Total Protein Concentration (µM)**", value=20.0)

        df_to_process = None

        if uploaded_file2 is not None:
            df2 = pd.read_csv(uploaded_file2, header=None)
            df2.columns = ['Ligand Concentration (µM)', 'STD short', 'STD long']
            df_to_process = df2
        else:
            d = {
                'Ligand Concentration (µM)': [150, 300, 600, 1500], 
                'STD short': [37, 27, 16, 9],
                'STD long': [59, 50, 39, 26]
            }
            df_initial = pd.DataFrame(data=d)
            gridoptions = get_aggrid_options(df_initial)
            
            with st.form('STD NMR Dissociation Form'):
                st.write("""#### :blue[Complete the table below] :writing_hand:""")
                response = AgGrid(df_initial, gridOptions=gridoptions, editable=True, allow_unsafe_jscode=True, theme='balham', height=250)
                st.write("*Note: Don't forget to hit enter ↩ on new entry.*")
                submitted = st.form_submit_button("Confirm item(s) 🔒", type="primary")
            
            if submitted:
                raw_df = pd.DataFrame(response['data'])
                raw_df = raw_df.dropna(subset=['Ligand Concentration (µM)', 'STD short', 'STD long'])
                df_to_process = raw_df.copy()

        if df_to_process is not None and not df_to_process.empty:
            try:
                # Coerción numérica segura
                for col in ['Ligand Concentration (µM)', 'STD short', 'STD long']:
                    df_to_process[col] = pd.to_numeric(df_to_process[col], errors='coerce')
                df_to_process = df_to_process.dropna()

                df_to_process['ksat'] = (-np.log((df_to_process['STD long'] - df_to_process['STD short']) / df_to_process['STD long'])) / tsat_short
                df_to_process['STD0'] = df_to_process['ksat'] * df_to_process['STD long']
                df_to_process['ratio'] = df_to_process['Ligand Concentration (µM)'] / protconc
                df_to_process['STD_AF0'] = df_to_process['STD0'] * df_to_process['ratio']
                
                X = df_to_process["Ligand Concentration (µM)"].values
                Y = df_to_process["STD_AF0"].values

                # --- Ajuste Langmuir ---
                try:
                    ans1, cov1 = curve_fit(model_langmuir, X, Y, absolute_sigma=False, bounds=[[0,0], [5000,5000]])
                    stdev1 = np.sqrt(np.diag(cov1))
                    
                    res_langmuir = df_to_process.copy()
                    res_langmuir['Kd Langmuir'] = ans1[1]
                    res_langmuir['Standard Deviation'] = stdev1[1]
                    res_langmuir.drop(['Ligand Concentration (µM)', 'STD short', 'STD long', 'ksat', 'STD0', 'ratio', 'STD_AF0'], inplace=True, axis=1, errors='ignore')
                    res_langmuir.drop_duplicates(subset=['Kd Langmuir'], ignore_index=True, inplace=True)
                    
                    st.subheader("Results from Langmuir Isotherm")
                    st.dataframe(res_langmuir)
                    csv1 = convert_df_to_csv(res_langmuir)
                    st.download_button("Download Langmuir Results 🗳️", csv1, "Kd_Langmuir.csv", "text/csv", key='btn_langmuir')
                except RuntimeError:
                    st.warning("Curve fitting for Langmuir failed to find optimal parameters.")

                # --- Ajuste Law of Mass Action ---
                try:
                    # Usamos una función lambda para pasar la concentración de proteína dinámicamente
                    fit_func_mass = lambda x, b, Kd: model_mass_action(x, b, Kd, protconc)
                    ans2, cov2 = curve_fit(fit_func_mass, X, Y, absolute_sigma=False, bounds=[[0,0], [5000,5000]]) 
                    stdev2 = np.sqrt(np.diag(cov2))
                    
                    res_mass = df_to_process.copy()
                    res_mass['Kd Law of Mass'] = ans2[1]
                    res_mass['Standard Deviation'] = stdev2[1]
                    res_mass.drop(['Ligand Concentration (µM)', 'STD short', 'STD long', 'ksat', 'STD0', 'ratio', 'STD_AF0'], inplace=True, axis=1, errors='ignore')
                    res_mass.drop_duplicates(subset=['Kd Law of Mass'], ignore_index=True, inplace=True)
                    
                    st.subheader("Results from Law of Mass Action")
                    st.dataframe(res_mass)
                    csv2 = convert_df_to_csv(res_mass)
                    st.download_button("Download Mass Action Results 🗳️", csv2, "Kd_MassLaw.csv", "text/csv", key='btn_mass')
                except RuntimeError:
                    st.warning("Curve fitting for Law of Mass Action failed to find optimal parameters.")

            except Exception as e:
                st.error(f"Data processing error: {e}. Check your inputs for mathematically invalid states.")


if __name__ == '__main__':
    main_menu()
