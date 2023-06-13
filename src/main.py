import streamlit as st
import streamlit_authenticator as stauth
import st_aggrid
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
from scipy.optimize import curve_fit
from st_aggrid.shared import GridUpdateMode
import pandas as pd
import numpy as np
import yaml
from yaml.loader import SafeLoader

with open('./authentication.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
    #config['preauthorized']
)

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    authenticator.logout('Logout', 'main', key='stdrd')
    st.write(f'Welcome *{name}*')
    #st.title('Some content')

    def main_menu():
        menu_options = ['Binding Epitope Determination', 'Dissociation Constant Determination']
        choice = st.sidebar.selectbox('Main Menu', menu_options)
        if choice == 'Binding Epitope Determination':
            st.write(""" ### Determination of STD NMR Binding Epitopes using the Reduced Dataset Approach from STD factors at 2 saturation times """)
            st.sidebar.header('CSV File Upload Option')
            # Collects user input features into dataframe
            uploaded_file1 = st.sidebar.file_uploader("**You can also upload a CSV File (without column headers) containing 3 columns, the first with the proton names, then the experimental STD factors at short (typically 0.5, 0.75 or 1 s; 0.75 s is recommended) and large (typically 6 or 8 s) saturation times**", type=["csv"])
            tsat_short1 = st.sidebar.number_input("**Enter the Short Saturation Time, in seconds**", value = 0.75, min_value=0.50, max_value=1.00)
            if uploaded_file1 is not None:
                df1 = pd.read_csv(uploaded_file1, header = None)
                df1.columns=['Proton Name', 'STD at short saturation time (s)', 'STD at long saturation time (s)']
                df1['ksat'] = (- np.log((df1['STD at long saturation time (s)'] - df1['STD at short saturation time (s)'])/df1['STD at long saturation time (s)']))/tsat_short1
                df1['STD0'] = df1.ksat*df1['STD at long saturation time (s)']
                df1['Epitope (%)'] = df1.STD0/(np.max(df1.STD0))*100
                df1.drop(['STD at short saturation time (s)', 'STD at long saturation time (s)'], inplace=True, axis=1)
                # Function
                @st.cache_data
                def convert_df1(data1): 
                    "Converts the data to a CSV format"
                    return data1.to_csv(index=False).encode('utf-8')
                st.subheader("Download Results")
                col1,col2 = st.columns(2)
                csv = convert_df1(df1)
                col1.write("Save locally")
                col2.download_button(
                "Press to Download 🗳️",
                csv,
                "STD0_Reduced_Dataset.csv",
                "text/csv",
                key='download-csv'
                )
            else:
                # Create an AgGrid table from a pandas DataFrame
                tsat_short_label = st.write("##### Enter below the Short Saturation Time employed, in seconds")
                tsat_short = st.number_input("**Enter the Short Saturation Time employed, in seconds**", value = 0.75, min_value=0.50, max_value=1.00, label_visibility='hidden')
                d = {'Proton Name': ["H1", "H2", "H3", "H4", "H5"],'STD at short saturation time (s)': [1.7,4.3,4.8,3.7,3.8],'STD at long saturation time (s)': [4.8,14.3,12.2,7.0,7.1]}
                df = pd.DataFrame(data = d)
                # Display the Dataframe in AgGrid
                # JavaScript function to add a new row to the AgGrid table
                js_add_row = JsCode("""
                function(e) {
                    let api = e.api;
                    let rowPos = e.rowIndex + 1; 
                    api.applyTransaction({addIndex: rowPos, add: [{}]})    
                };
                """     
                )
                # Cell renderer for the '🔧' column to render a button
                cellRenderer_addButton = JsCode('''
                    class BtnCellRenderer {
                        init(params) {
                            this.params = params;
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
                                }
                                </style>
                                <button id='click-button' 
                                    class="btn_add" 
                                    >&#x2193; Add</button>
                            </span>
                        `;
                        }
                        getGui() {
                            return this.eGui;
                        }
                    };
                    ''')
                # Create a GridOptionsBuilder object from our DataFrame
                gd = GridOptionsBuilder.from_dataframe(df)
                #gd_results = GridOptionsBuilder.from_dataframe(results)
                # Configure the default column to be editable
                # sets the editable option to True for all columns
                gd.configure_default_column(editable=True)
                #gd_results.configure_default_column(editable=True)
                # Configure the '🔧' column to use our the cell renderer 
                # and onCellClicked function
                gd.configure_column( field = '🔧', 
                                    onCellClicked = js_add_row,
                                    cellRenderer = cellRenderer_addButton
                                    )   
                gridoptions = gd.build()
                # AgGrid Table with Button Feature
                # Streamlit Form helps from rerunning on every widget-click
                # Also helps in providing layout
                with st.form('STD NMR Reduced Dataset Approach') as f:
                    #st.subheader(""" Complete the table below with your Ligand Proton Names and STD factors :writing_hand: """)
                    st.write(""" #### :blue[Complete the table below with your Ligand Proton Names and STD Factors (an example is shown in the table)] :writing_hand: """)
                # Inside the form, we are displaying an AgGrid table using the AgGrid function. 
                # The allow_unsafe_jscode parameter is set to True, 
                # which allows us to use JavaScript code in the AgGrid configuration
                # The theme parameter is set to 'balham', 
                # which applies the Balham theme to the table
                # The height parameter is set to 200, 
                # which specifies the height of the table in pixels.
                # The fit_columns_on_grid_load parameter is set to True, 
                # which ensures that the columns of the table are resized to fit 
                # the width of the table when it is first displayed
                    response = AgGrid(df,
                                    gridOptions = gridoptions, 
                                    editable=True,
                                    allow_unsafe_jscode = True, 
                                    theme = 'balham',
                                    height = 250
                                    )
                    st.write(" *Note: Don't forget to hit enter ↩ on new entry.*")
                    st.form_submit_button("Confirm item(s) 🔒", type="primary")
                # Visualize the AgGrid when submit button triggered           
                st.subheader("Results")
                # Fetch the data from the AgGrid Table
                res = response['data'] 
                res['STD at long saturation time (s)'] = res['STD at long saturation time (s)'].astype(float)
                res['STD at short saturation time (s)'] = res['STD at short saturation time (s)'].astype(float)
                res['ksat'] = (- np.log((res['STD at long saturation time (s)'] - res['STD at short saturation time (s)'])/res['STD at long saturation time (s)']))/tsat_short
                res['STD0'] = res.ksat*res['STD at long saturation time (s)']
                res['Epitope (%)'] = res.STD0/(np.max(res.STD0))*100
                #res.drop(['STD at short saturation time (s)', 'STD at long saturation time (s)'], inplace=True, axis=1)
                st.table(res)
                # Function
                @st.cache_data
                def convert_df(data2): 
                    "Converts the data to a CSV format"
                    return data2.to_csv(index=False).encode('utf-8')
                st.subheader("Download Results")
                col1,col2 = st.columns(2)
                csv = convert_df(response['data'])
                col1.write("Save locally")
                col2.download_button(
                "Press to Download Results 🗳️",
                csv,
                "file.csv",
                "text/csv",
                key='download-csv'
                )
            if uploaded_file1 is not None:
                st.write(df1)
        elif choice == 'Dissociation Constant Determination':
            st.write(""" ### Dissociation Constant Determination by STD NMR using the Reduced Dataset Approach from STD factors at 2 saturation times """)
            st.sidebar.header('CSV File Upload Option')
            # Collects user input features into dataframe
            uploaded_file1 = st.sidebar.file_uploader("**You can also upload a CSV File (without column headers) containing 3 columns, the first with the ligand concentrations (in µM), then the experimental STD factors at short (typically 0.5, 0.75 or 1 s; 0.75 s is recommended) and large (typically 6 or 8 s) saturation times**", type=["csv"])
            tsat_short1 = st.sidebar.number_input("**Enter the Short Saturation Time, in seconds**", value = 0.75, min_value=0.50, max_value=1.00)
            protconc1 = st.sidebar.number_input("**Enter the Total Protein Concentration, in µM**", value = 20)
            # Langmuir isotherm
            def model(x,Bmax,Kd):
                return (Bmax*x/(Kd+x))
            # Law of Mass Action
            def model2(x, b, Kd):
                return b*((x+protconc1+Kd)-np.sqrt(np.square(x+protconc1+Kd)-4*x*protconc1))/2
            if uploaded_file1 is not None:
                df1 = pd.read_csv(uploaded_file1, header = None)
                df1.columns=['Ligand Concentration (µM)', 'STD at short saturation time (s)', 'STD at long saturation time (s)']
                df1['ksat'] = (- np.log((df1['STD at long saturation time (s)'] - df1['STD at short saturation time (s)'])/df1['STD at long saturation time (s)']))/tsat_short1
                df1['STD0'] = df1.ksat*df1['STD at long saturation time (s)']
                df1['ratio'] = df1['Ligand Concentration (µM)']/protconc1
                df1['STD_AF0'] = df1['STD0']*df1['ratio']
                df2 = df1.copy()
                # Creating the STD build-up curve figure with all curves in one plot
                X1 = pd.DataFrame(df1,columns=["Ligand Concentration (µM)"])
                X1 = np.ravel(X1)
                Y1 = pd.DataFrame(df1,columns=["STD_AF0"])
                Y1 = np.ravel(Y1)
                # Curve Fit
                ans1, cov1 = curve_fit(model, X1, Y1, absolute_sigma=False, bounds=[[0,0], [5000,5000]])   
                stdev1 = np.sqrt(np.diag(cov1)) 
                
                #df1['Bmax'] = ans1[0]
                #df1['Bmax_stdev'] = stdev1[0]
                df1['Kd Langmuir'] = ans1[1]
                df1['Standard Deviation'] = stdev1[1]
                df1.drop(['Ligand Concentration (µM)','STD at short saturation time (s)', 'STD at long saturation time (s)', 'ksat', 'STD0', 'ratio', 'STD_AF0'], inplace=True, axis=1)
                df1.drop_duplicates(subset = ['Kd Langmuir'], ignore_index=True, inplace=True)
                #df2['Bmax'] = ans2[0]
                #df2['Bmax_stdev'] = stdev2[0]
                # Function
                @st.cache_data
                def convert_df1(df1): 
                    "Converts the data to a CSV format"
                    return df1.to_csv(index=False).encode('utf-8')
                st.subheader("Download Results from the Langmuir Isotherm")
                col1,col2 = st.columns(2)
                csv1 = convert_df1(df1)
                #col1.write("Save File")
                col2.download_button("Press to Download the Results using the Langmuir Isotherm 🗳️", csv1, "Kd_Langmuir.csv", "text/csv", key='download-csv1')
                st.write(df1)

                ans2, cov2 = curve_fit(model2, X1, Y1, absolute_sigma=False, bounds=[[0,0], [5000,5000]]) 
                stdev2 = np.sqrt(np.diag(cov2)) 
                df2['Kd Law of Mass'] = ans2[1]
                df2['Standard Deviation'] = stdev2[1]
                df2.drop(['Ligand Concentration (µM)','STD at short saturation time (s)', 'STD at long saturation time (s)', 'ksat', 'STD0', 'ratio', 'STD_AF0'], inplace=True, axis=1)
                df2.drop_duplicates(subset = ['Kd Law of Mass'], ignore_index=True, inplace=True)
                @st.cache_data
                def convert_df2(df2): 
                    "Converts the data to a CSV format"
                    return df2.to_csv(index=False).encode('utf-8')
                st.subheader("Download Results from the Law of Mass Action")
                col3,col4 = st.columns(2)
                csv2 = convert_df2(df2)
                #col3.write("Save")
                col4.download_button("Press to Download the Results using the Law of Mass Action 🗳️", csv2, "Kd_MassLaw.csv", "text/csv", key='download-csv2')
                st.write(df2)
            else:
                # Create an AgGrid table from a pandas DataFrame
                tsat_short_label = st.write("##### Enter below the Short Saturation Time employed, in seconds")
                tsat_short = st.number_input("**Enter the Short Saturation Time employed, in seconds**", value = 0.75, min_value=0.50, max_value=1.00, label_visibility='hidden')
                protconc_label = st.write("##### Enter the Total Protein Concentration, in µM")
                protconc = st.number_input("**Enter the Total Protein Concentration (in µM)**", value = 20, label_visibility='hidden')
                d = {'Ligand Concentration (µM)': [150,300,600,1500,2500,4000], 'STD at short saturation time (s)': [37,27,16,9,6,4],'STD at long saturation time (s)': [59,50,39,26,19,14]}
                df = pd.DataFrame(data = d)
                # Display the Dataframe in AgGrid
                # JavaScript function to add a new row to the AgGrid table
                js_add_row = JsCode("""
                function(e) {
                    let api = e.api;
                    let rowPos = e.rowIndex + 1; 
                    api.applyTransaction({addIndex: rowPos, add: [{}]})    
                };
                """     
                )
                # Cell renderer for the '🔧' column to render a button
                cellRenderer_addButton = JsCode('''
                    class BtnCellRenderer {
                        init(params) {
                            this.params = params;
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
                                }
                                </style>
                                <button id='click-button' 
                                    class="btn_add" 
                                    >&#x2193; Add</button>
                            </span>
                        `;
                        }
                        getGui() {
                            return this.eGui;
                        }
                    };
                    ''')
                # Create a GridOptionsBuilder object from our DataFrame
                gd = GridOptionsBuilder.from_dataframe(df)
                # Configure the default column to be editable
                # sets the editable option to True for all columns
                gd.configure_default_column(editable=True)
                # Configure the '🔧' column to use our the cell renderer 
                # and onCellClicked function
                gd.configure_column( field = '🔧', 
                                    onCellClicked = js_add_row,
                                    cellRenderer = cellRenderer_addButton
                                    )
                gridoptions = gd.build()
                # AgGrid Table with Button Feature
                # Streamlit Form helps from rerunning on every widget-click
                # Also helps in providing layout
                with st.form('STD NMR Reduced Dataset Approach') as f:
                    #st.subheader(""" Complete the table below with your Ligand Proton Names and STD factors :writing_hand: """)
                    st.write(""" #### :blue[Complete the table below with the Ligand Proton Concentrations employed (in µM) and the STD Factors (an example is shown in the table)] :writing_hand: """)
                # Inside the form, we are displaying an AgGrid table using the AgGrid function. 
                # The allow_unsafe_jscode parameter is set to True, 
                # which allows us to use JavaScript code in the AgGrid configuration
                # The theme parameter is set to 'balham', 
                # which applies the Balham theme to the table
                # The height parameter is set to 200, 
                # which specifies the height of the table in pixels.
                # The fit_columns_on_grid_load parameter is set to True, 
                # which ensures that the columns of the table are resized to fit 
                # the width of the table when it is first displayed
                    response = AgGrid(df,
                                    gridOptions = gridoptions, 
                                    editable=True,
                                    allow_unsafe_jscode = True, 
                                    theme = 'balham',
                                    height = 200
                                    )
                    st.write(" *Note: Don't forget to hit enter ↩ on new entry.*")
                    st.form_submit_button("Confirm item(s) 🔒", type="primary")
                # Visualize the AgGrid when submit button triggered           
                #st.subheader("Results")
                # Fetch the data from the AgGrid Table
                res = response['data'] 
                res['STD at long saturation time (s)'] = res['STD at long saturation time (s)'].astype(float)
                res['STD at short saturation time (s)'] = res['STD at short saturation time (s)'].astype(float)
                res['ksat'] = (- np.log((res['STD at long saturation time (s)'] - res['STD at short saturation time (s)'])/res['STD at long saturation time (s)']))/tsat_short
                res['STD0'] = res.ksat*res['STD at long saturation time (s)']
                res['ratio'] = res['Ligand Concentration (µM)']/protconc
                res['STD_AF0'] = res['STD0']*res['ratio']
                # Creating the STD build-up curve figure with all curves in one plot
                X = pd.DataFrame(res,columns=["Ligand Concentration (µM)"])
                X = np.ravel(X)
                Y = pd.DataFrame(res,columns=["STD_AF0"])
                Y = np.ravel(Y)
                # Curve Fit 
                ans1, cov1 = curve_fit(model, X, Y, absolute_sigma=False, bounds=[[0,0], [5000,5000]])  
                stdev1 = np.sqrt(np.diag(cov1)) 
                res.drop(['Ligand Concentration (µM)','STD at short saturation time (s)', 'STD at long saturation time (s)', 'ksat', 'STD0', 'ratio', 'STD_AF0'], inplace=True, axis=1)
                #res['Bmax'] = ans[0]
                #res['Bmax_stdev'] = stdev[0]
                res['Kd Langmuir'] = ans1[1]
                res['Standard Deviation'] = stdev1[1]
                res.drop_duplicates(subset = ['Kd Langmuir'], ignore_index=True, inplace=True)
                #st.table(res)
                # Function
                @st.cache_data
                def convert_df1(res): 
                    "Converts the data to a CSV format"
                    return res.to_csv(index=False).encode('utf-8')
                st.subheader("Download Results from the Langmuir Isotherm")
                col1,col2 = st.columns(2)
                csv1 = convert_df1(res)
                #col1.write("Save File")
                col2.download_button("Press to Download the Results using the Langmuir Isotherm 🗳️", csv1, "Kd_Langmuir.csv", "text/csv", key='download-csv1i')
                st.write(res)

                ans2, cov2 = curve_fit(model2, X, Y, absolute_sigma=False, bounds=[[0,0], [5000,5000]]) 
                stdev2 = np.sqrt(np.diag(cov2)) 
                d = {'Kd Law of Mass': [ans2[1]],'Standard Deviation': [stdev2[1]]}
                res2 = pd.DataFrame(data=d)
                #df2['Kd Law of Mass'] = ans2[1]
                #df2['Standard Deviation'] = stdev2[1]
                #res2.drop(['Ligand Concentration (µM)','STD at short saturation time (s)', 'STD at long saturation time (s)', 'ksat', 'STD0', 'ratio', 'STD_AF0'], inplace=True, axis=1)
                res2.drop_duplicates(subset = ['Kd Law of Mass'], ignore_index=True, inplace=True)
                @st.cache_data
                def convert_df2(res2): 
                    "Converts the data to a CSV format"
                    return res2.to_csv(index=False).encode('utf-8')
                st.subheader("Download Results from the Law of Mass Action")
                col3,col4 = st.columns(2)
                csv2 = convert_df2(res2)
                #col3.write("Save")
                col4.download_button("Press to Download the Results using the Law of Mass Action 🗳️", csv2, "Kd_MassLaw.csv", "text/csv", key='download-csv2i')
                st.write(res2)
            #if uploaded_file1 is not None:
            #    st.write(df1)

    if __name__ == '__main__':
        main_menu()

elif authentication_status is False:
    st.error('Username/password is incorrect')
elif authentication_status is None:
    st.warning('Please enter your username and password')