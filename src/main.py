import streamlit as st
import st_aggrid
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
from st_aggrid.shared import GridUpdateMode
import pandas as pd
import numpy as np

st.write(""" ## Determination of STD NMR Binding Epitopes using the Reduced Dataset Approach from STD factors at 2 saturation times """)

st.sidebar.header('CSV File Upload Option')

# Collects user input features into dataframe
uploaded_file1 = st.sidebar.file_uploader("**You can also upload a CSV File (without column headers) containing 3 columns, the first with the proton names, then the experimental STD factors at short (typically 0.5, 0.75 or 1 s; 0.75 s is recommended) and large (typically 6 or 8 s) saturation times**", type=["csv"])
tsat_short1 = st.sidebar.number_input("**Enter the Short Saturation Time, in seconds**", value = 0.75, min_value=0.50, max_value=1.00)
if uploaded_file1 is not None:
    df1 = pd.read_csv(uploaded_file1, header = None)
    df1.columns=['Proton', 'STD_short_tsat', 'STD_long_tsat']
    df1['ksat'] = (- np.log((df1.STD_long_tsat - df1.STD_short_tsat)/df1.STD_long_tsat))/tsat_short1
    df1['STD0'] = df1.ksat*df1.STD_long_tsat
    df1['Epitope'] = df1.STD0/(np.max(df1.STD0))*100
    df1.drop(['STD_short_tsat', 'STD_long_tsat'], inplace=True, axis=1)
    # Function
    @st.experimental_memo
    def convert_df1(data1): 
        "Converts the data to a CSV format"
        return data1.to_csv(index=False).encode('utf-8')

    st.subheader("Download Results")
    col1,col2 = st.columns(2)
    csv = convert_df1(df1)
    col1.write("Save locally")
    col1.download_button(
    "Press to Download 🗳️",
    csv,
    "STD0_Reduced_Dataset.csv",
    "text/csv",
    key='download-csv'
    )

else:
    # Create an AgGrid table from a pandas DataFrame
    tsat_short = st.number_input("**Enter the Short Saturation Time Employed, in seconds**", value = 0.75, min_value=0.50, max_value=1.00)
    d = {'Proton_Name': [""],'STD_short_tsat': [np.nan],'STD_long_tsat': [np.nan]}
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
    #gd_results.configure_column( field = 'Results', 
    #                    onCellClicked = js_add_row
    #                    )
    gridoptions = gd.build()
    #gridoptions = gd_results.build()

    # AgGrid Table with Button Feature
    # Streamlit Form helps from rerunning on every widget-click
    # Also helps in providing layout

    with st.form('STD NMR Reduced Dataset Approach') as f:
        st.subheader(""" Complete the table below with your Ligand Proton Names and STD factors :writing_hand: """)
        
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
                        height = 200,
                        fit_columns_on_grid_load = True)
        st.write(" *Note: Don't forget to hit enter ↩ on new entry.*")
        st.form_submit_button("Confirm item(s) 🔒", type="primary")

    # Visualize the AgGrid when submit button triggered           
    st.subheader("Updated Table")
    # Fetch the data from the AgGrid Table
    res = response['data'] 
    res['STD_long_tsat'] = res['STD_long_tsat'].astype(float)
    res['STD_short_tsat'] = res['STD_short_tsat'].astype(float)
    res['ksat'] = (- np.log((res.STD_long_tsat - res.STD_short_tsat)/res.STD_long_tsat))/tsat_short
    res['STD0'] = res.ksat*res.STD_long_tsat
    res['Epitope'] = res.STD0/(np.max(res.STD0))*100
    #res.drop(['STD_short_tsat', 'STD_long_tsat'], inplace=True, axis=1)
    st.table(res)

    # Function
    @st.experimental_memo
    def convert_df(data2): 
        "Converts the data to a CSV format"
        return data2.to_csv(index=False).encode('utf-8')

    st.subheader("Download Results")
    col1,col2 = st.columns(2)
    csv = convert_df(response['data'])
    col1.write("Save locally")
    col1.download_button(
    "Press to Download Results 🗳️",
    csv,
    "file.csv",
    "text/csv",
    key='download-csv'
    )

if uploaded_file1 is not None:
    st.write(df1)
else:
    st.write('Awaiting CSV file to be uploaded')