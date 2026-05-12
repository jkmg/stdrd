import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go

# --- MATH MODELS ---
def model_langmuir(x, Bmax, Kd):
    return (Bmax * x / (Kd + x))

def model_mass_action(x, b, Kd, protconc):
    return b * ((x + protconc + Kd) - np.sqrt(np.square(x + protconc + Kd) - 4 * x * protconc)) / 2

# --- PLOTTING ---
def create_epitope_buildup_plot(df, t_short):
    fig = go.Figure()
    t_max_plot = 6.5
    t_smooth = np.linspace(0, t_max_plot, 100)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, (_, row) in enumerate(df.iterrows()):
        # Use .get() or check existence to avoid errors with empty rows
        proton = str(row.get('Proton Name', f'H{i}'))
        std_long = row.get('STD long', 0)
        ksat = row.get('ksat', 0)
        
        color = colors[i % len(colors)]
        y_smooth = std_long * (1 - np.exp(-ksat * t_smooth))
        fig.add_trace(go.Scatter(x=t_smooth, y=y_smooth, mode='lines', name=proton, line=dict(color=color)))
        
    fig.update_layout(title="STD Build-up Curves", template="plotly_white", xaxis_title="Time (s)", yaxis_title="STD Factor")
    return fig

# --- MAIN APP ---
def main_menu():
    st.set_page_config(page_title="STD NMR App", layout="wide")
    
    st.sidebar.title("🔬 STD NMR Tools")
    choice = st.sidebar.selectbox('Analysis Mode', ['Binding Epitope', 'Kd Determination'])
    tsat_short = st.sidebar.number_input("Short Saturation Time (s)", value=0.75, step=0.05)

    if choice == 'Binding Epitope':
        st.header("Binding Epitope Determination")
        st.info("💡 **Tip:** You can add rows by clicking the '**+**' at the bottom of the table or delete them by selecting and pressing 'Backpace'.")

        # Initial Data
        if 'df_epitope' not in st.session_state:
            st.session_state.df_epitope = pd.DataFrame([
                {'Proton Name': "H1", 'STD short': 1.7, 'STD long': 4.8},
                {'Proton Name': "H2", 'STD short': 4.3, 'STD long': 14.3},
                {'Proton Name': "H3", 'STD short': 4.8, 'STD long': 12.2}
            ])

        # NATIVE DATA EDITOR (Replaces AgGrid)
        # num_rows="dynamic" handles the adding/deleting automatically
        edited_df = st.data_editor(
            st.session_state.df_epitope,
            num_rows="dynamic",
            use_container_width=True,
            key="epitope_editor"
        )

        if st.button("🚀 Run Analysis", type="primary"):
            try:
                df = edited_df.copy()
                df[['STD short', 'STD long']] = df[['STD short', 'STD long']].apply(pd.to_numeric, errors='coerce')
                df = df.dropna()

                # Core Calculations
                df['ksat'] = (-np.log((df['STD long'] - df['STD short']) / df['STD long'])) / tsat_short
                df['STD0'] = df['ksat'] * df['STD long']
                df['Epitope (%)'] = (df['STD0'] / df['STD0'].max()) * 100
                
                st.subheader("Results")
                st.dataframe(df.style.highlight_max(axis=0, subset=['Epitope (%)'], color='#d4edda'))
                st.plotly_chart(create_epitope_buildup_plot(df, tsat_short), use_container_width=True)
                
                # Save to session state so it persists
                st.session_state.df_epitope = edited_df
            except Exception as e:
                st.error("Ensure 'STD long' > 'STD short' and all numeric fields are filled.")

    elif choice == 'Kd Determination':
        st.header("Dissociation Constant (Kd)")
        protconc = st.sidebar.number_input("Protein Conc. (µM)", value=20.0)

        if 'df_kd' not in st.session_state:
            st.session_state.df_kd = pd.DataFrame([
                {'Ligand Conc (µM)': 150.0, 'STD short': 37.0, 'STD long': 59.0},
                {'Ligand Conc (µM)': 300.0, 'STD short': 27.0, 'STD long': 50.0},
                {'Ligand Conc (µM)': 600.0, 'STD short': 16.0, 'STD long': 39.0}
            ])

        edited_kd = st.data_editor(
            st.session_state.df_kd,
            num_rows="dynamic",
            use_container_width=True,
            key="kd_editor"
        )

        if st.button("🚀 Calculate Kd", type="primary"):
            try:
                df = edited_kd.copy()
                df = df.apply(pd.to_numeric, errors='coerce').dropna()
                
                df['ksat'] = (-np.log((df['STD long'] - df['STD short']) / df['STD long'])) / tsat_short
                df['STD0'] = df['ksat'] * df['STD long']
                df['STD_AF0'] = df['STD0'] * (df['Ligand Conc (µM)'] / protconc)
                
                X, Y = df['Ligand Conc (µM)'].values, df['STD_AF0'].values
                popt, _ = curve_fit(model_langmuir, X, Y, bounds=(0, 5000))
                
                st.metric("Calculated Kd", f"{popt[1]:.2f} µM")
                
                # Plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=X, y=Y, mode='markers+text', name='Data', text=df.index))
                x_fit = np.linspace(0, max(X)*1.2, 100)
                fig.add_trace(go.Scatter(x=x_fit, y=model_langmuir(x_fit, *popt), name='Fit (Langmuir)', line=dict(dash='dash')))
                st.plotly_chart(fig, use_container_width=True)
                
                st.session_state.df_kd = edited_kd
            except:
                st.error("Calculation failed. Check for valid numbers and enough data points.")

if __name__ == '__main__':
    main_menu()