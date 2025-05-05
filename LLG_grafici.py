# Marginalità LLG
# env neuraplprophet conda


import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
import warnings
import matplotlib.pyplot as plt
import plotly.express as px
warnings.filterwarnings('ignore')
import plotly.graph_objects as go


####### Impaginazione

st.set_page_config(layout="wide")

url_immagine = 'sae-scientifica-logo-web-v2.png'#?raw=true' #LOGO-Artigrafiche_Italia.png

col_1, col_2 = st.columns([2, 3])

with col_1:
    st.image(url_immagine, width=300)

with col_2:
    st.title('Analisi marginalità LLG')

st.subheader('Caricamento dati | file ordinato', divider='green')

####### Caricamento dati

uploaded_ordinato = st.file_uploader('caricare file ordinato')
if not uploaded_ordinato:
    st.stop()


colonne_ordinato = ['Codice anagrafica','Cliente/Fornitore','Codice articolo','Descrizione Articolo','Quantità',
                    'Prezzo finale','Cliente - Fornitore','Data consegna','Ricavi','Costi','MACROPROGETTO',
                    'Chiave di registrazione documento','Numero Registrazione','Evasione totale documento',
                    'Quantità trasformata','Stato evasione riga','Anno data consegna','Mese data consegna']


@st.cache_data
def carica_file(file, colonne):
    try:
        df = pd.read_excel(file, usecols=colonne)
        return df
    except Exception as e:
        st.error(f"Errore durante il caricamento del file: {e}")
        return None


ordinato_raw = carica_file(uploaded_ordinato, colonne_ordinato)


st.write('File caricato:')
st.dataframe(ordinato_raw)


ordinato_raw = ordinato_raw[ordinato_raw['Anno data consegna'] == 2025]

articoli_LLG = ordinato_raw[ordinato_raw['Codice anagrafica']== 9067]['Codice articolo'].unique().tolist()

# Pulisce ogni elemento rimuovendo spazi iniziali e finali
articoli_LLG = [art.strip() for art in articoli_LLG]

# Rimuove gli elementi specifici
articoli_LLG = [art for art in articoli_LLG if art not in ['SP TRASP', 'F00001','T00001']]

st.write('Numero di articoli LLG:', len(articoli_LLG))

ordinato_raw['Codice articolo'] = ordinato_raw['Codice articolo'].astype(str)
# Rimuove gli spazi iniziali e finali da ogni elemento della colonna 'Codice articolo'
ordinato_raw['Codice articolo'] = ordinato_raw['Codice articolo'].str.strip()
ordinato_raw = ordinato_raw[ordinato_raw['Codice articolo'].isin(articoli_LLG)]

pivot_ordinato = ordinato_raw.pivot_table(index=['Codice articolo','Cliente - Fornitore', 'Cliente/Fornitore'], 
                                          columns=['Mese data consegna'], values=['Prezzo finale','Quantità'], aggfunc='sum').reset_index()

pivot_ordinato_summary = ordinato_raw.pivot_table(index=['Codice articolo'], 
                                                  columns=['Cliente - Fornitore'], 
                                                  values=['Prezzo finale','Quantità'], 
                                                  aggfunc='sum').reset_index()

#st.write('Pivot ordinato_summary:')
#st.dataframe(pivot_ordinato_summary)

# Flatten the multi-level columns
pivot_flatten = pivot_ordinato_summary.copy()
pivot_flatten.columns = ['_'.join(map(str, col)).strip() for col in pivot_flatten.columns.values]
pivot_flatten = pivot_flatten.rename(columns={'Codice articolo_': 'Codice articolo'})
pivot_flatten = pivot_flatten.rename(columns={'Prezzo finale_': 'Prezzo finale', 'Quantità_': 'Quantità'})
pivot_flatten = pivot_flatten.rename(columns={'Cliente - Fornitore_': 'Cliente - Fornitore'})
pivot_flatten['marginalità'] = pivot_flatten['Prezzo finale_Cliente'] / pivot_flatten['Prezzo finale_Fornitore'] - 1
pivot_flatten['marginalità'] = pivot_flatten['marginalità'].round(3)
pivot_flatten['marginalità'] = pivot_flatten['marginalità'].apply(lambda x: f"{x:.1%}")
marginalita_totale = pivot_flatten['Prezzo finale_Cliente'].sum() / pivot_flatten['Prezzo finale_Fornitore'].sum() - 1
marginalita_totale = marginalita_totale.round(3)
marginalita_totale = f"{marginalita_totale:.1%}"

st.subheader('Ordinato LLG per mese', divider='green')
st.dataframe(pivot_ordinato)


st.subheader('Calcolo marginalità consegnato 2025 - YTD', divider='green')
st.write('Marginalità totale:', marginalita_totale)

# Calcola min e max per definire i bin
min_val = pivot_flatten['marginalità'].min()
max_val = pivot_flatten['marginalità'].max()

# Crea istogramma con bin di larghezza 25
fig = px.histogram(
    pivot_flatten,
    x='marginalità',
    title="Distribuzione della Marginalità",
    labels={'marginalità': 'Marginalità [%]'},
    color_discrete_sequence=['green'],
    height=800,
)
fig.update_traces(xbins=dict(
    start=min_val,
    end=max_val,
    size=25
))

# Crea scatter plot
plot_data = pivot_flatten[['Codice articolo','marginalità', 'Prezzo finale_Cliente', 'Quantità_Cliente']].dropna()
fig_scatter = px.scatter(
    plot_data,
    x='marginalità',
    y='Prezzo finale_Cliente',
    size='Quantità_Cliente',
    color_discrete_sequence=['green'],
    title="Marginalità vs Prezzo Finale Cliente | dimensione = Quantità Cliente",
    hover_name='marginalità',
    hover_data=['Prezzo finale_Cliente', 'Quantità_Cliente', 'Codice articolo'],
    height=800,
    labels={
        'marginalità': 'Marginalità [%]',
        'Prezzo finale_Cliente': 'Prezzo Finale Cliente',
        'Quantità_Cliente': 'Quantità Cliente',
        'Codice articolo': 'Codice articolo'
    }
)

# Visualizza i grafici
col_3, col_4 = st.columns([1, 1])

with col_3:
    st.plotly_chart(fig, use_container_width=True)

with col_4:
    st.plotly_chart(fig_scatter, use_container_width=True)


# tabella marginalità con codice e descrizione
pivot_ordinato_descrizione = ordinato_raw.pivot_table(index=['Codice articolo', 'Descrizione Articolo'], 
                                                  columns=['Cliente - Fornitore'], 
                                                  values=['Prezzo finale','Quantità'], 
                                                  aggfunc='sum').reset_index()


pivot_descrizione_flatten = pivot_ordinato_descrizione.copy()
pivot_descrizione_flatten.columns = ['_'.join(map(str, col)).strip() for col in pivot_descrizione_flatten.columns.values]
pivot_descrizione_flatten = pivot_descrizione_flatten.rename(columns={'Codice articolo_': 'Codice articolo'})
pivot_descrizione_flatten = pivot_descrizione_flatten.rename(columns={'Descrizione Articolo_': 'Descrizione Articolo'})
pivot_descrizione_flatten = pivot_descrizione_flatten.rename(columns={'Prezzo finale_': 'Prezzo finale', 'Quantità_': 'Quantità'})
pivot_descrizione_flatten = pivot_descrizione_flatten.rename(columns={'Cliente - Fornitore_': 'Cliente - Fornitore'})
pivot_descrizione_flatten['marginalità'] = pivot_descrizione_flatten['Prezzo finale_Cliente'] / pivot_descrizione_flatten['Prezzo finale_Fornitore'] - 1
pivot_descrizione_flatten['marginalità'] = pivot_descrizione_flatten['marginalità'].round(3)
pivot_descrizione_flatten['marginalità'] = pivot_descrizione_flatten['marginalità'].apply(lambda x: f"{x:.1%}")

st.write('Marginalità per articolo e descrizione:')
st.dataframe(pivot_descrizione_flatten)


# Pareto per cliente
st.subheader('Pareto valore acquistato per Cliente', divider='green')
# pivot per cliente
pivot_ordinato_cliente = ordinato_raw.pivot_table(index=['Codice articolo', 'Cliente - Fornitore', 'Cliente/Fornitore'],  
                                                  values=['Prezzo finale'], 
                                                  aggfunc='sum').reset_index()
pivot_ordinato_cliente = pivot_ordinato_cliente[pivot_ordinato_cliente['Cliente - Fornitore']== 'Cliente']

#st.write('Pivot ordinato per cliente:')
#st.dataframe(pivot_ordinato_cliente)

pareto_cliente = pivot_ordinato_cliente.groupby('Cliente/Fornitore')['Prezzo finale'].sum().reset_index()
pareto_cliente = pareto_cliente.sort_values(by='Prezzo finale', ascending=False)

#st.write('Pareto cliente:')
#st.dataframe(pareto_cliente)

df_pareto = pareto_cliente.copy()

# Calcola percentuale cumulativa
df_pareto['Cumulata'] = df_pareto['Prezzo finale'].cumsum()
df_pareto['Cumulata (%)'] = 100 * df_pareto['Cumulata'] / df_pareto['Prezzo finale'].sum()

# Grafico a barre con linea cumulativa
fig_pareto = px.bar(df_pareto, x='Cliente/Fornitore', y='Prezzo finale', labels={'Prezzo finale': 'Acquistato YTD'}, title='Diagramma di Pareto',
                    height=800, color_discrete_sequence=['green'])
fig_pareto.add_scatter(x=df_pareto['Cliente/Fornitore'], y=df_pareto['Cumulata (%)'],
                mode='lines+markers', name='Cumulata (%)', yaxis='y2',
                line=dict(color='red', width=2))

# Aggiunta asse secondario per percentuale
fig_pareto.update_layout(
    yaxis2=dict(
        title='Percentuale Cumulata',
        overlaying='y',
        side='right',
        range=[0, 110]
    ),
    xaxis_tickangle=45,
    hovermode='x unified'
)

# Mostra il grafico in Streamlit
st.plotly_chart(fig_pareto, use_container_width=True)

# opzione per scaricare file

def to_excel_bytes(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Foglio1') #index=False,
    return output.getvalue()

# Crea il bottone per scaricare il pivot ordinato
file_pivot_ordinato = to_excel_bytes(pivot_ordinato)
st.download_button(
    label="📥 Scarica file tabella ordinato",
    data=file_pivot_ordinato,
    file_name='file_pivot_ordinato.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)

st.stop()

# vedi pivot MaGa per ricavare marginalità fornitore 
# estensione a tutti i fornitori (grafico a mosaico o pareto)
# report finale per tutti i fornitori
# eventuale selezione del periodo
