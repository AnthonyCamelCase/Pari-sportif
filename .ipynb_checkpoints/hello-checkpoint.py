import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import bokeh

st.title('Hello')
st.header('hello world')
st.markdown('Bonjour Adrien')


#st.image("uneimage.png", width=250)
st.title('Demo Streamlit')

st.markdown("Objectif : prédire la classe d'un **pinguins**")

with st.echo():
    df = pd.read_csv('atp_data.csv')

if st.checkbox("Affichier les données"):
    st.dataframe(df)

st.sidebar.header("colonnes du dataframe")
st.sidebar.write(df.columns)


st.markdown("Affichons les valeurs manquantes")
st.write(df.isna().sum())



fig = plt.figure(figsize=(15, 15))
plt.subplot(121)
df.Surface.value_counts().plot.pie(
    autopct='%1.1f%%', labeldistance=0.7, pctdistance=0.5)
plt.title('Nombre de match  par type de surface')

plt.subplot(122)
df.Court.value_counts().plot.pie(
    autopct='%1.1f%%', labeldistance=0.7, pctdistance=0.5)
plt.title('Nombre de match  indoor et outdoor')

st.pyplot(fig)


#*****    variable WRank     *****
p1 = figure(title="Distribution de la variable WRank", y_range=[0, 7200])
p1.xaxis.axis_label = "rang mondial du joueur"
p1.yaxis.axis_label = "nombre de match gagné"
hist, bins_edges = np.histogram(a=df["WRank"], bins=200)
source = ColumnDataSource({'hist': hist, 'x': bins_edges[:-1]})
p1.vbar(top='hist', x='x', width=bins_edges[1]-bins_edges[0], source=source)
hover = HoverTool(tooltips=[("rang du joueur", "@x"),
                            ("nombre de match gagné", "@hist")])
p1.add_tools(hover)

#*****    variable LRank     *****
p2 = figure(title="Distribution de la variable LRank", y_range=[0, 7200])
p2.axis.axis_label = "rang mondial du joueur"
p2.yaxis.axis_label = "nombre de match gagné"
hist, bins_edges = np.histogram(a=df["LRank"], bins=200)
source = ColumnDataSource({'hist': hist, 'x': bins_edges[:-1]})
p2.vbar(top='hist', x='x', width=bins_edges[1]-bins_edges[0], source=source)
hover = HoverTool(tooltips=[("LRank", "@x"), ("match", "@hist")])
p2.add_tools(hover)

#*****    variable elo_winner    *****
p3 = figure(title="Distribution de la variable elo_winner",
            x_range=[1300, 2400], y_range=[0, 3000])
p3.xaxis.axis_label = "nombre de points elo"
p3.yaxis.axis_label = "nombre de match gagné"
hist, bins_edges = np.histogram(a=df["elo_winner"], bins=100)
source = ColumnDataSource({'hist': hist, 'x': bins_edges[:-1]})
p3.vbar(top='hist', x='x', width=bins_edges[1]-bins_edges[0], source=source)
hover = HoverTool(tooltips=[("elo winner", "@x"), ("match", "@hist")])
p3.add_tools(hover)

#*****    variable elo_loser     *****
p4 = figure(title="Distribution de la variable elo_loser",
            x_range=[1300, 2400], y_range=[0, 3000])
p4.xaxis.axis_label = "nombre de points elo"
p4.yaxis.axis_label = "nombre de match gagné"
hist, bins_edges = np.histogram(a=df["elo_loser"], bins=100)
source = ColumnDataSource({'hist': hist, 'x': bins_edges[:-1]})
p4.vbar(top='hist', x='x', width=bins_edges[1]-bins_edges[0], source=source)
hover = HoverTool(tooltips=[("elo loser", "@x"), ("match", "@hist")])
p4.add_tools(hover)

#*****    variable elo_loser     *****
p5 = figure(title="Distribution de la variable proba_elo")
p5.xaxis.axis_label = "probabilité de victoire qu'avait le vainqueur selon son elo"
p5.yaxis.axis_label = "nombre de match"
hist, bins_edges = np.histogram(a=df["proba_elo"], bins=100)
source = ColumnDataSource({'hist': hist, 'x': bins_edges[:-1]})
p5.vbar(top='hist', x='x', width=bins_edges[1]-bins_edges[0], source=source)
hover = HoverTool(tooltips=[("chance de victoire", "@x"), ("match", "@hist")])
p5.add_tools(hover)

#*****   affichage des graphiques    *****
tab1 = Panel(child=p1, title="WRank")
tab2 = Panel(child=p2, title="LRank")
tab3 = Panel(child=p3, title="elo_winner")
tab4 = Panel(child=p4, title="elo_loser")
tab5 = Panel(child=p5, title="proba_elo")

tabs = Tabs(tabs=[tab1, tab2, tab3, tab4, tab5])

st.bokeh_chart(tabs)
