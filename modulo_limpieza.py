import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import stylecloud
from PIL import Image
import unicodedata
import re
from unidecode import unidecode
from nltk.corpus import stopwords
from nltk import FreqDist
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


##### Renombrar Variables
def renombrar_variables(lista_variables, tipo_variables):
    """ renombrar_variables(lista_variables, tipo_variables) -> {variable_original : variable con prefijo}
    Recibe una lista con el nombre de las variables de un sólo tipo
    y regresa un diccionario que tiene como llave el nombre de la 
    variable original y como valor el nombre de la variable con el 
    prefijo del tipo de variable indicada
    
    tipo_variables: 'continua', 'discreta', 'fecha', 'texto'
    """
    try:
        if type(lista_variables) != list:
            print('Se debe introducir una lista no vacia con el nombre de las variables')
        else:
            if tipo_variables not in ['continua', 'discreta', 'fecha', 'texto']:
                print('tipo_variable debe se alguna de las siguientes opciones \n "continua", "discreta", "fecha", "texto"')
            else:
                # Se elige el tipo de variable
                if tipo_variables == 'continua':
                    pref = 'c'
                elif tipo_variables == 'discreta':
                    pref = 'v'
                elif tipo_variables == 'fecha':
                    pref = 'd'
                elif tipo_variables == 'texto':
                    pref = 't'
    
                # Se junta el prefijo de la variable con espacio, luego los espacios se cambian por _
                nuevas = map(lambda x: (' '.join((pref , x))).replace(' ', '_'), lista_variables)
                
                #Diccionario final
                return dict(zip(lista_variables, nuevas))
            
    except:
        print('Se debe introducir una lista con valores no repetidos')

###### Limpieza de Texto

def limp_texto(Serie):
    """ limp_texto(Serie) -> pd.Series
    Recibe una serie de pandas con una columna de texto en su interior y
    regresa una serie de texto limpia en minúsculas y sin números en su interior.
    """
    # Lista de texto limpia
    clean_t = []

    # entrar en cada texto y eliminar mayusculas o caracteres especiales
    for text in Serie:
        aux1 = unicodedata.normalize('NFD', text).encode('ascii', 'ignore')
        aux2 = re.sub("[^a-zA-Z0-9 ]", " ", aux1.decode("utf-8"), flags=re.UNICODE)
        aux3 = u' '.join(aux2.lower().split())
        clean_t.append(aux3)
    
    return clean_t

# Grafica de Pie
def Pie(labels, values, title, path):
    """ Pie(lista de etiquetas, lista de valores, titulo de grafica, path salvar) -> Grafica
    Recibe todos los parametros de una gráfica, su título y path para salvar la
    grafica y regresa una gráfica de Pie. Si el path es False no guradrá la 
    gráfica.
    """
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent')])
    fig.update_layout(title_text=title)
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    if path != False:
        fig.write_image(path, scale = 1.5)
    fig.show()
    
# grafica de barras
def Bar(etiquetas, valores, titulo, path, color = 'blue'):
    """ Bar(lista de etiquetas, lista de valores, titulo de grafica, path salvar, color) -> Grafica
    Recibe todos los parametros de una gráfica, su título y path para salvar la
    grafica y regresa una gráfica de Barras. Si el path es False no guradrá la 
    gráfica. El color puede recibir una lista.
    """
    fig = go.Figure(data=[go.Bar(x=etiquetas, y=valores, text=valores, marker_color=color)])
    fig.update_layout(title_text=titulo)
    fig.update_traces(textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    if path != False:
        fig.write_image(path, scale = 1.5)
    fig.show()
    
# grafica de lineas
def Line(etiquetas, valores, titulo, path, color = 'blue'):
    """ Line(lista de etiquetas, lista de valores, titulo de grafica, path salvar, color) -> Grafica
    Recibe todos los parametros de una gráfica, su título y path para salvar la
    grafica y regresa una gráfica de Lineas. Si el path es False no guradrá la 
    gráfica. El color puede recibir una lista.
    """
    fig = go.Figure(data=[go.Scatter(x=etiquetas, y=valores, text=valores, marker_color=color)])
    fig.update_layout(title_text=titulo)
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    if path != False:
        fig.write_image(path, scale = 1.5)
    fig.show()


    
### Creacion de nube de palabras
def word_cloud(texto, path_texto, path_imagen):
    """
    Parameters
    ----------
    texto : Lista o Serie
        Serie de pandas con textos a crear
    path_texto : cadena de texto
        Path para guardar el txt generado
    path_imagen : TYPE
        Path para salvar la imagen
    Returns
    -------
    TYPE
        Recibe una lista con sus path para guardar los textos e imagenes y regresa
        un wordcloud como imagen.
    """
    # CReacion de .txt
    with open(path_texto,"w") as f:
        for text in texto:
            f.write(str(text + ' '))
    f.close()

    # paths
    path_texto=path_texto
    path_imagen=path_imagen

    stylecloud.gen_stylecloud(file_path = path_texto,output_name=path_imagen)
    # mostrar imagen
    return Image.open(path_imagen)

## HIstogramas
def Histograma(data, bins, log, xlabel, titulo, path, color = 'blue'):
    """ Histograma(lista de valores, divisiones, logaritmo, xlabel, titulo de grafica, path salvar, color) -> Grafica
    Recibe todos los parametros de una gráfica, su título y path para salvar la
    grafica y regresa una gráfica Box PLot. Si el path es False no guradrá la 
    gráfica.
    """
    #tamaño
    plt.figure(figsize=(10, 7))
    #histograma
    plt.hist(data, bins = bins, log = log, color = color)
    #xlabel
    plt.xlabel(xlabel)
    #titulo
    plt.title(titulo)
    #salvar imagen
    if path != False:
        plt.savefig(path, dpi = 200)
    
    return plt.show()

def Box(data, xlabel, titulo, path):
    """ Box(lista de valores, titulo de grafica, path salvar, color) -> Grafica
    Recibe todos los parametros de una gráfica, su título y path para salvar la
    grafica y regresa una gráfica Box PLot. Si el path es False no guradrá la 
    gráfica.
    """
    plt.figure(figsize = (10, 7))
    #grafica de caja
    plt.boxplot(data)
    #xlabel
    plt.xlabel(xlabel)
    #titulo
    plt.title(titulo)
    #salvar imagen
    if path != False:
        plt.savefig(path, dpi = 200)
    
    return plt.show()

# fuera de la naturaleza de los datos
def out_nature(df, column, tipo):
    """out_nature(df, column, ['alpha', 'digit]) -> Lista de Valores diferentes
    Recibe una dataframe, columna y el tipo de datos que hay y esta regresa 
    una lista de los indices diferentes a la columna.
    """
    # guardar indices
    aux = []
    # tipo de variable
    if tipo == 'alpha':
        # solo valores no nulos
        for i in df.loc[:, column].dropna().index:
            # prueba alfabética
            if not df.loc[i, column].isalpha():
                aux.append(i)
                
    elif tipo == 'digit':
        # solo valores no nulos
        for i in df.loc[:, column].dropna().index:
            # prueba numérica
            try:
                float(df.loc[i, column])
            except:
                aux.append(i)
    # lista de índices diferentes        
    return aux

def IQR(df):
    """ IQR(data_frame sin target) -> lista de indices a eliminar, data frame con cantidades a eliminar
    La función recibe el data frame sin target y regresa una lista de indices a eliminar
    por el método de IQR y un data frame con estadísticas de valores eliminados.
    """
    # Obtenemos los cuantíles para las variables
    Q1 = df.quantile(.25) # se elimina el target
    Q3 = df.quantile(.75) # se elimina el target

    #IQR
    IQR = Q3 - Q1
    
    # Valores superiores e inferiores
    INF=Q1-1.5*(IQR)
    SUP=Q3+1.5*(IQR)
    
    # número de outliers por variable
    out_IQR = ((df<INF)|(df>SUP)).sum()[SUP.index]

    #outliers con respecto al total
    out_IQR_100 = (out_IQR/df.shape[0])*100

    # guardar lista de indices
    index_general = []
    # recorrer cada variables para obtener indices
    for variable in SUP.index:
        iqr_aux = (df[variable]<INF[variable])|(df[variable]>SUP[variable])
        index_general.append(list(iqr_aux[iqr_aux].index))
    
    # diccionario de variables y sus indices con outliers
    dict_IQR = dict(zip(SUP.index, index_general))
    
    # Outliers IQR
    IQR_df = pd.concat([out_IQR, out_IQR_100], axis = 1, join='inner')
    IQR_df.columns = ['n outliers IQR', 'n outliers IQR %']
    
    return (dict_IQR, IQR_df)

def percentiles(df):
    """percetiles(data_frame sin target) -> lista de indices a eliminar, data frame con cantidades a eliminar
    La función recibe el data frame sin target y regresa una lista de indices a eliminar
    por el método de Percentiles y un data frame con estadísticas de valores eliminados.
    """
    Q5 = df.quantile(.5) # eliminamos la varible objectivo
    Q95 = df.quantile(.95) # eliminamos la varible objectivo
    
    # número de outliers por variable
    out_Per = ((df<Q5)|(df>Q95)).sum()
    
    #outliers con respecto al total
    out_Per_100 = (out_Per / df.shape[0]) * 100
    
    # guardar lista de indices
    index_general = []
    # recorrer cada variables para obtener indices
    for variable in Q95.index:
        per_aux = (df[variable]<Q5[variable])|(df[variable]>Q95[variable])
        index_general.append(list(per_aux[per_aux].index))
    
    # diccionario de variables y sus indices con outliers
    dict_Per = dict(zip(Q95.index, index_general))
    
    # Outliers Percentiles
    Per_df = pd.concat([out_Per, out_Per_100], axis = 1, join='inner')
    Per_df.columns = ['n outliers Percentil', 'n outliers Percentil %']
    
    return (dict_Per, Per_df)

def z_score(df, columns):
    """percetiles(data_frame sin target, columnas a analizar) -> lista de indices a eliminar, data frame con cantidades a eliminar
    La función recibe el data frame sin target y una lista de columnas a anlizar regresa una 
    lista de indices a eliminar por el método Z Score y un data frame con estadísticas 
    de valores eliminados.
    """
    dict_Z = {}
    # iteraciones por variables númericas para z-score usando variables anteriores SUP
    for i in columns:
        # datos de variables numéricas
        data = df.loc[:, i]
        # z-score para valuación
        z = np.abs(stats.zscore(data))
        # indices de variables a eliminar
        indices_aux_z = np.where(z > 3)
        # agregar al diccionario
        dict_Z[i] = list(indices_aux_z[0])
        
    # número de outliers por variable
    out_Z = pd.Series(index = list(dict_Z.keys()), data = [len(x) for x in dict_Z.values()])
    
    #outliers con respecto al total
    out_Z_100 = (out_Z / df.shape[0]) * 100
    
    # Outliers IQR
    Z_df = pd.concat([out_Z, out_Z_100], axis = 1, join='inner')
    Z_df.columns = ['n outliers Z-Score', 'n outliers Z-Score %']
    
    return (dict_Z, Z_df)

def tabla_out(IQR_df, Per_df, Z_df, dict_IQR, dict_Per, dict_Z, df):
    """ tabla_out(IQR_df, Per_df, Z_df, dict_IQR, dict_Per, dict_Z, df) -> Tabla final con Outliers
    Regresa una tabla final organizada con estadísticas e indices con outliers
    """
    # union de IQR y Percentil
    df_outliers = pd.concat([IQR_df, Per_df], axis = 1, join='inner')
    # union de tabla y Z-Score
    df_outliers = pd.concat([df_outliers, Z_df], axis = 1, join='inner')
    
    # indices de registros en dos métodos
    aux_ind_out = []
    # iteración por variable
    for i in df_outliers.index:
        # variables que existen en dos métodos, todas las combinaciones
        a = list(set(dict_IQR[i]) & set(dict_Per[i]))
        b = list(set(dict_IQR[i]) & set(dict_Z[i]))
        c = list(set(dict_Z[i]) & set(dict_Per[i]))

        # extensión de valores y creación de lista única
        a.extend(b)
        a.extend(c)
        aux_ind_out.append(list(set(a)))
        
    #Agregar indices de outliers
    df_outliers['indices'] = aux_ind_out

    #total outliers
    df_outliers['total outliers'] = df_outliers['indices'].apply(lambda x: len(x)).values

    #porcentaje de outliers
    df_outliers['% outliers'] = (df_outliers['total outliers'] / df.shape[0]) * 100

    # nuevo indice
    df_outliers = df_outliers.reset_index()

    # cambio de nombre a variable index
    df_outliers.rename(columns = {'index': 'features'}, inplace = True)

    # acomodo de columnas
    df_outliers = df_outliers[['features', 'n outliers IQR', 'n outliers Percentil', 'n outliers Z-Score',
                          'n outliers IQR %', 'n outliers Percentil %', 'n outliers Z-Score %',
                          'total outliers', '% outliers', 'indices']]
    
    return df_outliers