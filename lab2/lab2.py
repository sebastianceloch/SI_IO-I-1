import gradio as gr
import pandas as pd
import random
import time


def plt(csv,rows,queries=""):
    df = pd.read_csv(csv)
    df.dropna(axis=0, inplace=True)
    df = df.head(int(rows))
    class_shape_int = df.shape
    class_count = df.iloc[:, -1].value_counts()
    if queries == "":
        describe = df.describe()
        class_shape = f"Liczba atrybutow: {class_shape_int[1]}, liczba obiektow: {class_shape_int[1]*class_shape_int[0]}"
        return class_shape, df, describe
    if queries == "ile klas decyzyjnych":
        query =  "ile klas decyzyjnych"
        result = f"Liczba klas decyzyjnych: {len(class_count)}"
        return query, df, result
    if queries == "wielkość klasy decyzyjnej":
        query = "wielkość klasy decyzyjnej"
        result = f"wielkości klas decyzyjnych: {class_count.to_dict()}"
        return query, df, result

inputs = [gr.Textbox(label="CSV File"), 
          gr.Number(label="Number of Rows"),
          gr.inputs.Dropdown(["ile klas decyzyjnych", "wielkość klasy decyzyjnej"], label="Pytanie")]
outputs = [gr.outputs.Textbox(label="Wybrane pytanie"),
           gr.outputs.Dataframe(label="Tabela",type='pandas'),
           gr.outputs.Textbox(label="Wynik")]

gr.Interface(plt, inputs=inputs, outputs=outputs).launch()
