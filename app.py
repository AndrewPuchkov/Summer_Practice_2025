import pandas as pd
import streamlit as st
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, recall_score
import seaborn as sns
import joblib
import  matplotlib.pyplot as plt
import shap
import json
import numpy as np
import plotly.express as px
import statsmodels.api as sm
import xgboost as xgb
def main():
    st.set_page_config(layout="wide", page_title="Анализ качества яблок")
    data_fruit = pd.read_csv("quotes3.csv")

    def load_model():
        model = joblib.load('apple_quality_model.pkl')
        return model

    def load_metrics():
        metrics_df = pd.read_csv('metrics.csv')
        return metrics_df
    metrics = load_metrics()
    model = load_model()

    def load_test_x():
        with open('test_data.json', 'r') as f:
            test_data = json.load(f)
        X_test = pd.DataFrame(test_data["X_test"])
        return X_test
    
    def load_test_y():
        with open('test_data.json', 'r') as f:
            test_data = json.load(f)
        y_test = pd.DataFrame(test_data["y_test"])
        return y_test
    
    def load_pred_y():
        with open('test_data.json', 'r') as f:
            test_data = json.load(f)
        y_pred = pd.DataFrame(test_data["y_pred"])
        return y_pred
    
    def load_features():
        with open('model_metadata2.json', 'r') as f:
            test_data = json.load(f)
        return test_data["features"] 

    features = load_features()
    X_test = load_test_x()
    y_test = load_test_y()
    y_pred = load_pred_y()

    col1,col2,col3 = st.columns([7,7,7])

    with col2:
        st.title("Фруктовый ниндзя")
    st.title("Пучков Андрей Андреевич 2023-ФГиИБ-ПИ-1б 17 Вариант")
    st.write("Цель моей работы: на основе характеристик яблок анализировать их качество и выявить, какие хар-ки являются более значимыми при оценке.")
    st.write("У меня есть 7 признаков:")
    st.write("""
    - A_id: Уникальный идентификатор каждого фрукта  
    - Size: Размер фрукта  
    - Weight: Вес фрукта  
    - Sweetness: Степень сладости фрукта  
    - Crunchiness: Текстура, указывающая на хрусткость фрукта  
    - Juiciness: Уровень сочности фрукта  
    - Ripeness: Стадия зрелости фрукта  
    - Acidity: Уровень кислотности фрукта  
    - Quality: Общая оценка качества фрукта  
    """)

    tab1, tab2, tab3, tab4 = st.tabs(["Исходные данные", "Графики зависимостей", "Матрица ошибок и метрики модели",  "Интерпретация результатов обучения модели"])

    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            st.subheader("SHAP-значения признаков")
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
            st.pyplot(fig, bbox_inches='tight')
            plt.close()
            st.write('На графике видно, что больше всего на качество яблок влияют такие признаки размер, потом сочность, сладость, зрелость')
        
        with col2:
            st.subheader("SHAP-анализ")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            plt.figure(figsize=(6, 4))
            shap.summary_plot(
                shap_values, 
                X_test, 
                feature_names=features,
                plot_type="dot",
                show=False,
                max_display=min(20, len(features)))
            plt.tight_layout()
            st.pyplot(plt.gcf())
            plt.close()
            st.write('Интерпретация:')
            st.write('- Красные точки: высокие значения признака увеличивают вероятность класса "good"')
            st.write('- Синие точки: низкие значения уменьшают вероятность класса "good"')

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.header("1. Распределение признаков")
            feature = st.selectbox('Выберите признак:', ['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Ripeness', 'Acidity', 'Density'])
            color_scale = alt.Scale(
                domain=['good', 'bad'],
                range=['green','red']
            )
            sorted_data = data_fruit.sort_values('Quality', ascending=[False])
            hist = alt.Chart(sorted_data).mark_bar(
                opacity=0.5, 
                binSpacing=0, stroke='black',
            strokeWidth=0.5
            ).encode(
                alt.X(f'{feature}:Q').bin(maxbins=50),
                alt.Y('count()').stack(None),
                alt.Color('Quality:N', scale=color_scale, 
                        legend=alt.Legend(title="Качество")),
                tooltip=['count()', 'Quality'],
                order=alt.Order('Quality', sort='descending')
            ).properties(
                width=600,
                height=400
            ).interactive()
            st.altair_chart(hist, use_container_width=True)

        with col2:
            st.header("2. Зависимость сочности от зрелости")
            x_axis = st.selectbox(
            'Выберите признак для оси X:',
            ['Size', 'Weight', 'Sweetness', 'Crunchiness', 
             'Juiciness', 'Ripeness', 'Acidity', 'Density'],
             index=5,
            key='x_axis'
        )
            y_axis = st.selectbox(
                    'Выберите признак для оси Y:',
                    ['Size', 'Weight', 'Sweetness', 'Crunchiness', 
                    'Juiciness', 'Ripeness', 'Acidity', 'Density'],
                    index=4, 
                    key='y_axis'
                )
            
            st.write(f"**Зависимость {y_axis} от {x_axis}**")
            
            fig = px.scatter(
                data_fruit,
                x=x_axis,
                y=y_axis,
                color='Quality',
                color_discrete_map={'good': 'blue', 'bad': 'red'},
                trendline="lowess",
                trendline_options=dict(frac=0.3),
                width=800,
                height=500
            )
            
            fig.update_layout(
                xaxis_title=f"{x_axis}",
                yaxis_title=f"{y_axis}",
                legend_title="Качество",
                hovermode='closest'
            )
            
            fig.update_traces(
                line=dict(width=4),
                marker=dict(size=1, opacity=0.5)
            )
            st.plotly_chart(fig, use_container_width=True)


    with tab3:
        final_matrix = confusion_matrix(y_test, y_pred)

        fig = px.imshow(final_matrix,
                   labels=dict(x="Предсказано", y="Истинное", color="Count"),
                   x=['Плохие', 'Хорошие'],
                   y=['Плохие', 'Хорошие'],
                   text_auto=True,
                   color_continuous_scale='Greens')
        fig.update_layout(title='Матрица ошибок')
        st.plotly_chart(fig)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Тестовая Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        
        with col2:
            st.metric("Test F1-score", f"{f1_score(y_test, y_pred):.4f}")
        
        with col3:
            st.metric("recall", f"{recall_score(y_test, y_pred):.4f}")



    with tab1:
        st.dataframe(data_fruit)

    return data_fruit


if __name__ == '__main__':
    main()


