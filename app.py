import joblib
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.preprocessing import LabelEncoder
import base64
from flask import Flask, render_template, request
import shap
import warnings

warnings.filterwarnings('ignore')

# Load data and model
data = pd.read_csv('dummy_adolescent_concern_dataset.csv')
stacking_clf = joblib.load('model.pkl')

# Encode for consistency
categorical_cols = ['Gender', 'Parental_Education', 'Ad_Exposure_Freq', 'Health_Knowledge', 'Parental_Influence', 'Concern_Level']
label_encoders = {col: LabelEncoder().fit(data[col]) for col in categorical_cols}
for col in categorical_cols:
    data[col] = label_encoders[col].transform(data[col])

X = data.drop('Concern_Level', axis=1)

# SHAP explainer (using Random Forest from stacking for simplicity)
rf = stacking_clf.estimators_[0]  # First estimator (RandomForest)
explainer = shap.Explainer(rf, X)
shap_values = explainer(X)

# Feature importances
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

app = Flask(__name__)

def get_plot_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route('/', methods=['GET', 'POST'])
def dashboard():
    age_min, age_max = 13, 18
    gender = 'All'
    
    if request.method == 'POST':
        age_min = int(request.form.get('age_min', 13))
        age_max = int(request.form.get('age_max', 18))
        gender = request.form.get('gender', 'All')
    
    filtered = data[(data['Age'] >= age_min) & (data['Age'] <= age_max)]
    if gender != 'All':
        filtered = filtered[filtered['Gender'] == label_encoders['Gender'].transform([gender])[0]]
    
    # Concern distribution plot with better styling
    fig, ax = plt.subplots(figsize=(10, 6))
    concern_counts = filtered['Concern_Level'].value_counts()
    colors = ['#e74c3c', '#3498db', '#f39c12']
    concern_counts.plot(kind='bar', ax=ax, color=colors, edgecolor='white', linewidth=2)
    ax.set_title('Concern Level Distribution', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Concern Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8f9fa')
    fig.tight_layout()
    concern_plot = get_plot_base64(fig)
    plt.close(fig)
    
    # Feature importances plot with better styling
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_imp = plt.cm.viridis(importances.values / importances.values.max())
    importances.plot(kind='barh', ax=ax, color=colors_imp, edgecolor='white', linewidth=1.5)
    ax.set_title('Global Feature Importances', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Features', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8f9fa')
    fig.tight_layout()
    importances_plot = get_plot_base64(fig)
    plt.close(fig)
    
    # Data head as HTML table
    data_head = data.head(10).to_html(classes='table table-hover', index=False)
    
    # Statistics
    total_records = len(filtered)
    concern_distribution = filtered['Concern_Level'].value_counts().to_dict()
    
    return render_template('dashboard.html', 
                           data_head=data_head, 
                           concern_plot=concern_plot, 
                           importances_plot=importances_plot,
                           age_min=age_min, age_max=age_max, gender=gender,
                           total_records=total_records,
                           concern_distribution=concern_distribution)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form['age'])
        gender_input = request.form['gender']
        parental_education = request.form.get('parental_education', 'Bachelor')
        ad_exposure_freq = request.form.get('ad_exposure_freq', 'Medium')
        ad_type_celebrity = int(request.form.get('ad_type_celebrity', 0))
        ad_type_free_toys = int(request.form.get('ad_type_free_toys', 0))
        ad_type_cartoon = int(request.form.get('ad_type_cartoon', 0))
        health_knowledge = request.form.get('health_knowledge', 'Medium')
        parental_influence = request.form.get('parental_influence', 'Medium')
        
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [label_encoders['Gender'].transform([gender_input])[0]],
            'Parental_Education': [label_encoders['Parental_Education'].transform([parental_education])[0]],
            'Ad_Exposure_Freq': [label_encoders['Ad_Exposure_Freq'].transform([ad_exposure_freq])[0]],
            'Ad_Type_Celebrity': [ad_type_celebrity],
            'Ad_Type_Free_Toys': [ad_type_free_toys],
            'Ad_Type_Cartoon': [ad_type_cartoon],
            'Health_Knowledge': [label_encoders['Health_Knowledge'].transform([health_knowledge])[0]],
            'Parental_Influence': [label_encoders['Parental_Influence'].transform([parental_influence])[0]]
        })
        
        pred = stacking_clf.predict(input_data)[0]
        proba = stacking_clf.predict_proba(input_data)[0]
        class_names = ['High', 'Low', 'Medium']
        prediction = class_names[pred]
        confidence = proba[pred] * 100
        
        # SHAP explanation for this instance
        shap_instance = explainer(input_data)
        shap_exp = [(X.columns[i], shap_instance.values[0, i, pred]) for i in range(len(X.columns))]
        shap_exp = sorted(shap_exp, key=lambda x: abs(x[1]), reverse=True)[:5]
        
        # Create probability chart
        fig, ax = plt.subplots(figsize=(8, 5))
        colors_bar = ['#e74c3c' if cn == prediction else '#95a5a6' for cn in class_names]
        ax.bar(class_names, proba * 100, color=colors_bar, edgecolor='white', linewidth=2)
        ax.set_title('Prediction Probabilities', fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel('Probability (%)', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_facecolor('#f8f9fa')
        for i, v in enumerate(proba * 100):
            ax.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
        fig.tight_layout()
        proba_plot = get_plot_base64(fig)
        plt.close(fig)
        
        return render_template('predict.html', 
                             prediction=prediction, 
                             confidence=confidence,
                             shap_exp=shap_exp,
                             proba_plot=proba_plot,
                             input_summary={
                                 'Age': age,
                                 'Gender': gender_input,
                                 'Parental Education': parental_education,
                                 'Ad Exposure': ad_exposure_freq,
                                 'Health Knowledge': health_knowledge,
                                 'Parental Influence': parental_influence
                             })
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)