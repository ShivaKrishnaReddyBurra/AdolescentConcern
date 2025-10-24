import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from sklearn.preprocessing import LabelEncoder
import base64
from flask import Flask, render_template, request
import shap
import warnings
import numpy as np

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load data and model
data = pd.read_csv('adolescent_concern_dataset.csv')
stacking_clf = joblib.load('model.pkl')

# Encode for consistency
categorical_cols = ['Gender', 'Parental_Education', 'Ad_Exposure_Freq', 'Health_Knowledge', 'Parental_Influence', 'Concern_Level']
label_encoders = {col: LabelEncoder().fit(data[col]) for col in categorical_cols}
data_encoded = data.copy()
for col in categorical_cols:
    data_encoded[col] = label_encoders[col].transform(data[col])

X = data_encoded.drop('Concern_Level', axis=1)

# SHAP explainer
rf = stacking_clf.estimators_[0]
explainer = shap.Explainer(rf, X)
shap_values = explainer(X)

# Feature importances
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

app = Flask(__name__)

def get_plot_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white', edgecolor='none')
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
    
    filtered = data_encoded[(data_encoded['Age'] >= age_min) & (data_encoded['Age'] <= age_max)]
    filtered_original = data[(data['Age'] >= age_min) & (data['Age'] <= age_max)]
    
    if gender != 'All':
        gender_encoded = label_encoders['Gender'].transform([gender])[0]
        filtered = filtered[filtered['Gender'] == gender_encoded]
        filtered_original = filtered_original[filtered_original['Gender'] == gender]
    
    # 1. Concern Level Distribution (Enhanced Donut Chart)
    fig1, ax1 = plt.subplots(figsize=(10, 6), facecolor='white')
    concern_counts = filtered_original['Concern_Level'].value_counts()
    colors = ['#FF6B6B', '#4ECDC4', '#FFD93D']
    wedges, texts, autotexts = ax1.pie(concern_counts, labels=concern_counts.index, autopct='%1.1f%%',
                                         startangle=90, colors=colors, textprops={'fontsize': 12, 'weight': 'bold'},
                                         wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax1.add_artist(centre_circle)
    ax1.set_title('Concern Level Distribution', fontsize=16, weight='bold', pad=20)
    concern_plot = get_plot_base64(fig1)
    plt.close(fig1)
    
    # 2. Feature Importances (Horizontal Bar)
    fig2, ax2 = plt.subplots(figsize=(10, 6), facecolor='white')
    top_features = importances.head(8)
    colors_gradient = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    bars = ax2.barh(range(len(top_features)), top_features.values, color=colors_gradient, edgecolor='white', linewidth=1.5)
    ax2.set_yticks(range(len(top_features)))
    ax2.set_yticklabels(top_features.index, fontsize=11, weight='bold')
    ax2.set_xlabel('Importance Score', fontsize=12, weight='bold')
    ax2.set_title('Top Feature Importances', fontsize=16, weight='bold', pad=20)
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width + 0.001, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                ha='left', va='center', fontsize=10, weight='bold')
    importances_plot = get_plot_base64(fig2)
    plt.close(fig2)
    
    # 3. Age Distribution by Concern Level
    fig3, ax3 = plt.subplots(figsize=(10, 6), facecolor='white')
    for concern in filtered_original['Concern_Level'].unique():
        ages = filtered_original[filtered_original['Concern_Level'] == concern]['Age']
        ax3.hist(ages, bins=range(13, 20), alpha=0.7, label=concern, edgecolor='white', linewidth=1.5)
    ax3.set_xlabel('Age', fontsize=12, weight='bold')
    ax3.set_ylabel('Frequency', fontsize=12, weight='bold')
    ax3.set_title('Age Distribution by Concern Level', fontsize=16, weight='bold', pad=20)
    ax3.legend(fontsize=11, framealpha=0.9)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    age_dist_plot = get_plot_base64(fig3)
    plt.close(fig3)
    
    # 4. Gender vs Concern Level
    fig4, ax4 = plt.subplots(figsize=(10, 6), facecolor='white')
    gender_concern = pd.crosstab(filtered_original['Gender'], filtered_original['Concern_Level'])
    gender_concern.plot(kind='bar', ax=ax4, color=['#FF6B6B', '#4ECDC4', '#FFD93D'], 
                        edgecolor='white', linewidth=1.5, width=0.7)
    ax4.set_xlabel('Gender', fontsize=12, weight='bold')
    ax4.set_ylabel('Count', fontsize=12, weight='bold')
    ax4.set_title('Concern Level by Gender', fontsize=16, weight='bold', pad=20)
    ax4.legend(title='Concern Level', fontsize=10, title_fontsize=11, framealpha=0.9)
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    gender_plot = get_plot_base64(fig4)
    plt.close(fig4)
    
    # 5. Ad Exposure Impact
    fig5, ax5 = plt.subplots(figsize=(10, 6), facecolor='white')
    ad_concern = pd.crosstab(filtered_original['Ad_Exposure_Freq'], filtered_original['Concern_Level'], normalize='index') * 100
    ad_concern.plot(kind='bar', stacked=True, ax=ax5, color=['#FF6B6B', '#4ECDC4', '#FFD93D'], 
                    edgecolor='white', linewidth=1.5, width=0.7)
    ax5.set_xlabel('Ad Exposure Frequency', fontsize=12, weight='bold')
    ax5.set_ylabel('Percentage (%)', fontsize=12, weight='bold')
    ax5.set_title('Impact of Ad Exposure on Concern Level', fontsize=16, weight='bold', pad=20)
    ax5.legend(title='Concern Level', fontsize=10, title_fontsize=11, framealpha=0.9)
    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=0)
    ax5.grid(axis='y', alpha=0.3, linestyle='--')
    ad_exposure_plot = get_plot_base64(fig5)
    plt.close(fig5)
    
    # 6. Correlation Heatmap
    fig6, ax6 = plt.subplots(figsize=(12, 8), facecolor='white')
    correlation = filtered[X.columns.tolist() + ['Concern_Level']].corr()
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    sns.heatmap(correlation, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax6)
    ax6.set_title('Feature Correlation Matrix', fontsize=16, weight='bold', pad=20)
    correlation_plot = get_plot_base64(fig6)
    plt.close(fig6)
    
    # Statistics
    total_records = len(filtered)
    concern_distribution = filtered_original['Concern_Level'].value_counts().to_dict()
    avg_age = filtered['Age'].mean()
    
    # Additional insights
    high_concern_pct = (concern_distribution.get('High', 0) / total_records * 100) if total_records > 0 else 0
    
    return render_template('dashboard.html', 
                           concern_plot=concern_plot, 
                           importances_plot=importances_plot,
                           age_dist_plot=age_dist_plot,
                           gender_plot=gender_plot,
                           ad_exposure_plot=ad_exposure_plot,
                           correlation_plot=correlation_plot,
                           age_min=age_min, age_max=age_max, gender=gender,
                           total_records=total_records,
                           concern_distribution=concern_distribution,
                           avg_age=round(avg_age, 1),
                           high_concern_pct=round(high_concern_pct, 1))

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
        
        # SHAP explanation
        shap_instance = explainer(input_data)
        shap_exp = [(X.columns[i], shap_instance.values[0, i, pred]) for i in range(len(X.columns))]
        shap_exp = sorted(shap_exp, key=lambda x: abs(x[1]), reverse=True)[:5]
        
        # Probability Chart (Enhanced)
        fig1, ax1 = plt.subplots(figsize=(10, 6), facecolor='white')
        colors_bar = ['#FF6B6B' if cn == 'High' else '#4ECDC4' if cn == 'Low' else '#FFD93D' for cn in class_names]
        bars = ax1.bar(class_names, proba * 100, color=colors_bar, edgecolor='white', linewidth=2, width=0.6)
        ax1.set_title('Prediction Probabilities', fontsize=16, weight='bold', pad=20)
        ax1.set_ylabel('Probability (%)', fontsize=12, weight='bold')
        ax1.set_ylim(0, 110)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        for i, (bar, v) in enumerate(zip(bars, proba * 100)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{v:.1f}%', ha='center', va='bottom', fontsize=12, weight='bold')
            if class_names[i] == prediction:
                ax1.text(bar.get_x() + bar.get_width()/2., height/2,
                        '✓', ha='center', va='center', fontsize=30, color='white', weight='bold')
        proba_plot = get_plot_base64(fig1)
        plt.close(fig1)
        
        # SHAP Waterfall Plot
        fig2, ax2 = plt.subplots(figsize=(10, 6), facecolor='white')
        features = [item[0] for item in shap_exp]
        values = [item[1] for item in shap_exp]
        colors_shap = ['#4ECDC4' if v > 0 else '#FF6B6B' for v in values]
        bars = ax2.barh(range(len(features)), values, color=colors_shap, edgecolor='white', linewidth=1.5)
        ax2.set_yticks(range(len(features)))
        ax2.set_yticklabels(features, fontsize=11, weight='bold')
        ax2.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=12, weight='bold')
        ax2.set_title('Feature Impact Analysis (SHAP)', fontsize=16, weight='bold', pad=20)
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        ax2.invert_yaxis()
        for i, (bar, val) in enumerate(zip(bars, values)):
            width = bar.get_width()
            ax2.text(width + (0.001 if width > 0 else -0.001), bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', ha='left' if width > 0 else 'right', va='center', 
                    fontsize=10, weight='bold')
        shap_plot = get_plot_base64(fig2)
        plt.close(fig2)
        
        return render_template('predict.html', 
                             prediction=prediction, 
                             confidence=confidence,
                             shap_exp=shap_exp,
                             proba_plot=proba_plot,
                             shap_plot=shap_plot,
                             input_summary={
                                 'Age': age,
                                 'Gender': gender_input,
                                 'Parental Education': parental_education,
                                 'Ad Exposure': ad_exposure_freq,
                                 'Celebrity Ads': 'Yes' if ad_type_celebrity else 'No',
                                 'Free Toys Ads': 'Yes' if ad_type_free_toys else 'No',
                                 'Cartoon Ads': 'Yes' if ad_type_cartoon else 'No',
                                 'Health Knowledge': health_knowledge,
                                 'Parental Influence': parental_influence
                             })
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)