# selección de idioma 
TEXT = {
    "app_title": {
        "Español": "📊 SAPEED - FESAc-UNAM",
        "English": "📊 SAPEED - FESAc-UNAM"
    },
    "app_subtitle": {
        "Español": "Plataforma exploratoria estructural y psicométrica para monitoreo institucional.",
        "English": "Exploratory structural and psychometric platform for institutional monitoring."
    },
    "app_description": {
        "Español": "Carga archivos por profesor con respuestas estudiantiles a nivel ítem para calcular confiabilidad, promedios institucionales y relaciones estructurales entre dimensiones.",
        "English": "Upload professor-level files with student item-level responses to compute reliability, institutional averages, and structural relationships across dimensions."
    },
    "upload_header":{
        "Español": "📂 Carga de archivos por profesor (Respuestas a nivel estudiante)",
        "English": "📂 Upload professor files (Student-level responses)"
    },
    "select_professor": {
        "Español": "Seleccionar Profesor",
        "English": "Select Professor"
    },
    "professor_selection_header": {
        "Español": "👤 Comparación de Profesor", 
        "English": "👤 Professor comparison"
    },
    "Institutional_Average_Header":{
    "Español": "🏛 Promedio Institucional (ponderado por estudiante)", 
    "English": "🏛 Institutional Average (weighted by student)"
    },
    "Professor_Summary_Header":{
        "Español": "📊 Resumen de Desempeño del Profesor", 
        "English": "📊 Professor Performance Summary"
    },
    "Bar_Comparison_Subheader": {
        "Español": "📊 Comparación en Barras",
        "English": "📊 Bar Comparison"
    },
    "Institutional_Average_text": {
        "Español": "Media Institucional",
        "English": "Institutional Average"
    },
    "Lower_Limit_95_Inst": {
        "Español": "Límite Inferior 95% Inst.",
        "English": "Lower Limit 95% Inst."
    },
    "Radar_Comparison_Subheader": {
        "Español": "🕸 Comparación Radar",
        "English": "🕸 Radar Comparison"
    },
    "Institutional_IC_95": {
        "Español": "IC 95% Institucional",
        "English": "Institutional IC 95%"
    },
    "radar_comparison_caption": {
        "Español": "La gráfica radar compara las medias del profesor seleccionado contra la media institucional. \n\nEl área sombreada representa el intervalo de confianza del 95% institucional.",
        "English": "The radar chart compares the selected professor's means against the institutional average. \n\nThe shaded area represents the institutional 95% confidence interval."
    },
    "Institucional_Dimension_Subheader": {
        "Español": "📋 Interpretación por Dimensión",
        "English": "📋 Interpretation by Dimension"
    },
    "caption_text_Criteria": {
        "Español": "Criterios de clasificación: - Superior -> la media del profesor supera el IC 95% institucional \n\n- Inferior -> la media del profesor está por debajo del IC 95% institucional \n\n- Dentro del rango -> sin diferencia estadísticamente relevante",
        "English": "Classification criteria: - Superior -> professor's mean exceeds institutional 95% CI \n\n- Inferior -> professor's mean is below institutional 95% CI \n\n- Within range -> no statistically relevant difference"
    },
    "Reliability_Header": {
        "Español": "🔬 Análisis de Confiabilidad",
        "English": "🔬 Reliability Analysis"
    },
    "Variance_text": {
        "Español": "Varianza",
        "English": "Variance"
    },
    "alpha_Stratified_metric": {
        "Español": "α Estratificada Institucional",
        "English": "Institutional Stratified α"
    },
    "student_responses_metric": {
        "Español": "Total de respuestas estudiantiles",
        "English": "Total student responses"
    },
    "alpha_Institutional":{
        "Español":"α Institucional",
        "English":"Institutional  α"
    },
    "Dimension_Reliability_Subheader": {
        "Español": "Confiabilidad por Dimensión",
        "English": "Reliability by Dimension"
    },
    "Ecological_Interpretation_Note": {
    "Español": """
    **Nota importante sobre interpretación ecológica:**
    - Las correlaciones a nivel profesor reflejan asociaciones entre promedios, no entre estudiantes individuales.
    - Pueden ser influenciadas por la variabilidad entre profesores y el tamaño de muestra de profesores.
    - No se deben interpretar como relaciones causales a nivel estudiante.
    """,
    "English": """
    **Important note on ecological interpretation:**
    - Professor-level correlations reflect associations between averages, not individual students.
    - They can be influenced by between-professor variability and professor sample size.
    - They should not be interpreted as causal relationships at the student level.
    """
    },
    "minimum_professors_for_correlation":{
        "Español": "Se requieren al menos 2 profesores para calcular correlación a nivel profesor.", 
        "English": "At least 2 professors are required to compute professor-level correlation."
    },
    "Selected_Professor_Correlation_Subheader": {
        "Español": "🔍 Correlación dentro del Profesor Seleccionado", 
        "English": "🔍 Correlation within Selected Professor"
    },
    "Regresion_Ridge_Description": {
        "Español": "🧠 Regresión Ridge + Validación Cruzada (Red de Dimensiones)", 
        "English": "🧠 Ridge Regression + Cross-Validation (Dimensional Network)"
    },

    "Cronbach_Interpretation_Caption":{
        "Español": "Guía general de interpretación de α de Cronbach:\n\n• < 0.60 → Baja consistencia interna  \n• 0.70 → Aceptable  \n• 0.80 → Buena  \n• 0.90+ → Muy alta (posible redundancia de ítems)",
        "English": "General guide for interpreting Cronbach's α:\n\n• < 0.60 → Low internal consistency  \n• 0.70 → Acceptable  \n• 0.80 → Good  \n• 0.90+ → Very high (possible item redundancy)"
    },
    "Interpreted_r_Student_Level_Description":{
        "Español": "Interpretación (nivel estudiante combinado):\n\n- Mide cómo co-varían las dimensiones considerando a **todos** los estudiantes.\n\n- r alta → estudiantes que califican alto en una dimensión tienden a calificar alto en otra.\n\n- r cercana a 0 → dimensiones relativamente independientes.\n\n- r negativa → patrón inverso en la percepción estudiantil.", 
        "English": "Interpretation (combined student level):\n\n- Measures how dimensions co-vary considering **all** students.\n\n- High r → students who rate high on one dimension tend to rate high on another.\n\n- r near 0 → relatively independent dimensions.\n\n- Negative r → inverse pattern in student perception."
    },
    "Ecol_level_Correlacion": {
        "Español": "Correlación entre profesores (nivel ecológico)",
        "English": "Correlation between professors (ecological level)"
    },
    "Corr_Analysis_Header":{
        "Español": "Análisis de Correlaciones", 
        "English": "Correlation Analysis"
    },
    "Inst_Corre_Stud_Combi":{
        "Español": "Correlación institucional considerando a todos los estudiantes combinados (no promedios por profesor).",
        "English": "Institutional correlation considering all students combined (not professor averages)."
    },
    "Interpretation_Lambda_caption":{
        "Español":"Interpretación: λ* controla la contracción (fuerza de regularización). \n\nMSE_CV(λ*) estima el error de predicción fuera de muestra (menor es mejor).\n\nValores muy altos de λ* → modelo muy simple (coeficientes cercanos a 0).\n\nValores muy bajos de λ* → modelo más complejo (coeficientes menos penalizados).", 
        "English":"    Interpretation: λ* controls the shrinkage (strength of regularization). \n\nMSE_CV(λ*) estimates out-of-sample prediction error (lower is better).\n\nVery high λ* values → very simple model (coefficients close to 0).\n\nVery low λ* values → more complex model (less penalized coefficients)."
    },
    "Sample_Size_Warning":{
        "Español": "⚠️ Tamaño de muestra bajo puede producir estimaciones inestables (recomendado N ≥ 30).", 
        "English": "⚠️ Low sample size may produce, unstable estimates (recommended N ≥ 30)."
    },
    "K_Values_Slider":{
        "Español": "Pliegues de validación cruzada (K)", 
        "English": "Cross-validation folds (K)"
    },
    "sigma_1n_CV_MSE":{
        "Español":"-1 desv. estándar (CV)", 
        "English":"-1 std dev (CV)"
    },
    "sigma_1_CV_MSE":{
        "Español":"+1 desv. estándar (CV)", 
        "English":"+1 std dev (CV)"
    },
    "CV_Aver_MSE":{
        "Español":"MSE promedio (CV)", 
        "English":"CV Average MSE"
    },
    "RegVsLambda_Title":{
        "Español":"Curva de Regularización (λ vs Error CV)", 
        "English":"Regularization Curve (λ vs CV Error)"        
    },
    "Lambda_Esc_Log_Title":{
        "Español":"λ (escala log)", 
        "English":"λ (log scale)"
    },
    "Erro_Cuad_Avg_CV":{
        "Español":"Error cuadrático medio promedio (CV)", 
        "English":"Average Mean Squared Error (CV)"
    },
    "Alpha_PsiVsLambdaRidge":{
        "Español":"ℹ️ Aclaración de parámetros: α psicométrica vs λ de Ridge", 
        "English":"ℹ️ Clarification of parameters: Psychometric α vs Ridge λ"
    },
    "expander_Text":{
        "Español":"**α psicométrica (Cronbach / α estratificada)**\n\n- Mide la confiabilidad del instrumento (consistencia interna).  \n\n- Se calcula a partir de la estructura de covarianzas entre ítems.\n\n**λ de Ridge (fuerza de regularización)**  \n\n- Hiperparámetro que controla la contracción de coeficientes en regresión Ridge.  \n\n- En scikit-learn se llama `alpha`, pero matemáticamente corresponde a λ.\n\n**En este dashboard:**  \n\n- α → confiabilidad (calidad del instrumento)  \n\n- λ → regularización del modelo (modelado predictivo/estructural)",
        "English":"**Psychometric α (Cronbach / stratified α)**\n\n- Measures instrument reliability (internal consistency).  \n\n- Calculated from the covariance structure among items.\n\n**Ridge λ (regularization strength)**  \n\n- Hyperparameter that controls coefficient shrinkage in Ridge regression.  \n\n- In scikit-learn it's called `alpha`, but mathematically corresponds to λ.\n\n**In this dashboard:**  \n\n- α → reliability (instrument quality)  \n\n- λ → model regularization (predictive/structural modeling)" 
    },
    "BetaVsAlpha_Subheader":{
        "Español":"Trayectorias de estabilidad (β vs λ)", 
        "English":"Coefficient Stability Trajectories (β vs λ)"
    },
    "FP_UL_text_Title":{
        "Español":"Estabilidad de coeficientes a través de λ", 
        "English":"Coefficient Stability across λ"
    },
    "Lambda_XAxis_Title":{
        "Español":"λ (escala log)", 
        "English":"λ (log scale)"
    },
    "Beta_YAxis_Title":{
        "Español":"β promedio (entre pliegues)", 
        "English":"Average β (across folds)"
    },
    "predictors_Text":{
        "Español":"Predictores",
        "English":"Predictors"
    },
    "Institutional_Dimension_Subheader": {
        "Español": "📋 Interpretación por Dimensión",
        "English": "📋 Interpretation by Dimension"
    }
}