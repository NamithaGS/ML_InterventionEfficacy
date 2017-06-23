# Intervention Efficacy for clinicians
Modeling using supervised learning methods to predict the success of interventions for clinicians on the survey data.


- To predict efficacy or success of interventions like suggested alternatives, accountable justification and peer comparison on prescription of antibiotics by physicians.
- Interventions to physicians decisions has an effect on the unnecessary antibiotics prescriptions, which are a cause for lot of side effects and expensive treatements and lead to economical and behavioural problems.
- The dataset collected by researchers from the Keck School of Medicine at USC consists of 248 Clinicians, with around 1866 visits for Acute Respiratory Tract infections. 
- The data was then pruned, imputed and feature selection techniques ( LVQ and RFE) were used to understand the importance of each feature..
Clinicians and/or patient characteristics and features that are top indicators of effective interventions are extracted.
- Stochastic Gradient Boosted Trees and Neural Networks performed particularly well in terms of being able to classify the intervention as highly effective or not effective given clinician and patient characteristics as well as survey data.
- Regression methods such as Generalized Additive Models and Linear Regression work well with log transformations of the mean-effect size outcome.
