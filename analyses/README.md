## Hypothesis 1 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ie_b3nADIIvN6W9uaqFo1m5GuQYRzlrQ?usp=sharing)

**"There exists a 'critical reflection' threshold such that, we can see a statistically-significant difference
in the performance of the students."** 

- We collect the reflective journals of the candidate teachers from practical sessions, 
  associated with their evaluation on the associated session 
- We empirically select a threshold used to split the diaries into two groups
- We identify reflective sentences in the diaries in each of the two groups using our Neural Classifier, 
  and measure a relative reflectivity within each of the diaries: 
  **relative reflectivity = reflective sentences / all sentences**
- To see whether the performance of the two groups match, we perform a t-test on a significance level = 0.95
- Additionally, we repeat the process for each type of reflection

## Hypothesis 2 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eMjo-7gmjom1lF6r3MXUgX3wDV9ATSpG?usp=sharing)

**"The number of categories in reflective diaries is moderated by the ordering of submission of reflective diaries."**

- The second hypothesis (H2) is connected with results, that only individual and personalized supporting in writing reflection leads to better results in reflective writings Spalding and Wilson (2002, p. 1393). In our setting, we did not give personalized feedback to reflective diaries' writings. Based on this, we supposed hypothesis 2.

**Methodology:**
- Create prediction of categories of sentences for each participant's diary.
- Using group.by for overall numbers of each category for each participant's diary.
- We used Generalized linear mixed models (GLMMs) (Jiang, 2017; Stroup, 2012; Faraway, 2016) for each category as outcome and ordering of submission of reflective diaries as predictor.
- Calculation: We using Python language and R language (R Core Team, 2018), specifically: the lme4 package (Bates et al., 2015), glmmTMB package (Brooks et al., 2017) and DHARMa package for residual diagnostic (Hartig, 2019).
