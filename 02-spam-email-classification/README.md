# Spam Email Classification

**Course:** Data 100, Principles and Techniques of Data Science (UC Berkeley, Summer 2024), Projects B1 and B2

**Question:** How well can a logistic regression model built on hand-engineered text features separate spam from legitimate ("ham") email?

## Data
About 7,500 labeled training emails (subject and body) and a 1,000-email unlabeled test set.

## What I did
**Part 1: baseline classifier**
- Wrote a function that turns emails into binary "does this word appear" features, then trained a first logistic regression model.
- Computed accuracy, precision, recall and false positive rate by hand (no library metric functions) and compared them with a "zero predictor" that labels everything ham.

**Part 2: building my own model**
- Explored the data with a **correlation heatmap** of engineered features, a chart of the most common words that appear only in spam subjects, and a word-count box plot for spam vs. ham.
- Engineered features with Python and **regular expressions**: counts of HTML tags, promotional words ("free", "discount") and scam words ("winner", "guaranteed"), words that appear only in spam subjects, and presence flags for strong signal words. I also tested ideas that didn't work, such as an email-length threshold, and dropped them.
- Trained an **L2-regularized logistic regression** model, plotted its **ROC curve** to show the trade-off between catching spam and flagging real mail, and analyzed ambiguous emails where the "ground truth" label itself is debatable.

## Results
- The baseline reached only **75.8%** training accuracy, barely better than the 74.5% you get by calling everything ham, and it caught just 11% of spam.
- My engineered-feature model reached **86.5%** training accuracy.

## What I learned
- Accuracy alone is misleading on imbalanced data, so precision, recall and the ROC curve tell the real story.
- Feature quality beats feature count. Combining several related "money" words added noise, while a few distinctive spam-only signals helped.
- Labels come from people, so ambiguous examples should change how much you trust an evaluation metric.

## Skills and tools
Python, pandas, NumPy, scikit-learn (LogisticRegression, roc_curve), regular expressions, text feature engineering, EDA, Seaborn heatmaps, classification metrics (accuracy, precision, recall, FPR), ROC analysis, label-quality analysis.
