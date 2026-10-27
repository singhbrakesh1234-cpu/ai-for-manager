# FreshKeep AI – AI Food Waste & Expiry Tracker

AI Foundations for Managers (Semester 1, 2026) – group project prototype. All data is synthetic.

**What it does**
- **Model 1 – waste risk:** a Random Forest predicts which pantry items are likely to be wasted (ROC-AUC 0.92) and sends a few ranked "use it up" alerts instead of a reminder for every date.
- **Model 2 – consumption forecast:** a Random Forest forecasts next week's use per item and builds a smart shopping list.
- **Use It Up:** recipes ranked by how much at-risk food they rescue.

Back-test on 400 simulated households: food waste down 40% (₹728 → ₹436 per home per month), grocery spend down ₹579 a month.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
`FoodWaste_Expiry_Tracker.ipynb` rebuilds all data and models from scratch (delete `data/` first to regenerate).

## Deploy on Streamlit Community Cloud
1. Upload this whole folder (including the hidden `.streamlit` folder) to a GitHub repository.
2. On share.streamlit.io → Create app → pick the repo, branch `main`, main file `app.py` (or `food-waste-tracker/app.py` if the folder sits inside a bigger repo).
3. Advanced settings → Python 3.12 → Deploy.

Keep `scikit-learn==1.7.2` – the saved models need it.
