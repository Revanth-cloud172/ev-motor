# EV Motor Digital Twin & AI/ML Monitoring (Software-only)

This educational Streamlit prototype simulates motor telemetry, compares speed with a simple Digital Twin baseline, and uses Isolation Forest to flag unusual patterns.

## Run on a laptop
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Run:
   `python -m venv .venv`
4. Activate the environment:
   - Windows: `.venv\Scripts\activate`
   - macOS/Linux: `source .venv/bin/activate`
5. Install dependencies:
   `pip install -r requirements.txt`
6. Start:
   `streamlit run app.py`

## Run online
Upload `app.py` and `requirements.txt` to a GitHub repository, then deploy the repository using Streamlit Community Cloud. You can also run the code locally first.

## Important limitations
- All data is simulated; there is no physical motor or STM32 connection.
- The Digital Twin is a simple educational baseline, not a physics-validated motor model.
- Isolation Forest flags unusual data; it does not diagnose a particular failure.
- Scores on synthetic data are demonstration metrics only and do not establish real-world accuracy.
- This is not a safety system and must not control a real motor.
